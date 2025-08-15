import torch
import torch.nn as nn
import torch.nn.functional as F
 

class GraphSAGELayer(nn.Module):
    def __init__(self, out_features, aggregator_type='mean', assume_row_normalized=True):
        super(GraphSAGELayer, self).__init__()
        self.out_features = out_features
        self.aggregator_type = aggregator_type
        self.assume_row_normalized = assume_row_normalized

        self.linear = None  # Lazy: nn.Linear(2*C, out_features) 在首次 forward 时创建

    def _build_linear(self, two_c, device, dtype):
        self.linear = nn.Linear(two_c, self.out_features, bias=True).to(device=device, dtype=dtype)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def mean_aggregator(self, adj_matrix, features):
        # adj_matrix: [B,N,N], features: [B,N,C] -> [B,N,C]
        support = torch.matmul(adj_matrix, features)
        if not self.assume_row_normalized:
            deg = adj_matrix.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            support = support / deg
        return support

    def pooling_aggregator(self, adj_matrix, features):
        # Max-pooling over neighbors with mask
        B, N, C = features.shape
        feat_nb = features.unsqueeze(1).expand(B, N, N, C)   # [B,N,N,C]
        mask = adj_matrix.unsqueeze(-1) > 0                  # [B,N,N,1]
        neg_inf = torch.finfo(features.dtype).min
        feat_nb = feat_nb.masked_fill(~mask, neg_inf)
        agg, _ = feat_nb.max(dim=2)                          # [B,N,C]
        agg[~torch.isfinite(agg)] = 0
        return F.relu(agg)

    def forward(self, adj_matrix, features):
        """
        adj_matrix: [B, N, N]
        features:   [B, N, C]
        return:     [B, N, out_features]
        """
        target_dtype = features.dtype 
        if self.linear is not None:
            target_dtype = self.linear.weight.dtype
        features = features.to(dtype=target_dtype)
        adj_matrix = adj_matrix.to(dtype=target_dtype)

        if self.aggregator_type == 'mean':
            neigh = self.mean_aggregator(adj_matrix, features)     # [B,N,C]
        elif self.aggregator_type == 'pooling':
            neigh = self.pooling_aggregator(adj_matrix, features)  # [B,N,C]
        else:
            raise ValueError("Aggregator type not supported! Use 'mean' or 'pooling'.")

        combined = torch.cat([features, neigh], dim=-1)            # [B,N,2*C]
        two_c = combined.size(-1)

        if self.linear is None or self.linear.in_features != two_c:
            self._build_linear(two_c, device=combined.device, dtype=combined.dtype)

        out = self.linear(combined)                                 # [B,N,out_features]
        return out

class GraphVAEWithGraphSAGE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes, hidden_dim=128, aggregator_type='mean'):
        super(GraphVAEWithGraphSAGE, self).__init__()
        
        self.gcn = GraphSAGELayer(hidden_dim, aggregator_type=aggregator_type)
        
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2_mu = nn.Linear(128, latent_dim)  # 均值
        self.fc2_logvar = nn.Linear(128, latent_dim)  # 对数方差

        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)
        
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc2_mu(h1)
        logvar = self.fc2_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  
        eps = torch.randn_like(std)  
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)  
    def forward(self, adj_matrix, features, labels=None):
        x = self.gcn(adj_matrix, features)  # [b, num_nodes, hidden_dim]
        # print('self.gcn(adj_matrix, features)', x.shape)
        
        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        recon_x = self.decode(z)

        logits = self.classifier(x)  # [b, num_nodes, num_classes]

        return recon_x, x, mu, logvar, logits

    def compute_uncertainty(
        self,
        recon_features: torch.Tensor,   # [B, N, C]
        features: torch.Tensor,         # [B, N, C] (重构目标，建议已做L2或z-score)
        logits: torch.Tensor,           # [B, N, K]
        labels: torch.Tensor = None,    # [B, N] (仅用于返回valid掩码；不参与熵计算)
        mu: torch.Tensor = None,        # [B, N, Z] (可选：用于KL不确定度)
        logvar: torch.Tensor = None,    # [B, N, Z]
        w_rec: float = 1.0,             # 三种不确定度的融合权重
        w_ent: float = 1.0,
        w_kl: float = 0.0,              # 不用KL就设0
        tau: float = 2.0,               # 温度，越大越惩罚高不确定
        w_min: float = 0.1,             # 最小权重，避免完全无梯度
        ignore_index: int = 255,
        eps: float = 1e-8,
    ):
        """
        返回:
          U:        [B,N] 综合不确定度 (detach)
          w:        [B,N] 样本权重 (detach)，可用于加权CE: (w*ce).sum()/w.sum()
          parts:    dict，含各分量与valid掩码，均detach
        """
        B, N, _ = features.shape
        device = features.device

        rec_err = F.mse_loss(recon_features, features, reduction='none').mean(dim=-1)  # [B,N]

        with torch.no_grad():
            p = torch.softmax(logits, dim=-1)                         # [B,N,K]
            entropy = -(p * (p.clamp_min(eps).log())).sum(dim=-1)     # [B,N]

        if (mu is not None) and (logvar is not None):
            kl_node = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B,N,Z]
            kl_node = kl_node.mean(dim=-1)                            # [B,N] 也可用 .sum(-1)
        else:
            kl_node = torch.zeros_like(rec_err)

        def _minmax_norm(t):
            t_min = t.amin(dim=(0,1), keepdim=True)
            t_max = t.amax(dim=(0,1), keepdim=True)
            return (t - t_min) / (t_max - t_min + eps)

        rec_norm = _minmax_norm(rec_err)
        ent_norm = _minmax_norm(entropy)
        kl_norm  = _minmax_norm(kl_node) if w_kl > 0 else kl_node

        U = (w_rec * rec_norm + w_ent * ent_norm + w_kl * kl_norm).detach()  # [B,N]

        w = torch.exp(-tau * U).detach()                                     # [B,N]
        if w_min is not None and w_min > 0:
            w = torch.clamp(w, min=w_min)

        if labels is not None:
            valid = (labels != ignore_index)
            w_eff = (w * valid.float()).detach()
        else:
            valid = torch.ones((B, N), dtype=torch.bool, device=device)
            w_eff = w

        parts = {
            'rec_err': rec_err.detach(),
            'entropy': entropy.detach(),
            'kl_node': kl_node.detach(),
            'rec_norm': rec_norm.detach(),
            'ent_norm': ent_norm.detach(),
            'kl_norm': kl_norm.detach(),
            'valid': valid.detach(),
        }
        return U, w_eff, parts
    
     
    def combined_loss(self, recon_x, x, mu, logvar, logits, labels,
                      step:int, total_steps:int,
                      lambda_rec:float=1.0, lambda_ce:float=1.0,
                      beta_start:float=1e-4, beta_end:float=1e-2,
                      ignore_index:int=255,
                      w_rec:float=1.0, w_ent:float=1.0,     # U 的两个分量系数 a,b
                      tau:float=2.0,                        # 温度: 越大越“惩罚”高不确定
                      w_min:float=0.1,                      # 最小权重下限（防止0梯度）
                      apply_weight_on_rec:bool=False,       # 是否也给重构项加权
                      eps:float=1e-8):
        """
        recon_x, x: [B, N, C]
        mu, logvar: [B, N, Z]
        logits: [B, N, K]
        labels: [B, N]
        return: total_loss(scale), logs(dict)
        """
    
        def _beta_warmup(step, total_steps, beta_start=1e-4, beta_end=1e-2):
            t = min(max(step / max(total_steps, 1), 0.0), 1.0)
            return beta_start + (beta_end - beta_start) * t  # 线性; 可换cos
    
        B, N, K = logits.shape
    
        
        rec_err_node = F.mse_loss(recon_x, x, reduction='none').mean(dim=-1)  # [B,N]
        reconstruction_loss = rec_err_node.mean()
    
        
        kl_node = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())              # [B,N,Z]
        kl_loss = kl_node.mean()
    
        
        ce_node = F.cross_entropy(
            logits.reshape(-1, K),
            labels.reshape(-1),
            ignore_index=ignore_index,
            reduction='none'
        ).view(B, N)  # [B,N]

       
        U, w_ce, parts = self.compute_uncertainty(
            recon_x, x, logits, labels,
            mu=mu, logvar=logvar,
            w_rec=1.0, w_ent=1.0, w_kl=1.0, tau=2.0, w_min=0.1, ignore_index=255
        )
        # print('w_ce', w_ce.shape)
        # print('ce_node', ce_node.shape)
        ce_weighted = ((w_ce * ce_node).sum() / (w_ce.sum() + 1e-8) ).mean()

    
       
        beta_kl = _beta_warmup(step, total_steps, beta_start, beta_end) * kl_loss
        total = lambda_rec * reconstruction_loss + beta_kl + lambda_ce * ce_weighted
    
        logs = {
            'rec': reconstruction_loss.detach(),
            'ce': ce_weighted.detach(),
            'beta_kl': torch.tensor(beta_kl * kl_loss, device=logits.device)
        }
        return total, logs






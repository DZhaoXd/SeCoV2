import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphBuilder(nn.Module):
    def __init__(
        self,
        alpha: float = 0.95,            # Weight of semantic similarity, position similarity weight = 1 - alpha
        num_pos_freqs: int = 4,         # Number of positional encoding frequencies (multi-frequency sin/cos)
        cross_weight: float = 0.5,      # Edge strength between current batch ↔ historical nodes
        topk: int = 16,                 # Maximum number of neighbors to keep for each node (including history); None means no filtering
        add_self_loops: bool = False,   # Whether to add self-loops
        normalize: str = "sym",         # 'sym' -> D^-1/2 A D^-1/2, 'row' -> D^-1 A, others -> no normalization
        clamp_neg: bool = False,        # Whether to clamp negative similarities to 0
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.num_pos_freqs = int(num_pos_freqs)
        self.cross_weight = float(cross_weight)
        self.topk = None if topk is None or topk <= 0 else int(topk)
        self.add_self_loops = bool(add_self_loops)
        self.normalize = normalize
        self.clamp_neg = bool(clamp_neg)

    @staticmethod
    def _cosine_sim_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Row-wise cosine similarity matrix: x:[N,C], y:[M,C] -> [N,M]
        """
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        return x @ y.t()

    def _pos_enc(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Multi-frequency sine/cosine positional encoding
        coords: [N, 4]  (x1, y1, x2, y2), recommended to be normalized to [0,1]
        return: [N, 4 * 2 * num_pos_freqs]
        """
        # Ensure dtype/device safety
        coords = coords.float()
        N, D = coords.shape
        assert D == 4, f"coordinates must be [N,4], got {coords.shape}"

        # Frequencies: 1,2,4,8,... (times 2π)
        freqs = (2.0 ** torch.arange(self.num_pos_freqs, device=coords.device)).view(1, -1)  # [1,F]
        angles = coords.unsqueeze(-1) * (2.0 * math.pi) * freqs  # [N,4,F]

        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [N,4,2F]
        return pe.view(N, -1)  # [N, 4*2F]

    def _sparsify_topk(self, A: torch.Tensor, k: int) -> torch.Tensor:
        """
        Row-wise top-k (keep self-loop), zero out the rest
        A: [N,N]
        """
        N = A.size(0)
        # Keep self-loop: set diagonal to max value so it’s always in top-k
        diag = torch.arange(N, device=A.device)
        A_with_self = A.clone()
        A_with_self[diag, diag] = torch.finfo(A.dtype).max
        topk_vals, topk_idx = torch.topk(A_with_self, k=k, dim=-1)
        mask = torch.zeros_like(A, dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk_idx, value=True)
        A = A * mask.float()
        # Restore self-loop weights (not forced to 1, will be handled later)
        A[diag, diag] = torch.diag(A)
        return A

    def _normalize(self, A: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "sym":
            deg = A.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            D_inv_sqrt = deg.pow(-0.5)
            return D_inv_sqrt * A * D_inv_sqrt.transpose(0, 1)
        elif mode == "row":
            deg = A.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            return A / deg
        else:
            return A

    @torch.no_grad()
    def build_graph(
        self,
        features: torch.Tensor,            # [B, C] Node features of current batch (one node per sample)
        coordinates: torch.Tensor,         # [B, 4] Position boxes (x1,y1,x2,y2) of current batch samples
        history_features: torch.Tensor = None,  # [n, K, C] Historical memory (K per class, n classes total)
        history_coords: torch.Tensor = None,    # [n, K, 4]
    ):
        """
        Outputs:
          adj: [1, N, N] combined adjacency matrix (current B nodes + flattened H history nodes)
          node_feats: [1, N, C] corresponding node features
          meta: dict containing (num_current, num_history)
        """
        assert features.dim() == 2, f"features must be [B,C], got {features.shape}"
        assert coordinates.dim() == 2 and coordinates.size(1) == 4, \
            f"coordinates must be [B,4], got {coordinates.shape}"

        B, C = features.shape
        dev, dtype = features.device, features.dtype

        # Current batch nodes
        cur_feats = features.to(dev).to(dtype)                    # [B,C]
        cur_pos = self._pos_enc(coordinates.to(dev))              # [B,P]

        # Intra-batch similarity
        S_ff = self._cosine_sim_matrix(cur_feats, cur_feats)      # [B,B]
        S_pp = self._cosine_sim_matrix(cur_pos, cur_pos)          # [B,B]
        A_cur = self.alpha * S_ff + (1.0 - self.alpha) * S_pp     # [B,B]

        # Process historical memory
        has_hist = (history_features is not None) and (history_coords is not None)
        if has_hist:
            assert history_features.dim() == 3 and history_coords.dim() == 3, \
                f"history_features [n,K,C], history_coords [n,K,4]; got {history_features.shape}, {history_coords.shape}"
            n, K, C_h = history_features.shape
            assert C_h == C, f"history feature dim {C_h} must match current {C}"
            H = n * K
            hist_feats = history_features.reshape(H, C).to(dev).to(dtype)  # [H,C]
            hist_pos = self._pos_enc(history_coords.reshape(H, 4).to(dev)) # [H,P]

            # History-to-history similarity
            H_ff = self._cosine_sim_matrix(hist_feats, hist_feats)     # [H,H]
            H_pp = self._cosine_sim_matrix(hist_pos, hist_pos)         # [H,H]
            A_hist = self.alpha * H_ff + (1.0 - self.alpha) * H_pp     # [H,H]

            # Current-to-history cross similarity
            X_ff = self._cosine_sim_matrix(cur_feats, hist_feats)      # [B,H]
            X_pp = self._cosine_sim_matrix(cur_pos, hist_pos)          # [B,H]
            A_cross = self.alpha * X_ff + (1.0 - self.alpha) * X_pp    # [B,H]
            A_cross = self.cross_weight * A_cross

            # Combine into block adjacency
            A_top = torch.cat([A_cur,         A_cross], dim=1)         # [B, B+H]
            A_bot = torch.cat([A_cross.t(),   A_hist],  dim=1)         # [H, B+H]
            A = torch.cat([A_top, A_bot], dim=0)                        # [N,N], N=B+H

            node_feats = torch.cat([cur_feats, hist_feats], dim=0)      # [N,C]
        else:
            A = A_cur
            node_feats = cur_feats

        N = A.size(0)

        # Clamp negative values (optional)
        if self.clamp_neg:
            A = A.clamp_min(0)

        # Sparsify (optional)
        if self.topk is not None and self.topk < N:
            A = self._sparsify_topk(A, k=self.topk)

        # Self-loops (optional)
        if self.add_self_loops:
            A = A.clone()
            idx = torch.arange(N, device=dev)
            A[idx, idx] = torch.maximum(A[idx, idx], torch.ones(N, device=dev, dtype=dtype))

        # Normalize
        A = self._normalize(A, mode=self.normalize)

        # Expand to batch=1 3D tensor
        adj = A.unsqueeze(0)                 # [1,N,N]
        node_feats = node_feats.unsqueeze(0) # [1,N,C]

        meta = {
            "num_current": B,
            "num_history": (N - B),
        }
        return adj, node_feats, meta

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from core.models.modules.decoders.segformer_head import SegFormerHead
from core.models.utils.modules import init_weight

class ASPP_Classifier_V2(nn.Module):
    def __init__(self, in_channels, dilation_series, padding_series, num_classes):
        super(ASPP_Classifier_V2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(
                    in_channels,
                    num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                )
            )

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x, size=None):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        if size is not None:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out

class SegFormerHead_Classifier(nn.Module):
    def __init__(self,align_corners,channels,num_classes,in_channels):
        super(SegFormerHead_Classifier, self).__init__()

        self.align_corners = align_corners
        # BN_op = getattr(nn, decoder.settings.norm_layer)
        channels = channels
        self.decoder = SegFormerHead(in_channels=in_channels)
        self.classifier = nn.Conv2d(channels, num_classes, 1, 1)
        init_weight(self.classifier)

    def forward(self, x, size=None):
        #size = (x.shape[2], x.shape[3])  x为最初始的图片形状
        output = self.decoder(x)
        out = {}
        out['embeddings'] = output
        output = self.classifier(output)
        out['pre_logits'] = output
        if size is not None:
            out['logits'] = F.interpolate(output, size=size, mode='bilinear', align_corners=self.align_corners)
            return out['logits']
        else:
            return output 


class SAMClassifier(nn.Module):
    """
    SegFormerHead 聚合多尺度 -> 掩码引导的通道门控 -> (前景/背景)掩码池化 -> 线性分类
    """
    def __init__(
        self,
        align_corners: bool,
        channels: int,
        num_classes: int,
        in_channels,                    # 传给 SegFormerHead 的配置，保持与你原来一致
        use_bg_branch: bool = True,     # 是否同时做背景池化并与前景特征拼接
        leak: float = 0.2,              # 门控泄露系数，0~1；越大越保留背景
        dilate: int = 3,                # 掩码膨胀核大小(奇数)，<=1则不膨胀
        blur_ks: int = 3                # 掩码平滑核大小(奇数)，<=1则不平滑
    ):
        super(SAMClassifier, self).__init__()
        self.align_corners = align_corners
        self.channels = channels
        self.num_classes = num_classes
        self.use_bg_branch = use_bg_branch
        self.leak = leak
        self.dilate = max(int(dilate), 1)
        self.blur_ks = max(int(blur_ks), 1)

        # 多尺度特征聚合
        self.decoder = SegFormerHead(in_channels=in_channels)

        # 掩码 -> 通道级门控
        self.mask_gate = nn.Conv2d(1, channels, kernel_size=1, bias=True)

        # 分类头（前景分支 或 前景+背景拼接）
        in_feats = channels * (2 if use_bg_branch else 1)
        self.sam_classifier = nn.Linear(in_feats, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.mask_gate.weight)
        if self.mask_gate.bias is not None:
            nn.init.zeros_(self.mask_gate.bias)
        nn.init.xavier_uniform_(self.sam_classifier.weight)
        nn.init.zeros_(self.sam_classifier.bias)

    @torch.no_grad()
    def _prep_mask(self, ins_mask: torch.Tensor, target_size):
        """
        预处理掩码：resize -> 可选膨胀 -> 可选平滑 -> clamp 到 [0,1]
        ins_mask: [B, H, W] 或 [B,1,H,W]
        return:  [B,1,h,w] float
        """
        if ins_mask.dim() == 3:
            m = ins_mask[:, None]
        else:
            m = ins_mask
        m = m.float()
        h, w = target_size
        m = F.interpolate(m, size=(h, w), mode='nearest')

        # 膨胀（保留边界上下文）
        if self.dilate > 1:
            pad = self.dilate // 2
            m = F.max_pool2d(m, kernel_size=self.dilate, stride=1, padding=pad)

        # 平滑（变成软权重，缓解硬边）
        if self.blur_ks > 1:
            pad = self.blur_ks // 2
            m = F.avg_pool2d(m, kernel_size=self.blur_ks, stride=1, padding=pad)

        return m.clamp_(0.0, 1.0)

    def forward(self, x_multi, ins_mask: torch.Tensor, size=None):
        """
        x_multi: SegFormerHead 的输入（多尺度特征列表/字典，保持你原来的用法）
        ins_mask: [B,H,W] 或 [B,1,H,W] 的前景掩码（原图尺度）
        """
        # 1) 多尺度聚合
        feat = self.decoder(x_multi)             # [B, C, h, w]
        B, C, h, w = feat.shape
        # print('B, C, h, w: ',  (B, C, h, w))
        # 2) 掩码预处理（到 decoder 输出分辨率）
        m = self._prep_mask(ins_mask, (h, w))    # [B,1,h,w]

        # 3) 掩码门控（通道级、可学习，带泄露系数避免背景全灭）
        m = m.to(dtype=feat.dtype, device=feat.device)   # 关键一步
        g = torch.sigmoid(self.mask_gate(m))
        feat = feat * (self.leak + (1.0 - self.leak) * g)

        # 4) 掩码池化
        # 前景池化（masked GAP）
        m_sum = m.sum(dim=(2, 3), keepdim=False) + 1e-6          # [B,1]
        fg_feat = (feat * m).sum(dim=(2, 3)) / m_sum              # [B,C]

        if self.use_bg_branch:
            mb = 1.0 - m
            mb_sum = mb.sum(dim=(2, 3), keepdim=False) + 1e-6     # [B,1]
            bg_feat = (feat * mb).sum(dim=(2, 3)) / mb_sum        # [B,C]
            pooled = torch.cat([fg_feat, bg_feat], dim=1)         # [B,2C]
        else:
            pooled = fg_feat                                      # [B,C]

        # 5) 线性分类
        out = self.sam_classifier(pooled)                          # [B,num_classes]
        return out, fg_feat





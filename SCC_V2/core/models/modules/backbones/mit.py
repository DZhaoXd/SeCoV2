import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_)
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=2, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # ---- Mask-aware gating (keeps 3-channel pretrained weights intact) ----
        self.use_stage_gate = True
        self.gate_leak = 0.2           # keep some background
        self.mask_dilate = 3           # odd int; <=1 disables
        self.mask_blur_ks = 3          # odd int; <=1 disables
        self.mask_gate_s1 = nn.Conv2d(1, embed_dims[0], kernel_size=1, bias=True)
        self.mask_gate_s2 = nn.Conv2d(1, embed_dims[1], kernel_size=1, bias=True)
        self.mask_gate_s3 = nn.Conv2d(1, embed_dims[2], kernel_size=1, bias=True)
        self.mask_gate_s4 = nn.Conv2d(1, embed_dims[3], kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.mask_gate_s1.weight); nn.init.zeros_(self.mask_gate_s1.bias)
        nn.init.xavier_uniform_(self.mask_gate_s2.weight); nn.init.zeros_(self.mask_gate_s2.bias)
        nn.init.xavier_uniform_(self.mask_gate_s3.weight); nn.init.zeros_(self.mask_gate_s3.bias)
        nn.init.xavier_uniform_(self.mask_gate_s4.weight); nn.init.zeros_(self.mask_gate_s4.bias)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.no_grad()
    def _prep_mask(self, ins_mask, target_hw):
        """Resize -> optional dilation -> optional average blur; return [B,1,h,w] in [0,1]."""
        if ins_mask is None:
            return None
        if ins_mask.dim() == 3:
            m = ins_mask[:, None]
        else:
            m = ins_mask
        m = m.float().clamp(0, 1)
        h, w = target_hw
        m = F.interpolate(m, size=(h, w), mode='nearest')
        if self.mask_dilate and self.mask_dilate > 1:
            pad = self.mask_dilate // 2
            m = F.max_pool2d(m, kernel_size=self.mask_dilate, stride=1, padding=pad)
        if self.mask_blur_ks and self.mask_blur_ks > 1:
            pad = self.mask_blur_ks // 2
            m = F.avg_pool2d(m, kernel_size=self.mask_blur_ks, stride=1, padding=pad)
        return m.clamp_(0.0, 1.0)

    def _apply_gate(self, feat, gate_conv, ins_mask):
        if ins_mask is None or not self.use_stage_gate:
            return feat
        B, C, h, w = feat.shape
        m = self._prep_mask(ins_mask, (h, w))
        # 关键：与特征/卷积保持相同 dtype & device（AMP 下会是 fp16）
        m = m.to(dtype=feat.dtype, device=feat.device)
        g = torch.sigmoid(gate_conv(m))
        return feat * (self.gate_leak + (1.0 - self.gate_leak) * g)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            #logger = get_root_logger()
            #load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            state_dict = torch.load(pretrained, map_location='cpu')
            # Remove last FC layer
            #del state_dict['fc.weight'], state_dict['fc.bias']
            self.load_state_dict(state_dict)
            del state_dict

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]
        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]
        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]
        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x, ins_mask):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self._apply_gate(x, self.mask_gate_s1, ins_mask)
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self._apply_gate(x, self.mask_gate_s2, ins_mask)
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self._apply_gate(x, self.mask_gate_s3, ins_mask)
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self._apply_gate(x, self.mask_gate_s4, ins_mask)
        outs.append(x)

        return outs

    def forward(self, x, ins_mask=None):
        return self.forward_features(x, ins_mask)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                                bias=True, groups=dim)

    def forward(self, x, H: int = None, W: int = None):
        """
        兼容两种输入：
        - x.shape == [B, C, H, W]：直接做 depthwise conv，返回 [B, C, H, W]
        - x.shape == [B, N, C]：需要提供 H, W（或 N=H*W 且后续会用到），
          会先还原为 [B, C, H, W] 做 conv，再展平回 [B, N, C]
        """
        if x.dim() == 4:
            # [B, C, H, W] -> [B, C, H, W]
            return self.dwconv(x)

        elif x.dim() == 3:
            # [B, N, C] -> [B, C, H, W] -> depthwise conv -> [B, N, C]
            B, N, C = x.shape
            assert H is not None and W is not None and H * W == N, \
                f"DWConv expects H*W==N, got H={H}, W={W}, N={N}"
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.dwconv(x)
            x = x.flatten(2).transpose(1, 2)
            return x

        else:
            raise ValueError(f"DWConv expects 3D or 4D tensor, got {x.dim()}D")



class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8],
                                     mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                                     norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
                                     sr_ratios=[8, 4, 2, 1])

class mit_b1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                                     mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                                     norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
                                     sr_ratios=[8, 4, 2, 1])

class mit_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                                     mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                                     norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
                                     sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)

class mit_b3(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                                     mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                                     norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3],
                                     sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)

class mit_b4(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                                     mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                                     norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3],
                                     sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)

class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


def _clean_ckpt_keys(state_dict, model_state):
    """
    清理并重映射 checkpoint 的键：
    - 若包含 'state_dict' 则取其子dict
    - 去掉常见前缀：'module.', 'backbone.', 'encoder.'
    - 丢弃模型中不存在的 head、分类层等键
    """
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # 去前缀
    def strip_prefix(k):
        for p in ('module.', 'backbone.', 'encoder.'):
            if k.startswith(p):
                return k[len(p):]
        return k

    new_sd = {}
    for k, v in state_dict.items():
        nk = strip_prefix(k)
        # 丢弃明显不需要的分类头；也避免覆盖我们新加的门控
        if nk.startswith('head.'):
            continue
        # 仅保留模型里存在的键的形状兼容项
        if nk in model_state and model_state[nk].shape == v.shape:
            new_sd[nk] = v
    return new_sd


def get_mit(
    pretrain: str,
    variety: str,
) -> MixVisionTransformer:
    series = variety.lower()
    mit_type = f"mit_{series}()"
    mitnet = eval(mit_type)

    if pretrain:
        print('get_mit load_checkpoint', pretrain)
        ckpt = torch.load(pretrain, map_location='cpu')
        model_state = mitnet.state_dict()
        cleaned = _clean_ckpt_keys(ckpt, model_state)

        # 使用 strict=False 忽略缺失（例如 mask_gate_s* 和 head）
        missing, unexpected = mitnet.load_state_dict(cleaned, strict=False)

        # 可打印一下信息，便于核查（训练时可关掉）
        if missing:
            print('[mit] missing keys (expected, will be randomly init):', missing)
        if unexpected:
            print('[mit] unexpected keys (ignored):', unexpected)

        del ckpt, cleaned

    return mitnet


import logging

logger = logging.getLogger(__name__)
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import RandomRotation
import numpy as np
import random


"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class CrossAttention(nn.Module):
    def __init__(
        self,
        kv_in_dim,
        q_in_dim,
        atten_dim,
        value_dim,
        out_dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = atten_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.atten_dim = atten_dim
        self.value_dim = value_dim

        self.q = nn.Linear(q_in_dim, atten_dim, bias=qkv_bias)
        self.k = nn.Linear(kv_in_dim, atten_dim, bias=qkv_bias)
        self.v = nn.Linear(kv_in_dim, value_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(value_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond):
        Bx, Nx, Cx = x.shape
        Bc, Nc, Cc = cond.shape

        q = (
            self.q(x)
            .reshape(Bx, Nx, self.num_heads, self.atten_dim // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(cond)
            .reshape(Bc, Nc, self.num_heads, self.atten_dim // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(cond)
            .reshape(Bc, Nc, self.num_heads, self.value_dim // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(Bx, Nx, Cx)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CABlock(nn.Module):
    def __init__(
        self,
        bloc_dim,
        cond_dim,
        atten_dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(bloc_dim)
        self.attn = CrossAttention(
            kv_in_dim=cond_dim,
            q_in_dim=bloc_dim,
            atten_dim=atten_dim,
            value_dim=bloc_dim,
            out_dim=bloc_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(bloc_dim)
        mlp_hidden_dim = int(bloc_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=bloc_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, cond, return_attention=False):
        y, attn = self.attn(self.norm1(x), cond)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConditionalPrototypes(nn.Module):
    def __init__(
        self,
        bloc_dim,
        cond_dim,
        atten_dim,
        num_heads,
        depth,
        in_dim=3,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        cond_bottleneck=None,
    ):
        super().__init__()

        self.proj = nn.Linear(in_dim, bloc_dim)
        self.blocks = nn.ModuleList(
            [
                CABlock(
                    bloc_dim,
                    cond_dim,
                    atten_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop,
                    attn_drop,
                    drop_path,
                    act_layer,
                    norm_layer,
                )
            ]
            * depth
        )
        self.cond = cond_bottleneck is not None
        if self.cond:
            self.cond_proj = (
                Mlp(cond_dim, cond_bottleneck) if cond_bottleneck > 0 else nn.Identity()
            )

    def forward(self, x, cond):
        x = self.proj(x)
        cond = self.cond_proj(cond) if self.cond else torch.zeros_like(cond)
        for block in self.blocks:
            x = block(x, cond)
        return x


from omegaconf import DictConfig
from od3d.models.backbones.dino import DINOv2


class DINOv2_Sph(DINOv2):
    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__(config=config)

        self.sphere_mapper = nn.Sequential(
            nn.Linear(self.extractor.embed_dim, self.extractor.embed_dim // 2),
            nn.GELU(),
            Block(
                dim=self.extractor.embed_dim // 2,
                num_heads=self.extractor.embed_dim // 2 // 64,
            ),
            nn.Linear(self.extractor.embed_dim // 2, 3),
        )

        if config.get("sph_weights", None) is not None:
            ckpt = torch.load(config.get("sph_weights"))
            self.load_state_dict(ckpt, strict=False)

        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False

        self.out_dims[-1] = self.out_dims[-1] + 3

    def forward(self, x):
        x_out = super().forward(x)

        x_out_feat = x_out.featmaps[-1]
        B, C, H, W = x_out_feat.size()
        x_out_feat = x_out_feat.permute(0, 2, 3, 1)
        x_out_feat = x_out_feat.reshape(B, H * W, C)

        sph = self.sphere_mapper(x_out_feat)

        sph = nn.functional.normalize(sph, dim=-1)  # * gt_mask
        x_out_feat = nn.functional.normalize(x_out_feat, dim=-1)

        sph = sph.reshape(B, H, W, 3).permute(0, 3, 1, 2)
        x_out_feat = x_out_feat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        sph = sph * 0.2
        x_out_feat = x_out_feat * 0.8

        x_out_feat = torch.cat([x_out_feat, sph], dim=1)

        x_out.featmaps[-1] = x_out_feat
        return x_out


#
# class DINOv2_MAPPER(nn.Module):
#     def __init__(self, backbone='dinov2_vitb14', n_cats=1):
#         super().__init__()
#         assert 'dino' in backbone, f"""{backbone} is not a DINO model """
#         if 'dinov2' in backbone:
#             repo = 'facebookresearch/dinov2'
#         else:
#             repo = 'facebookresearch/dino'
#         from od3d.models.model import OD3D_Model
#         from od3d.cv.transforms.sequential import SequentialTransform
#         self.model = OD3D_Model.create_by_name(backbone)
#         self.patch_size = self.model.backbone.extractor.patch_size
#         self.embed_dim = self.model.backbone.extractor.embed_dim
#         # self.dino = torch.hub.load(repo, backbone)
#         # self.patch_size = self.dino.patch_size
#         # self.embed_dim = self.dino.embed_dim
#         self.sphere_mapper = nn.Sequential(
#             nn.Linear(self.embed_dim, self.embed_dim//2),
#             nn.GELU(),
#             Block(dim=self.embed_dim//2, num_heads=self.embed_dim//2//64),
#             nn.Linear(self.embed_dim//2, 3),
#             )
#         ##################################################################
#         ## The next four lines are obsolete and serves no other purpose ##
#         ## than manipulating the RNG state in order to reproduce the    ##
#         ## numbers in the paper.                                        ##
#         ## They can safely be ignored for any other purposes.           ##
#         ##################################################################
#         sph_size = 2**10
#         self.sphere = nn.Embedding(sph_size, 3)
#         with torch.no_grad():
#             self.sphere.weight = nn.Parameter(nn.functional.normalize(torch.randn(sph_size, 3)), requires_grad=False)
#         head_dim = 64
#         self.n_cats = n_cats
#         self.prototypes = ConditionalPrototypes(bloc_dim=self.embed_dim,
#                                                 cond_dim=self.n_cats,
#                                                 atten_dim=self.embed_dim,
#                                                 num_heads=self.embed_dim//head_dim,
#                                                 depth=2,
#                                                 cond_bottleneck=0)
#     def forward(self, im, cats, gt_mask, max_layer=None):
#         # im = self.transform(im)
#         # dino = self.model.backbone.extractor
#         # # dino = self.dino
#         # max_layer = max_layer or len(dino.blocks)
#         # x = im
#         # b_size, _, h, w = x.shape
#         # fm_shape = (h//self.patch_size, w//self.patch_size)
#         # with torch.no_grad():
#         #     dino_out = dino.forward_features(x)
#         #     x = dino_out["x_prenorm"][:,1:]
#         #     cond = dino_out["x_prenorm"][:,0]
#         # x = x.contiguous()
#
#         #for param in self.model.parameters():
#         #    param.requires_grad = False
#
#         x = self.model(im)
#         B,C,H,W = x.size()
#         x = x.permute(0, 2, 3, 1)
#         x = x.reshape(B, H * W, C)
#         sph = self.sphere_mapper(x)
#         sph = nn.functional.normalize(sph, dim=-1) * gt_mask
#         return x, sph, None

import logging

logger = logging.getLogger(__name__)
import torch.nn as nn

# from od3d.models.heads.head import OD3D_Head
from typing import List
from omegaconf import DictConfig
import torch
from od3d.models.heads.mlp import MLP
from od3d.data.batch_datatypes import OD3D_ModelData


class HarmonicEmbedding(nn.Module):
    def __init__(self, n_harmonic_functions=10, scalar=1, dim=-1):
        """
        Positional Embedding implementation (adapted from Pytorch3D).
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**self.n_harmonic_functions * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**self.n_harmonic_functions * x[..., i])
            ]
        Note that `x` is also premultiplied by `scalar` before
        evaluting the harmonic functions.
        """
        super().__init__()
        self.frequencies = scalar * (2.0 ** torch.arange(n_harmonic_functions))
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        if self.dim != -1:
            x = x.transpose(self.dim, -1).contiguous()

        embed = (x[..., None] * self.frequencies.to(x.device)).view(*x.shape[:-1], -1)

        if self.dim != -1:
            embed = embed.transpose(self.dim, -1).contiguous()
            x = x.transpose(self.dim, -1).contiguous()

        return torch.cat((embed.sin(), embed.cos()), dim=self.dim)


class CoordMLP(MLP):
    def __init__(
        self,
        in_dims: List,
        in_upsample_scales: List,
        config: DictConfig,
        query_dim=3,
        n_harmonic_functions=10,
        embedder_scalar=2.8274,  # 2 * np.pi / 2 * 0.9  # originally (-0.5*s, 0.5*s) rescale to (-pi, pi) * 0.9
        embed_concat_pts=True,
    ):
        self.encoder_featmap_vae = config.get("encoder_featmap_vae", False)

        if config.get("resnet", None) is not None:
            if in_dims is None:
                resnet_in_dims = [config.in_dim]
                resnet_in_upsample_scales = []
            else:
                resnet_in_dims = in_dims
                resnet_in_upsample_scales = in_upsample_scales
            from od3d.models.heads.resnet import ResNet

            resnet = ResNet(
                in_dims=resnet_in_dims,
                in_upsample_scales=resnet_in_upsample_scales,
                config=config.resnet,
            )
            feat_dim = resnet.out_dim
            vit = None
        elif config.get("vit", None) is not None:
            if in_dims is None:
                vit_in_dims = [config.in_dim]
                vit_in_upsample_scales = []
            else:
                vit_in_dims = in_dims
                vit_in_upsample_scales = in_upsample_scales
            from od3d.models.heads.vit import ViT

            vit = ViT(
                in_dims=vit_in_dims,
                in_upsample_scales=vit_in_upsample_scales,
                config=config.vit,
            )
            feat_dim = vit.out_dim
            resnet = None
        else:
            resnet = None
            vit = None
            resnet_in_dims = None
            resnet_in_upsample_scales = None
            if in_dims is not None:
                feat_dim = in_dims[-1]
            else:
                feat_dim = config.in_dim

        if n_harmonic_functions > 0:
            self.embed_dim = query_dim * 2 * n_harmonic_functions
            self.embed_concat_pts = embed_concat_pts
            if embed_concat_pts:
                self.embed_dim += 3
        else:
            self.embed_dim = query_dim
        self.symmetrize = config.get("symmetrize", True)

        self.mlp_in_dim = feat_dim + self.embed_dim
        if in_dims is not None:
            in_dims = [self.mlp_in_dim]
        else:
            config.update({"in_dim": self.mlp_in_dim})
            # config.in_dim = self.mlp_in_dim

        if config.get("resnet", None) is not None:
            config.update({"resnet": None})
        if config.get("vit", None) is not None:
            config.update({"vit": None})

        super().__init__(
            in_dims=in_dims,
            in_upsample_scales=in_upsample_scales,
            config=config,
        )

        self.resnet = resnet
        self.vit = vit

        if n_harmonic_functions > 0:
            self.embedder = HarmonicEmbedding(n_harmonic_functions, embedder_scalar)

        else:
            self.embedder = None

        # if config.get('min_max', None) is not None:
        #    self.register_buffer('min_max', config.min_max)  # Cx2
        # else:
        #    self.min_max = None
        # self.bsdf = None

    def forward(self, x: OD3D_ModelData):
        pts3d = x.pts3d  # BxNx3
        B, N = pts3d.shape[0], pts3d.shape[1]
        if self.symmetrize:
            # pts3d[:, :, 0] = pts3d[:, :, 0].abs() # mirror -x to +x
            pts3d_x, pts3d_y, pts3d_z = pts3d.unbind(-1)
            pts3d = torch.stack(
                [pts3d_x.abs(), pts3d_y, pts3d_z], -1
            )  # mirror -x to +x

        if self.embedder is not None:
            pts3d_embed = self.embedder(pts3d)
            if self.embed_concat_pts:
                pts3d_embed = torch.cat([pts3d, pts3d_embed], -1)
        else:
            pts3d_embed = pts3d

        if x.latent is not None:
            x_latent = x.latent
            x_latent_mu = x.latent_mu
            x_latent_logvar = x.latent_logvar
        else:
            if self.resnet is not None:
                x_resnet = self.resnet(x)
                x_latent = x_resnet.feat
                x_latent_mu = x_resnet.feat_mu
                x_latent_logvar = x_resnet.feat_logvar
            elif self.vit is not None:
                x_vit = self.vit(x)
                x_latent = x_vit.featmap
                x_latent_mu = None
                x_latent_logvar = None
            else:
                x_latent = x.feat  # [-1].flatten(1)  # BxF
                x_latent_mu = x.feat_mu
                x_latent_logvar = x.feat_logvar

        if x_latent is not None:
            feats = torch.cat(
                [
                    pts3d_embed,
                    x_latent[:, None].expand(
                        *pts3d_embed.shape[:2], x_latent.shape[-1]
                    ),
                ],
                dim=-1,
            )  # BxNxE+F
        else:
            feats = pts3d_embed  # BxNxE+F

        x_in = OD3D_ModelData(feat=feats.reshape(-1, feats.shape[-1]))
        x_out = super().forward(x_in)
        x_out.feat = x_out.feat.reshape(B, N, -1)

        if x_latent is not None:
            x_out.latent = x_latent  # .reshape(B, -1)
            x_out.latent_mu = x_latent_mu  # .reshape(B, -1)
            x_out.latent_logvar = x_latent_logvar  # .reshape(B, -1)
        return x_out

        # pts3d_embed = self.in_layer(pts3d_embed)
        # x_in = torch.concat([pts3d_embed, x.feat], dim=-1)
        # out = self.mlp(self.relu(x_in))  # (B, ..., C)

        # if self.min_max is not None:
        #     out = out * (self.min_max[:, 1] - self.min_max[:, 0]) + self.min_max[:, 0]
        #
        # return out

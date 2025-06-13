import logging

logger = logging.getLogger(__name__)
from od3d.models.heads.head import OD3D_Head
from omegaconf import DictConfig
import torch
from torchvision.models.resnet import Bottleneck, BasicBlock
import torch.nn as nn
from typing import List
from od3d.data.ext_enum import ExtEnum
from od3d.data.batch_datatypes import OD3D_ModelData

class RESNET_CONV_BLOCK_TYPES(str, ExtEnum):
    BOTTLENECK = "bottleneck"
    BASIC = "basic"


def get_block(
    block_type: RESNET_CONV_BLOCK_TYPES,
    in_dim: int,
    out_dim: int,
    stride: int,
    out_conv1x1: bool = False,
    pre_upsampling: float = 1.0,
):
    moudule_list = nn.ModuleList()
    if pre_upsampling != 1.0:
        moudule_list.append(nn.Upsample(scale_factor=pre_upsampling))

    if block_type == RESNET_CONV_BLOCK_TYPES.BASIC:
        moudule_list.append(
            BasicBlock(
                inplanes=in_dim,
                planes=out_dim,
                stride=stride,
                downsample=nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_dim),
                ),
            ),
        )

    elif block_type == RESNET_CONV_BLOCK_TYPES.BOTTLENECK:
        moudule_list.append(
            Bottleneck(
                inplanes=in_dim,
                planes=out_dim // 4,
                stride=stride,
                downsample=nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_dim),
                ),
            ),
        )
    else:
        logger.error(f"Unknown block type {block_type}.")
        raise NotImplementedError

    if out_conv1x1:
        moudule_list.append(
            nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, bias=True),
        )

    return nn.Sequential(*moudule_list)


class ResNet(OD3D_Head):
    def __init__(
        self,
        in_dims: List,
        in_upsample_scales: List,
        config: DictConfig,
    ):
        super().__init__(
            in_dims=in_dims,
            in_upsample_scales=in_upsample_scales,
            config=config,
        )

        self.upsample_conv_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.block_type: RESNET_CONV_BLOCK_TYPES = config.block_type
        self.pad_zero = config.get("pad_zero", False)
        self.pad_width = 1
        # self.pca = config.pca.get("enabled", False)
        # self.pca_dim = config.pca.get("dim", 32)
        self.in_upsampled_dim = config.get("in_upsampled_dim", self.in_dims[-1])
        assert len(self.in_upsample_scales) == len(self.in_dims) - 1

        for i in range(len(self.in_dims) - 1):
            # downsample_channels = nn.Sequential(
            #    nn.Conv2d(self.in_dims[i] + self.in_dims[i + 1], self.in_dims[i + 1], kernel_size=1, stride=1, bias=False),
            #    nn.BatchNorm2d(self.in_dims[i + 1]))
            # self.upsample_conv_blocks.append(Bottleneck(inplanes=self.in_dims[i] + self.in_dims[i + 1], planes=self.in_dims[i + 1] // 4, downsample=downsample_channels))
            if self.in_upsampled_dim != self.in_dims[-1]:
                if i == 0:
                    self.upsample_conv_blocks.append(
                        get_block(
                            block_type=self.block_type,
                            in_dim=self.in_dims[i] + self.in_dims[i + 1],
                            out_dim=self.in_upsampled_dim,
                            stride=1,
                        ),
                    )
                else:
                    self.upsample_conv_blocks.append(
                        get_block(
                            block_type=self.block_type,
                            in_dim=self.in_upsampled_dim + self.in_dims[i + 1],
                            out_dim=self.in_upsampled_dim,
                            stride=1,
                        ),
                    )
            else:
                self.upsample_conv_blocks.append(
                    get_block(
                        block_type=self.block_type,
                        in_dim=self.in_dims[i] + self.in_dims[i + 1],
                        out_dim=self.in_dims[i + 1],
                        stride=1,
                    ),
                )

            self.upsample.append(nn.Upsample(scale_factor=self.in_upsample_scales[i]))

        if config.fully_connected.out_dim is not None:
            self.out_dim = config.fully_connected.out_dim
        elif len(config.conv_blocks.out_dims) > 0:
            self.out_dim = config.conv_blocks.out_dims[-1]
        else:
            self.out_dim = self.in_dims[-1]

        self.pred_feat_distr = config.get("pred_feat_distr", False)

        self.conv_blocks = nn.ModuleList()
        self.conv_blocks_out_dims = config.conv_blocks.out_dims
        self.conv_blocks_count = len(self.conv_blocks_out_dims)
        self.conv_blocks_strides = config.conv_blocks.strides
        self.conv_blocks_out_conv1x1 = config.conv_blocks.out_conv1x1

        if self.conv_blocks_count > 0:
            self.conv_blocks_in_dims = [self.in_upsampled_dim] + [
                config.conv_blocks.out_dims[i]
                for i in range(self.conv_blocks_count - 1)
            ]
        else:
            self.conv_blocks_in_dims = []
        self.conv_blocks_pre_upsampling = config.conv_blocks.pre_upsampling
        assert len(self.conv_blocks_in_dims) == len(self.conv_blocks_out_dims)
        assert len(self.conv_blocks_out_dims) == len(self.conv_blocks_strides)
        assert len(self.conv_blocks_out_dims) == len(self.conv_blocks_out_conv1x1)

        assert (
            len(self.conv_blocks_in_dims) == 0
            or self.conv_blocks_in_dims[0] == self.in_upsampled_dim
        )
        assert len(self.conv_blocks_out_dims) == len(self.conv_blocks_pre_upsampling)

        self.conv_block_scaling = [
            self.conv_blocks_strides[i] / self.conv_blocks_pre_upsampling[i]
            for i in range(len(self.conv_blocks_strides))
        ]
        # self.conv_blocks = nn.Sequential(*[Bottleneck(inplanes=self.conv_blocks_in_dims[i],
        #                                              planes=self.conv_blocks_out_dims[i] // 4,
        #                                              stride=self.conv_blocks_strides[i],
        #                                              downsample=nn.Sequential(
        #                                                    nn.Conv2d(self.conv_blocks_in_dims[i], self.conv_blocks_out_dims[i], kernel_size=1, stride=1, bias=False),
        #                                                    nn.BatchNorm2d(self.conv_blocks_out_dims[i])))
        #                                   for i in range(self.conv_blocks_count)])

        self.conv_blocks = nn.Sequential(
            *[
                get_block(
                    block_type=self.block_type,
                    in_dim=self.conv_blocks_in_dims[i],
                    out_dim=self.conv_blocks_out_dims[i],
                    stride=self.conv_blocks_strides[i],
                    pre_upsampling=self.conv_blocks_pre_upsampling[i],
                    out_conv1x1=self.conv_blocks_out_conv1x1[i]
                )
                for i in range(self.conv_blocks_count)
            ],
        )

        if config.fully_connected.out_dim is not None:
            self.fc_enabled = True
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.downsample_rate = 1
        else:
            from operator import mul
            from functools import reduce

            self.downsample_rate = reduce(mul, [1] + self.conv_block_scaling, 1)
            self.fc_enabled = False

        if len(self.conv_blocks_out_dims) > 0:
            self.fc_required = True
            if self.fc_enabled:
                self.linear_in_dim = self.conv_blocks_out_dims[-1]
            else:
                self.linear_in_dim = self.in_dims[-1]
            if not self.pred_feat_distr:
                self.fc = nn.Linear(self.linear_in_dim, self.out_dim)
            else:
                self.fc = nn.Linear(self.linear_in_dim, 2 * self.out_dim)
        else:
            self.fc_required = False

        if config.get("pca", None) is not None and config.pca.get("enable", False):
            self.pca_enabled = False
            pca_dim = config.pca.get("out_dim", 32)
            self.pca_layer = nn.Linear(self.out_dim, pca_dim, bias=False)
            self.mean_features = torch.zeros(1, self.out_dim)
            for param in self.pca_layer.parameters():
                param.requires_grad = False
            nn.init.eye_(self.pca_layer.weight)
            self.out_dim = pca_dim
        else:
            self.pca_enabled = False

    def get_feat_distr(self, feat):
        feat_dim = feat.shape[-1]
        feat_mu = feat[..., :feat_dim // 2]
        feat_logvar = feat[..., feat_dim // 2:]
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * feat_logvar)
        # Generate random noise using the same shape as std
        eps = torch.randn_like(std)
        feat = feat_mu + eps * std
        return feat, feat_mu, feat_logvar

    def forward(self, x: OD3D_ModelData):
        if len(self.in_dims) == 1:
            x_featmap = x.featmaps[0]
        else:
            x_featmap = None
            for i in range(len(self.in_dims) - 1):
                if i == 0:
                    x_low = x.featmaps[i]
                else:
                    x_low = x_featmap
                x_featmap = self.upsample_conv_blocks[i](
                    torch.cat([x.featmaps[i + 1], self.upsample[i](x_low)], dim=1),
                )

        if self.pad_zero:
            x_featmap = torch.nn.functional.pad(
                x_featmap,
                (self.pad_width, self.pad_width, self.pad_width, self.pad_width),
                mode="constant",
                value=0,
            )

        x_featmap = self.conv_blocks(x_featmap)

        if self.pad_zero:
            pad_width = int((1 / self.downsample_rate) * self.pad_width)
            x_featmap = x_featmap[:, :, pad_width:-pad_width, pad_width:-pad_width]

        if self.fc_enabled:
            x_feat = self.avgpool(x_featmap)
            x_feat = torch.flatten(x_feat, 1)
            x_feat = self.fc(x_feat)
        else:
            if self.fc_required:
                x_feat = self.fc(x.feat)
            else:
                x_feat = x.feat

        if self.pred_feat_distr:
            x_feat, x_feat_mu, x_feat_logvar = self.get_feat_distr(feat=x_feat)
        else:
            x_feat_mu = None
            x_feat_logvar = None

        if self.normalize:
            x_featmap = torch.nn.functional.normalize(x_featmap, p=2, dim=1)
            x_feat = torch.nn.functional.normalize(x_feat, p=2, dim=1)

        B, C, H, W = x_featmap.shape
        if self.pca_enabled:
            x_featmap = torch.flatten(x_featmap, 2)
            x_featmap = x_featmap.permute(0, 2, 1)
            x_featmap = x_featmap - self.mean_features
            x_featmap = self.pca_layer(x_featmap)
            x_featmap = x_featmap.permute(0, 2, 1)
            x_featmap = x_featmap.view(B, self.out_dim, H, W)

            if self.normalize:
                x_featmap = torch.nn.functional.normalize(x_featmap, p=2, dim=1)
                x_feat = torch.nn.functional.normalize(x_feat, p=2, dim=1)

        x_out = OD3D_ModelData(featmap=x_featmap, feat=x_feat, feat_mu=x_feat_mu, feat_logvar=x_feat_logvar)
        return x_out

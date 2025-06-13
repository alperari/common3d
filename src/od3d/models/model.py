import logging

import od3d.io

logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from omegaconf import DictConfig, open_dict
from od3d.models.backbones.backbone import OD3D_Backbone
from od3d.models.heads.head import OD3D_Head
from pathlib import Path

class OD3D_Model(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        if self.config.backbone is not None:
            self.backbone: OD3D_Backbone = OD3D_Backbone.subclasses[
                self.config.backbone.class_name
            ](config=self.config.backbone)
            self.transform = self.backbone.transform
        else:
            self.backbone = None
            self.transform = None
            
        if self.config.head is not None:
            if self.backbone is not None:
                with open_dict(self.config):
                    self.config.head.in_dims = self.backbone.out_dims
                    self.config.head.in_upsample_scales = self.backbone.out_downsample_scales

            self.head: OD3D_Head = OD3D_Head.subclasses[self.config.head.class_name](
                config=self.config.head,
                in_dims=self.config.head.get('in_dims', None),
                in_upsample_scales=self.config.head.get('in_upsample_scales', None),
            )
            
            self.out_dim = self.head.out_dim
            if self.backbone is not None:
                self.downsample_rate = (
                        self.backbone.downsample_rate * self.head.downsample_rate
                )
            else:
                self.downsample_rate = self.head.downsample_rate

        else:
            self.out_dim = self.backbone.out_dims[-1]
            self.downsample_rate = self.backbone.downsample_rate

        if config.get("nemo_checkpoint", None) is not None:
            self.load_nemo_checkpoint(config.nemo_checkpoint)
        elif config.get("nemo_checkpoint_old", None) is not None:
            self.load_nemo_checkpoint_old(config.nemo_checkpoint_old)

    def load_nemo_checkpoint_old(self, path_nemo_checkpoint):
        checkpoint = torch.load(path_nemo_checkpoint, map_location="cuda:0")
        self.backbone.net = torch.nn.DataParallel(self.backbone.net).cuda()
        self.backbone.net.load_state_dict(checkpoint["state"], strict=False)
        self.backbone.net = self.backbone.net.module

    def load_nemo_checkpoint(self, path_nemo_checkpoint):
        checkpoint = torch.load(path_nemo_checkpoint)
        self.load_state_dict(checkpoint["net_state_dict"])

    @staticmethod
    def create_by_name(name: str):
        config = od3d.io.read_config_intern(
            rfpath=Path("methods/model").joinpath(f"{name}.yaml"),
        )
        return OD3D_Model(config)

    def forward(self, x: torch.Tensor, return_backbone_output=False, *args, **kwargs):
        if self.config.backbone is not None:
            output = self.backbone(x, *args, **kwargs)
        else:
            output = x

        if self.config.head is not None:
            if return_backbone_output:
                return output, self.head(output)
            else:
                return self.head(output)
        else:
            return output

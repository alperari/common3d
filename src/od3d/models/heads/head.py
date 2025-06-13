from typing import List

from omegaconf import DictConfig
from torch import nn
from od3d.data.batch_datatypes import OD3D_ModelData

class OD3D_Head(nn.Module):
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self, in_dims: List, in_upsample_scales: List, config: DictConfig):
        super().__init__()
        self.config = config
        self.in_dims = in_dims
        self.in_upsample_scales = in_upsample_scales
        self.normalize = config.get("normalize", False)
        self.downsample_rate = None

    def forward(self, x: OD3D_ModelData):
        x.featmap = x.featmaps[-1]
        return x

import logging

logger = logging.getLogger(__name__)
import torch.nn as nn
from od3d.models.heads.head import OD3D_Head
from od3d.data.batch_datatypes import OD3D_ModelData
from typing import List
from omegaconf import DictConfig


class OD3D_Norm_Detach(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # ... x F
        x = x / ((x.norm(dim=-1, keepdim=True) + 1e-10).detach())
        return x


class OD3D_Norm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # ... x F
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-10)
        return x

def get_activation(name, inplace=True, lrelu_param=0.2):
    if name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'norm':
        return OD3D_Norm()
    elif name == 'norm_detach':
        return OD3D_Norm_Detach()
    elif name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        return nn.LeakyReLU(lrelu_param, inplace=inplace)
    elif name == 'none':
        return nn.Identity()
    else:
        raise NotImplementedError

class MLP(OD3D_Head):

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

        if self.config.get('resnet', None) is not None:
            from od3d.models.heads.resnet import ResNet
            if in_dims is None:
                in_dims = [config.in_dim, ]
                in_upsample_scales = [ ]
            self.resnet = ResNet(
                in_dims=in_dims,
                in_upsample_scales=in_upsample_scales,
                config=config.resnet,
            )
            self.in_dim = self.resnet.out_dim
        else:
            self.resnet = None
            if in_dims is not None:
                self.in_dim = in_dims[-1]
            else:
                self.in_dim = config.in_dim

        # def __init__(self, cin, cout, num_layers, nf=256, dropout=0, activation=None):
        #     super().__init__()
        num_layers = config.num_layers
        hidden_dim = config.hidden_dim
        self.out_dim = config.out_dim
        dropout = config.dropout
        activation = config.activation


        assert num_layers >= 1
        if num_layers == 1:
            network = [nn.Linear(self.in_dim, self.out_dim, bias=False)]
        else:
            network = [nn.Linear(self.in_dim, hidden_dim, bias=False)]
            for _ in range(num_layers-2):
                network += [
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim, bias=False)]
                if dropout:
                    network += [nn.Dropout(dropout)]
            network += [
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.out_dim, bias=False)]
        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, x: OD3D_ModelData):
        # if len(x.featmaps) > 1:
        #     logger.warning("MLP head only process last feature map.")

        if self.resnet is not None and x.featmaps is not None:
            x_resnet = self.resnet(x)
            x_res = x_resnet.feat
        else:
            x_res = x.feat # [-1].flatten(1)  # BxF

        x_out = OD3D_ModelData(feat=self.network(x_res))
        return x_out
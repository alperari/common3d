import logging

logger = logging.getLogger(__name__)
from dataclasses import dataclass
from od3d.datasets.frame_meta import (
    OD3D_FrameMeta,
    OD3D_FrameMetaMeshsMixin,
    OD3D_FrameMetaCategoryMixin,
    OD3D_FrameMetaCategoriesMixin,
    OD3D_FrameMetaRGBMixin,
    OD3D_FrameMetaSizeMixin,
    OD3D_FrameMetaBBoxsMixin,
    OD3D_FrameMetaSubsetMixin,
    OD3D_FrameMetaCamTform4x4ObjMixin,
    OD3D_FrameMetaCamIntr4x4Mixin,
    OD3D_FrameMetaSequenceMixin,
)
from od3d.datasets.frame import OD3D_Frame


@dataclass
class ShapNet_FrameMeta(
    OD3D_FrameMetaCamTform4x4ObjMixin,
    OD3D_FrameMetaMeshsMixin,
    OD3D_FrameMetaBBoxsMixin,
    OD3D_FrameMetaCategoryMixin,
    OD3D_FrameMetaRGBMixin,
    OD3D_FrameMetaSubsetMixin,
    OD3D_FrameMetaCamIntr4x4Mixin,
    OD3D_FrameMetaSizeMixin,
    OD3D_FrameMetaSequenceMixin,
    OD3D_FrameMeta,
):
    pass
    # @property
    # def name_unique(self):
    #    return f"{self.subset}/{self.name}"

    # import h5py
    # f = h5py.File('/misc/lmbraid19/sommerl/datasets/ObjectNet3D/Annotations/n03761084_13592.mat')

    # @staticmethod
    # def load_from_raw(
    #    path_raw: Path,

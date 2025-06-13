import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass
from od3d.datasets.omni6dpose.frame import Omni6DPose_Frame
from od3d.datasets.sequence import (
    OD3D_SequenceMeshMixin,
    OD3D_Sequence,
    OD3D_SequenceCategoryMixin
)
from od3d.datasets.frame_meta import OD3D_FrameMetaSubsetMixin
from od3d.datasets.sequence_meta import (
    OD3D_SequenceMeta, OD3D_SequenceMetaMeshMixin
)
from od3d.datasets.object import (
    OD3D_MaskTypeMixin,
    OD3D_CamTform4x4ObjTypeMixin,
    OD3D_DepthTypeMixin,
)
from od3d.datasets.omni6dpose.enum import MAP_CATEGORIES_OMNI6DPOSE_TO_OD3D, OMNI6DPOSE_CATEGORIES

@dataclass
class Omni6DPose_SequenceMeta(OD3D_FrameMetaSubsetMixin, OD3D_SequenceMeta):
    @property
    def name_unique(self):
        return f"{self.subset}/{super().name_unique}"

@dataclass
class Omni6DPose_Sequence(
    OD3D_SequenceMeshMixin,
    OD3D_DepthTypeMixin,
    OD3D_MaskTypeMixin,
    OD3D_CamTform4x4ObjTypeMixin,
    OD3D_SequenceCategoryMixin,
    OD3D_Sequence,
):
    frame_type = Omni6DPose_Frame
    map_categories_to_od3d = MAP_CATEGORIES_OMNI6DPOSE_TO_OD3D
    meta_type = Omni6DPose_SequenceMeta


    def read_mesh(self, mesh_type=None, device="cpu", tform_obj_type=None):
        return self.first_frame.read_mesh(mesh_type=mesh_type, device=device, tform_obj_type=tform_obj_type)

    @property
    def fpath_mesh(self):
        return self.get_fpath_mesh()

    @property
    def category(self):
        cats = [cat for cat in OMNI6DPOSE_CATEGORIES if self.name.startswith(f"{cat}_")]
        if len(cats) > 0:
            return cats[0]
        else:
            return None

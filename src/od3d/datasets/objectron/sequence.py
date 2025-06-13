import logging
logger = logging.getLogger(__name__)

from dataclasses import dataclass
from od3d.datasets.sequence_meta import OD3D_SequenceMeta, OD3D_SequenceMetaCategoryMixin
from od3d.datasets.objectron.frame import Objectron_Frame
from od3d.datasets.objectron.enum import MAP_CATEGORIES_OBJECTRON_TO_OD3D
import re

@dataclass
class Objectron_SequenceMeta(OD3D_SequenceMetaCategoryMixin, OD3D_SequenceMeta):
    pass
    # def preprocess_mesh(self, override=False):
    #     if self.fpath_mesh.exists() and not override:
    #         logger.warning(f"mesh already exists {self.fpath_mesh}")
    #         return
    #     else:
    #         logger.info(
    #             f"preprocessing mesh for {self.name_unique} with type {self.mesh_type}",
    #         )
    #
    #     match = re.match(r"([a-z]+)([0-9]+)", self.mesh_type, re.I)
    #     if match and len(match.groups()) == 2:
    #         mesh_type, mesh_vertices_count = match.groups()
    #         mesh_vertices_count = int(mesh_vertices_count)
    #     else:
    #         msg = f"could not retrieve mesh type and vertices count from mesh name {self.mesh_type}"
    #         raise Exception(msg)
    #
    # pass
    # pcl_pts_count: int
    # pcl_quality_score: float
    # rfpath_pcl: Path
    # viewpoint_quality_score: float



    # @staticmethod
    # def load_from_raw(sequence_annotation: SequenceAnnotation):
    #     name = sequence_annotation.sequence_name
    #     category = sequence_annotation.category
    #     if sequence_annotation.point_cloud is not None:
    #         rfpath_pcl = sequence_annotation.point_cloud.path
    #         pcl_pts_count = sequence_annotation.point_cloud.n_points
    #         pcl_quality_score = sequence_annotation.point_cloud.quality_score
    #     else:
    #         rfpath_pcl = Path("None")
    #         pcl_pts_count = 0
    #         pcl_quality_score = float("nan")
    #
    #     viewpoint_quality_score = sequence_annotation.viewpoint_quality_score
    #
    #     return CO3D_SequenceMeta(
    #         category=category,
    #         name=name,
    #         pcl_pts_count=pcl_pts_count,
    #         pcl_quality_score=pcl_quality_score,
    #         rfpath_pcl=rfpath_pcl,
    #         viewpoint_quality_score=viewpoint_quality_score,
    #     )


from od3d.datasets.sequence import OD3D_Sequence, OD3D_SequenceCategoryMixin, OD3D_SequenceMeshMixin
from od3d.datasets.object import OD3D_MaskTypeMixin, OD3D_CamTform4x4ObjTypeMixin, OD3D_FrameKpts2d3dTypeMixin

@dataclass
class Objectron_Sequence(
    OD3D_SequenceMeshMixin,
    OD3D_SequenceCategoryMixin,
    OD3D_MaskTypeMixin,
    OD3D_CamTform4x4ObjTypeMixin,
    OD3D_FrameKpts2d3dTypeMixin,
    OD3D_Sequence,
):
    frame_type = Objectron_Frame
    map_categories_to_od3d = MAP_CATEGORIES_OBJECTRON_TO_OD3D
    meta_type = Objectron_SequenceMeta
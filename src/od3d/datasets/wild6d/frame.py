import logging

logger = logging.getLogger(__name__)
import torch
from pathlib import Path
import os
import json
import numpy as np

from od3d.datasets.frame_meta import (
    OD3D_FrameMeta,
    OD3D_FrameMetaRGBMixin,
    OD3D_FrameMetaDepthMixin,
    OD3D_FrameMetaCategoryMixin,
    OD3D_FrameMetaMaskMixin,
    OD3D_FrameMetaSizeMixin,
    OD3D_FrameMetaSequenceMixin,
    OD3D_FrameMetaDepthMaskMixin,
    OD3D_FrameMetaCamIntr4x4Mixin,
    OD3D_FrameMetaCamTform4x4ObjMixin,
    OD3D_FrameMetaKpts2D3DMixin,
)

from od3d.datasets.frame import (
    OD3D_Frame,
    OD3D_FrameSizeMixin,
    OD3D_FrameRGBMixin,
    OD3D_FrameMaskMixin,
    OD3D_FrameRGBMaskMixin,
    OD3D_FrameRaysCenter3dMixin,
    OD3D_FrameMeshMixin,
    OD3D_FramePCLMixin,
    OD3D_FrameTformObjMixin,
    OD3D_FrameSequenceMixin,
    OD3D_FrameCategoryMixin,
    OD3D_CamProj4x4ObjMixin,
    OD3D_FrameDepthMixin,
    OD3D_FrameDepthMaskMixin,
    OD3D_FrameKpts2d3dMixin,
)
from dataclasses import dataclass
from od3d.datasets.object import (
    OD3D_PCLTypeMixin,
    OD3D_MeshTypeMixin,
    OD3D_SequenceSfMTypeMixin,
)
from od3d.datasets.wild6d.enum import (
    MAP_CATEGORIES_WILD6D_TO_OD3D,
    MAP_CATEGORIES_OD3D_TO_WILD6D,
)

# from od3d.datasets.wild6d.enum import CO3D_FRAME_TYPES
from od3d.cv.io import read_co3d_depth_image

# from co3d.dataset.data_types import (
#    FrameAnnotation,
# )
from od3d.cv.geometry.transform import transf4x4_from_rot3x3_and_transl3


def get_3d_bbox(size, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar
    Returns:
        bbox_3d: [3, N]

    """
    bbox_3d = (
        np.array(
            [
                [+size[0] / 2, +size[1] / 2, +size[2] / 2],
                [+size[0] / 2, +size[1] / 2, -size[2] / 2],
                [-size[0] / 2, +size[1] / 2, +size[2] / 2],
                [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                [-size[0] / 2, -size[1] / 2, -size[2] / 2],
            ]
        )
        + shift
    )
    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def transform_coordinates_3d(coordinates, sRT):
    """
    Args:
        coordinates: [3, N]
        sRT: [4, 4]

    Returns:
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack(
        [coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)]
    )
    new_coordinates = sRT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates


def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Args:
        coordinates_3d: [3, N]
        intrinsics: [3, 3]

    Returns:
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


@dataclass
class WILD6D_FrameMeta(
    OD3D_FrameMetaCamIntr4x4Mixin,
    OD3D_FrameMetaCamTform4x4ObjMixin,
    OD3D_FrameMetaDepthMixin,
    OD3D_FrameMetaMaskMixin,
    OD3D_FrameMetaRGBMixin,
    OD3D_FrameMetaSizeMixin,
    OD3D_FrameMetaCategoryMixin,
    OD3D_FrameMetaSequenceMixin,
    OD3D_FrameMetaKpts2D3DMixin,
    OD3D_FrameMeta,
):
    depth_scale: float
    # co3d_frame_type: CO3D_FRAME_TYPES

    @staticmethod
    def get_rfpath_frame_meta_with_category_sequence_and_frame_name(
        category: str,
        sequence_name: str,
        name: str,
    ):
        return WILD6D_FrameMeta.get_rfpath_metas().joinpath(
            category,
            sequence_name,
            name + ".yaml",
        )

    @staticmethod
    def get_fpath_frame_meta_with_category_sequence_and_frame_name(
        path_meta: Path,
        category: str,
        sequence_name: str,
        name: str,
    ):
        return path_meta.joinpath(
            WILD6D_FrameMeta.get_rfpath_frame_meta_with_category_sequence_and_frame_name(
                category=category,
                sequence_name=sequence_name,
                name=name,
            ),
        )

    @staticmethod
    def load_from_raw(frame_annotation, path_raw: Path):
        MAP_CATEGORIES_WILD6D_TO_OD3D
        cls_n, seq_idx, obj_idx, frame_idx = frame_annotation["name"].split("/")
        frame_name = str(frame_idx)

        category = str(MAP_CATEGORIES_OD3D_TO_WILD6D[cls_n])
        sequence_name = f"{obj_idx}-{seq_idx}"

        import json

        frame_annotation_intr_fpath = path_raw.joinpath(
            f"test_set/{category}", seq_idx, obj_idx, "metadata"
        )
        frame_annotation_intr = json.load(open(frame_annotation_intr_fpath, "rb"))

        # load 6d pose annotations
        scale = frame_annotation["size"]
        rot = frame_annotation["rotation"]
        trans = frame_annotation["translation"]
        RTs = np.eye(4)
        RTs[:3, :3] = rot
        RTs[:3, 3] = trans

        cam_tform4x4_obj = torch.from_numpy(RTs).float()

        K = np.array(frame_annotation_intr["K"]).reshape(3, 3).T
        cam_intr4x4 = torch.eye(4)
        cam_intr4x4[:3, :3] = torch.from_numpy(K).float()

        noc_cube = get_3d_bbox(scale, 0)
        bbox_3d = transform_coordinates_3d(noc_cube, RTs)
        projected_bbox = calculate_2d_projections(bbox_3d, K)

        l_kpts2d_annot = projected_bbox.to_list()  # : List[List[float]]
        l_kpts3d = noc_cube.to_list()  # List[List[float]]

        l_kpts2d_annot_vsbl = list((True,) * len(l_kpts3d))
        kpts_names = [
            "left_back_bottom",
            "left_front_bottom",
            "left_back_top",
            "left_front_top",
            "right_back_bottom",
            "right_front_bottom",
            "right_back_top",
            "right_front_top",
        ]

        # category = frame_annotation.image.path.split("/")[0]
        # sequence_name = frame_annotation.sequence_name
        # if frame_annotation.meta is not None:
        #    co3d_frame_type = CO3D_FRAME_TYPES(frame_annotation.meta["frame_type"])
        # else:
        #   co3d_frame_type = CO3D_FRAME_TYPES.CO3DV1
        name = f"{frame_annotation.frame_number}"

        rfpath_rgb = Path(f"test_set/{category}").joinpath(
            seq_idx, obj_idx, "images", f"{str(int(frame_idx))}.jpg"
        )
        rfpath_mask = Path(f"test_set/{category}").joinpath(
            seq_idx, obj_idx, "images", f"{str(int(frame_idx))}-mask.png"
        )
        rfpath_depth = Path(f"test_set/{category}").joinpath(
            seq_idx, obj_idx, "images", f"{str(int(frame_idx))}-depth.png"
        )

        depth_scale = frame_annotation.depth.scale_adjustment

        l_size = [int(frame_annotation_intr["h"]), int(frame_annotation_intr["w"])]
        l_cam_intr4x4 = cam_intr4x4.tolist()
        l_cam_tform4x4_obj = cam_tform4x4_obj.tolist()
        return WILD6D_FrameMeta(
            rfpath_rgb=rfpath_rgb,
            category=category,
            sequence_name=sequence_name,
            l_size=l_size,
            name=name,
            rfpath_mask=rfpath_mask,
            rfpath_depth=rfpath_depth,
            l_cam_intr4x4=l_cam_intr4x4,
            l_cam_tform4x4_obj=l_cam_tform4x4_obj,
            depth_scale=depth_scale,
            # co3d_frame_type=co3d_frame_type,
            l_kpts2d_annot=l_kpts2d_annot,
            l_kpts2d_annot_vsbl=l_kpts2d_annot_vsbl,
            l_kpts3d=l_kpts3d,
            kpts_names=kpts_names,
        )


@dataclass
class WILD6D_Frame(
    # OD3D_FramePCLMixin,
    # OD3D_FrameMeshMixin,
    OD3D_FrameRaysCenter3dMixin,
    OD3D_FrameMetaKpts2D3DMixin,
    OD3D_FrameTformObjMixin,
    OD3D_CamProj4x4ObjMixin,
    OD3D_FrameRGBMaskMixin,
    OD3D_FrameMaskMixin,
    OD3D_FrameRGBMixin,
    OD3D_FrameDepthMixin,
    # OD3D_FrameDepthMaskMixin,
    OD3D_FrameCategoryMixin,
    OD3D_FrameSequenceMixin,
    OD3D_FrameSizeMixin,
    # OD3D_MeshTypeMixin,
    # OD3D_PCLTypeMixin,
    # OD3D_SequenceSfMTypeMixin,
    OD3D_Frame,
):
    meta_type = WILD6D_FrameMeta
    map_categories_to_od3d = MAP_CATEGORIES_WILD6D_TO_OD3D

    def __post_init__(self):
        # hack: prevents circular import
        from od3d.datasets.wild6d.sequence import WILD6D_Sequence

        self.sequence_type = WILD6D_Sequence

    def get_depth(self):
        if self.depth is None:
            self.depth = read_co3d_depth_image(self.fpath_depth) * self.meta.depth_scale
        return self.depth

import logging

logger = logging.getLogger(__name__)
import torch
from pathlib import Path
from dataclasses import dataclass
from od3d.datasets.frame import OD3D_FrameMeta, OD3D_Frame
from od3d.cv.geometry.transform import transf4x4_from_spherical
from od3d.io import read_json
import math
import scipy
import numpy as np
from typing import List
from od3d.datasets.pascal3d.enum import (
    PASCAL3D_SCALE_NORMALIZE_TO_REAL,
)
from od3d.datasets.frame_meta import (
    OD3D_FrameMeta,
    #    OD3D_FrameMetaMeshMixin,
    OD3D_FrameMetaCategoryMixin,
    OD3D_FrameMetaRGBSMixin,
    OD3D_FrameMetaSizesMixin,
    OD3D_FrameMetaKpts2DSMixin,
    OD3D_FrameMetaBBoxsMixin,
    OD3D_FrameMetaSubsetMixin,
    OD3D_FrameMetaMaskMixin,
    OD3D_FrameMetaCamTform4x4ObjsMixin,
    OD3D_FrameMetaCamIntr4x4sMixin,
)


@dataclass
class SPair71KFrameMeta(
    OD3D_FrameMetaBBoxsMixin,
    OD3D_FrameMetaSubsetMixin,
    OD3D_FrameMetaKpts2DSMixin,
    OD3D_FrameMetaCategoryMixin,
    OD3D_FrameMetaCamTform4x4ObjsMixin,
    OD3D_FrameMetaCamIntr4x4sMixin,
    OD3D_FrameMetaRGBSMixin,
    OD3D_FrameMetaSizesMixin,
    OD3D_FrameMeta,
    # OD3D_FrameMetaMaskMixin
):
    @property
    def name_unique(self):
        return f"{self.subset}/{self.category}/{self.name}"

    @staticmethod
    def get_name_unique_from_category_subset_name(category, subset, name):
        return f"{subset}/{category}/{name}"

    @staticmethod
    def get_cams_from_annotation(annotation):
        name = annotation["record"]["filename"][0][0][0].split(".")[0]

        objects = annotation["record"]["objects"][0][0][0]

        objects = list(
            filter(
                lambda obj: hasattr(obj, "dtype")
                and obj.dtype.names is not None
                and "viewpoint" in obj.dtype.names
                and hasattr(obj["viewpoint"], "dtype")
                and obj["viewpoint"].dtype.names is not None,
                objects,
            ),
        )
        # assert len(objects) == 1
        if len(objects) < 1:
            incomplete_reason = f"num objects = {len(objects)}"
            logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
            return None

        object = objects[0]
        viewpoint = object["viewpoint"]
        azimuth = viewpoint["azimuth"][0][0][0][0] * math.pi / 180
        elevation = viewpoint["elevation"][0][0][0][0] * math.pi / 180
        distance = viewpoint["distance"][0][0][0][0]
        focal = viewpoint["focal"][0][0][0][0]
        if (azimuth + elevation + distance + focal) == 0:
            logger.warning(f"Skip frame {name}, due to incomplete viewpoint.")
            return None

        theta = viewpoint["theta"][0][0][0][0] * math.pi / 180
        principal = np.array(
            [
                viewpoint["px"][0][0][0][0],
                viewpoint["py"][0][0][0][0],
            ],
        )
        viewport = viewpoint["viewport"][0][0][0][0]

        cam_tform4x4_obj = SPair71KFrame.calc_cam_tform_obj(
            azimuth=azimuth,
            elevation=elevation,
            theta=theta,
            distance=distance,
        )
        # cam_tform4x4_obj = torch.from_numpy(cam_tform4x4_obj)

        cam_intr3x3 = np.array(
            [
                [1.0 * viewport * focal, 0, principal[0]],
                [0, 1.0 * viewport * focal, principal[1]],
                [0, 0, 1.0],
            ],
        )
        cam_intr4x4 = np.hstack((cam_intr3x3, [[0], [0], [0]]))
        cam_intr4x4 = np.vstack((cam_intr4x4, [0, 0, 0, 1]))
        cam_intr4x4 = torch.from_numpy(cam_intr4x4).to(dtype=cam_tform4x4_obj.dtype)

        return cam_tform4x4_obj, cam_intr4x4

    @staticmethod
    def load_bbox_kpts2d_cam_from_object_annotation_raw(
        pair_annotation,
        source_annotation,
        target_annotation,
        category=None,
    ):
        if category is None:
            category = pair_annotation["category"]

        source_cam = SPair71KFrameMeta.get_cams_from_annotation(source_annotation)
        if source_cam is not None:
            source_cam_tform4x4_obj, source_cam_intr4x4 = source_cam
        else:
            return None

        target_cam = SPair71KFrameMeta.get_cams_from_annotation(target_annotation)
        if target_cam is not None:
            target_cam_tform4x4_obj, target_cam_intr4x4 = target_cam
        else:
            return None
        cam_tform4x4_objs = [
            source_cam_tform4x4_obj.tolist(),
            target_cam_tform4x4_obj.tolist(),
        ]
        cam_intr4x4s = [source_cam_intr4x4.tolist(), target_cam_intr4x4.tolist()]

        bboxs = [pair_annotation["src_bndbox"], pair_annotation["trg_bndbox"]]

        kpts2d_annots = [pair_annotation["src_kps"], pair_annotation["trg_kps"]]
        # kpts2d_annot_dict_filtered = { int(kp_id) : kp for kp_id , kp in kpts2d_annot_dict.items()  if kp is not None }

        kpts2d_annots_ids = [int(kps) for kps in pair_annotation["kps_ids"]]  # N

        return (
            category,
            bboxs,
            kpts2d_annots,
            kpts2d_annots_ids,
            cam_tform4x4_objs,
            cam_intr4x4s,
        )

    @staticmethod
    def load_from_raw_annotation(
        pair_annotation,
        subset: str,
        category: str,
        path_raw: Path,
        rfpath_rgbs: Path,
        source_annotation,
        target_annotation,
    ):
        name = pair_annotation["filename"].split(":", 1)[0]

        sizes = torch.Tensor(
            [pair_annotation["src_imsize"][:2], pair_annotation["trg_imsize"][:2]],
        )
        sizes = torch.flip(sizes, [1])

        frame_meta = SPair71KFrameMeta.load_bbox_kpts2d_cam_from_object_annotation_raw(
            pair_annotation=pair_annotation,
            source_annotation=source_annotation,
            target_annotation=target_annotation,
        )
        if frame_meta is not None:
            (
                category,
                bboxs,
                kpts2d_annots,
                kpts2d_annots_ids,
                cam_tform4x4_objs,
                cam_intr4x4s,
            ) = frame_meta
        else:
            return None

        return SPair71KFrameMeta(
            subset=subset,
            name=name,
            rfpath_rgb=rfpath_rgbs,
            l_bboxs=bboxs,
            l_kpts2d_annots=kpts2d_annots,
            l_kpts2d_annots_ids=kpts2d_annots_ids,
            l_sizes=sizes.tolist(),
            category=category,
            l_cam_tform4x4_objs=cam_tform4x4_objs,
            l_cam_intr4x4s=cam_intr4x4s,
        )

    @staticmethod
    def load_from_raw(
        path_raw_pascal3d: Path,
        pair_name: str,
        subset: str,
        category: str,
        path_raw: Path,
    ):
        pair_rfpath = f"{subset}/{pair_name}:{category}"
        # assert category == pair_name.split(":")[-1] , f"category mismatch {category} , {pair_name}"
        rfpath_annotation = Path("PairAnnotation").joinpath(f"{pair_rfpath}.json")
        src_name = pair_name.split("-", 3)[1]
        trg_name = pair_name.split("-", 3)[2]
        rfpath_src_annotation = path_raw_pascal3d.joinpath(
            f"Annotations/{category}_pascal/{src_name}.mat",
        )
        rfpath_trg_annotation = path_raw_pascal3d.joinpath(
            f"Annotations/{category}_pascal/{trg_name}.mat",
        )
        source_annotation = scipy.io.loadmat(rfpath_src_annotation)
        target_annotation = scipy.io.loadmat(rfpath_trg_annotation)

        pair_annotation = read_json(path_raw.joinpath(rfpath_annotation))
        rfpath_rgbs = [
            Path("JPEGImages").joinpath(f"{category}/{pair_annotation['src_imname']}"),
            Path("JPEGImages").joinpath(f"{category}/{pair_annotation['trg_imname']}"),
        ]

        return SPair71KFrameMeta.load_from_raw_annotation(
            pair_annotation=pair_annotation,
            rfpath_rgbs=rfpath_rgbs,
            category=category,
            subset=subset,
            path_raw=path_raw,
            source_annotation=source_annotation,
            target_annotation=target_annotation,
        )


from od3d.datasets.frame import (
    OD3D_FrameRGBSMixin,
    # OD3D_FrameDepthMixin,
    # OD3D_FrameDepthMaskMixin,
    OD3D_FrameCategoryMixin,
    OD3D_FrameSizesMixin,
    OD3D_Frame,
    OD3D_FrameBBoxsMixin,
    OD3D_FrameKpts2dIDMixin,
    OD3D_FrameRGBMaskMixin,
    OD3D_CamProj4x4ObjsMixin,
    OD3D_FrameTformObjsMixin,
)

from od3d.datasets.object import OD3D_MESH_TYPES
from od3d.cv.geometry.transform import inv_tform4x4, tform4x4


@dataclass
class SPair71KFrame(
    OD3D_FrameBBoxsMixin,
    # OD3D_FrameMeshMixin,
    OD3D_FrameTformObjsMixin,
    OD3D_FrameKpts2dIDMixin,
    OD3D_CamProj4x4ObjsMixin,
    OD3D_FrameRGBMaskMixin,
    #  OD3D_FrameMaskMixin,
    OD3D_FrameRGBSMixin,
    # OD3D_FrameDepthMixin,
    # OD3D_FrameDepthMaskMixin,
    OD3D_FrameCategoryMixin,
    OD3D_FrameSizesMixin,
    OD3D_Frame,
):
    meta_type = SPair71KFrameMeta

    def read_cam_tform4x4_objs_raw(self):
        cam_tform4x4_objs = torch.Tensor(self.meta.cam_tform4x4_objs)
        cam_tform4x4_objs[:, :3, 3] *= PASCAL3D_SCALE_NORMALIZE_TO_REAL[self.category]
        return cam_tform4x4_objs

    def get_fpath_tform_obj(self, tform_obj_type=None):
        if tform_obj_type is None:
            tform_obj_type = self.tform_obj_type
        return self.path_preprocess.joinpath(
            "tform_obj",
            f"{tform_obj_type}",
            "tform_obj.pt",
        )

    def read_cam_tform4x4_objs(self, cam_tform4x4_obj_type=None, tform_obj_type=None):
        cam_tform4x4_objs = self.read_cam_tform4x4_objs_raw()

        tform_obj = self.get_tform_obj(tform_obj_type=tform_obj_type)
        if tform_obj is not None:
            cam_tform4x4_objs = tform4x4(cam_tform4x4_objs, inv_tform4x4(tform_obj))

        # note: note alignment of droid slam may include scale, therefore remove this scale.
        # note: projection does not change as we scale the depth z to the object as well
        scale = (
            cam_tform4x4_objs[:, :3, :3]
            .norm(dim=-1, keepdim=True)
            .mean(dim=-2, keepdim=True)
        )
        cam_tform4x4_objs[:, :3] = cam_tform4x4_objs[:, :3] / scale

        if (
            cam_tform4x4_obj_type is None
            or cam_tform4x4_obj_type == self.cam_tform4x4_obj_type
        ) and (tform_obj_type == self.tform_obj_type or tform_obj_type is None):
            self.cam_tform4x4_obj = cam_tform4x4_objs
        return cam_tform4x4_objs

    @staticmethod
    def calc_cam_tform_obj(azimuth, elevation, theta, distance):
        cam_tform4x4_obj = transf4x4_from_spherical(
            azim=torch.Tensor([azimuth]),
            elev=torch.Tensor([elevation]),
            theta=torch.Tensor([theta]),
            dist=torch.Tensor([distance]),
        )[0]
        # cam_tform4x4_obj[0, :] = cam_tform4x4_obj[0, :]
        # cam_tform4x4_obj[1, :] = -cam_tform4x4_obj[1, :]
        # cam_tform4x4_obj[2, :] = -cam_tform4x4_obj[2, :]
        # cam_tform4x4_obj[2, 2:3] = -cam_tform4x4_obj[2, 2:3]
        return cam_tform4x4_obj

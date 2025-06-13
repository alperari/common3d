import logging

import od3d.io

logger = logging.getLogger(__name__)
from od3d.datasets.shapenet.enum import (
    MAP_CATEGORIES_SHAPENET_V2_ID_TO_CATEGORY,
    MAP_CATEGORIES_SHAPENET_R2N2_ID_TO_CATEGORY,
    MAP_CATEGORIES_SHAPENET_ID_TO_CATEGORY,
    MAP_CATEGORIES_SHAPENET_CATEGORY_TO_ID,
    MAP_CATEGORIES_SHAPENET_TO_OD3D,
    SHAPENET_CATEGORIES,
    MAP_CATEGORIES_OD3D_TO_SHAPENET,
    SHAPENET_SUBSETS,
    LIST_CATEGORIES_SHAPENET_CANOC,
)

from od3d.datasets.dataset import OD3D_Dataset, OD3D_SequenceDataset
from od3d.datasets.frame_meta import (
    OD3D_FrameMeta,
    OD3D_FrameMetaRGBMixin,
    OD3D_FrameMetaCategoryMixin,
    OD3D_FrameMetaMeshMixin,
    OD3D_FrameMetaMaskMixin,
    OD3D_FrameMetaSizeMixin,
    OD3D_FrameMetaCamIntr4x4Mixin,
    OD3D_FrameMetaCamTform4x4ObjMixin,
    OD3D_FrameMetaPCLMixin,
    OD3D_FrameMetaDepthMixin,
    OD3D_FrameMetaDepthMaskMixin,
    OD3D_FrameMetaSubsetMixin,
    OD3D_FrameMetaSequenceMixin,
)
from od3d.datasets.frame import (
    OD3D_FRAME_MODALITIES,
    OD3D_Frame,
    OD3D_FrameMeshMixin,
    OD3D_CamProj4x4ObjMixin,
    OD3D_FrameRGBMaskMixin,
    OD3D_FrameMaskMixin,
    OD3D_FrameRGBMixin,
    OD3D_FrameDepthMixin,
    OD3D_FrameDepthMaskMixin,
    OD3D_FrameCategoryMixin,
    OD3D_FrameSequenceMixin,
    OD3D_FrameSizeMixin,
    OD3D_MeshTypeMixin,
    OD3D_FramePCLMixin,
    OD3D_FrameMeshMixin,
    OD3D_FrameRaysCenter3dMixin,
    OD3D_FrameTformObjMixin,
    OD3D_FrameSubsetMixin,
    OD3D_FrameMeta,
)

from od3d.datasets.sequence_meta import (
    OD3D_SequenceMeta,
    OD3D_SequenceMetaCategoryMixin,
    OD3D_SequenceMetaMeshMixin,
    OD3D_SequenceMetaPCLMixin,
)
from od3d.datasets.sequence import (
    OD3D_SequenceMeshMixin,
    OD3D_SequenceCategoryMixin,
    OD3D_Sequence,
)

from od3d.cv.geometry.transform import transf3d_broadcast
from od3d.cv.geometry.objects3d.meshes import Meshes
from od3d.datasets.frame import (
    OD3D_MaskTypeMixin,
    OD3D_CamTform4x4ObjTypeMixin,
    OD3D_DepthTypeMixin,
)

# from od3d.datasets.objectnet3d.enum import OBJECTNET3D_CATEOGORIES
from pathlib import Path
from typing import List, Dict
from omegaconf import DictConfig
import shutil
import time

from dataclasses import dataclass

from od3d.datasets.object import (
    OD3D_FRAME_MASK_TYPES,
    OD3D_CAM_TFORM_OBJ_TYPES,
    OD3D_MESH_TYPES,
    OD3D_MESH_FEATS_TYPES,
    OD3D_MESH_FEATS_DIST_REDUCE_TYPES,
    OD3D_TFROM_OBJ_TYPES,
    OD3D_SEQUENCE_SFM_TYPES,
    OD3D_PCL_TYPES,
    OD3D_FRAME_DEPTH_TYPES,
    OD3D_PCLTypeMixin,
    OD3D_SequenceSfMTypeMixin,
    OD3D_SubsetMixin,
)


@dataclass
class ShapeNet_FrameMeta(
    # OD3D_FrameMetaPCLMixin,
    # OD3D_FrameMetaMeshMixin,
    OD3D_FrameMetaDepthMixin,
    OD3D_FrameMetaDepthMaskMixin,
    OD3D_FrameMetaMaskMixin,
    OD3D_FrameMetaCategoryMixin,
    OD3D_FrameMetaCamTform4x4ObjMixin,
    OD3D_FrameMetaCamIntr4x4Mixin,
    OD3D_FrameMetaRGBMixin,
    OD3D_FrameMetaSizeMixin,
    OD3D_FrameMetaSubsetMixin,
    OD3D_FrameMetaSequenceMixin,
    OD3D_FrameMeta,
):
    @property
    def name_unique(self):
        return f"{self.subset}/{super().name_unique}"

    @property
    def sequence_name_unique(self):
        return f"{self.subset}/{super().sequence_name_unique}"

    # @property
    # def name_unique(self):
    #    return f"{self.category}/{self.name}"

    @staticmethod
    def load_from_raw():
        pass


@dataclass
class ShapeNet_SequenceMeta(
    OD3D_SequenceMetaPCLMixin,
    OD3D_SequenceMetaMeshMixin,
    OD3D_SequenceMetaCategoryMixin,
    OD3D_FrameMetaSubsetMixin,
    OD3D_SequenceMeta,
):
    @property
    def name_unique(self):
        return f"{self.subset}/{super().name_unique}"

    @staticmethod
    def load_from_raw():
        pass


@dataclass
class ShapeNet_Frame(
    OD3D_FramePCLMixin,
    OD3D_FrameMeshMixin,
    OD3D_FrameRaysCenter3dMixin,
    OD3D_FrameTformObjMixin,
    OD3D_CamProj4x4ObjMixin,
    OD3D_FrameRGBMaskMixin,
    OD3D_FrameMaskMixin,
    OD3D_FrameRGBMixin,
    OD3D_FrameDepthMixin,
    OD3D_FrameDepthMaskMixin,
    OD3D_FrameCategoryMixin,
    OD3D_FrameSequenceMixin,
    OD3D_FrameSizeMixin,
    OD3D_MeshTypeMixin,
    OD3D_PCLTypeMixin,
    OD3D_SequenceSfMTypeMixin,
    OD3D_Frame,
):
    meta_type = ShapeNet_FrameMeta
    map_categories_to_od3d = MAP_CATEGORIES_SHAPENET_TO_OD3D

    def __post_init__(self):
        # hack: prevents circular import
        # from od3d.datasets.co3d.sequence import CO3D_Sequence

        self.sequence_type = ShapeNet_Sequence


@dataclass
class ShapeNet_Sequence(
    OD3D_SequenceMeshMixin,
    OD3D_SequenceCategoryMixin,
    OD3D_DepthTypeMixin,
    OD3D_MaskTypeMixin,
    OD3D_CamTform4x4ObjTypeMixin,
    OD3D_Sequence,
):
    frame_type = ShapeNet_Frame
    map_categories_to_od3d = MAP_CATEGORIES_SHAPENET_TO_OD3D
    meta_type = ShapeNet_SequenceMeta

    def read_mesh(self, mesh_type=None, device="cpu", tform_obj_type=None):
        mesh = Meshes.read_from_ply_file(
            fpath=self.get_fpath_mesh(mesh_type=mesh_type),
            device=device,
            load_texts=False,
        )

        tform_obj = self.get_tform_obj(tform_obj_type=tform_obj_type, center=False)
        if tform_obj is not None:
            tform_obj = tform_obj.to(device=device)
            mesh.verts = transf3d_broadcast(pts3d=mesh.verts, transf4x4=tform_obj)

        if (mesh_type is None or mesh_type == self.mesh_type) and (
            tform_obj_type is None or tform_obj_type == self.tform_obj_type
        ):
            self.mesh = mesh
        return mesh


class ShapeNet(OD3D_SequenceDataset):
    all_categories = list(SHAPENET_CATEGORIES)
    map_od3d_categories = MAP_CATEGORIES_OD3D_TO_SHAPENET
    sequence_type = ShapeNet_Sequence
    frame_type = ShapeNet_Frame

    cam_tform_obj_type = OD3D_CAM_TFORM_OBJ_TYPES.META
    mask_type = OD3D_FRAME_MASK_TYPES.META
    mesh_type = OD3D_MESH_TYPES.CUBOID500
    mesh_feats_type = (
        OD3D_MESH_FEATS_TYPES.M_DINOV2_VITB14_FROZEN_BASE_NO_NORM_T_CENTERZOOM512_R_ACC
    )
    mesh_feats_dist_reduce_type = OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN_AVG

    tform_obj_type = OD3D_TFROM_OBJ_TYPES.RAW
    sfm_type = OD3D_SEQUENCE_SFM_TYPES.META
    pcl_type = OD3D_PCL_TYPES.META  # POISSON_DISK_FPS
    depth_type = OD3D_FRAME_DEPTH_TYPES.META  #
    dict_nested_frames_struct = "subset/category/sequence/frame"

    def __init__(
        self,
        name: str,
        modalities: List[OD3D_FRAME_MODALITIES],
        path_raw: Path,
        path_preprocess: Path,
        categories: List[SHAPENET_CATEGORIES] = None,
        dict_nested_frames: Dict[str, Dict[str, List[str]]] = None,
        dict_nested_frames_ban: Dict[str, Dict[str, List[str]]] = None,
        frames_count_max_per_sequence=None,
        sequences_count_max_per_category=None,
        transform=None,
        index_shift=0,
        subset_fraction=1.0,
        mesh_type=OD3D_MESH_TYPES.META,
        mesh_feats_type=OD3D_MESH_FEATS_TYPES.M_DINOV2_VITB14_FROZEN_BASE_NO_NORM_T_CENTERZOOM512_R_ACC,
        mesh_feats_dist_reduce_type=OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN_AVG,
        tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW,
    ):
        super().__init__(
            categories=categories,
            name=name,
            modalities=modalities,
            path_raw=path_raw,
            path_preprocess=path_preprocess,
            transform=transform,
            subset_fraction=subset_fraction,
            index_shift=index_shift,
            dict_nested_frames=dict_nested_frames,
            dict_nested_frames_ban=dict_nested_frames_ban,
            frames_count_max_per_sequence=frames_count_max_per_sequence,
            sequences_count_max_per_category=sequences_count_max_per_category,
        )

        self.mesh_type = mesh_type
        self.mesh_feats_type = mesh_feats_type
        self.mesh_feats_dist_reduce_type = mesh_feats_dist_reduce_type
        self.tform_obj_type = tform_obj_type

        # directories
        # images/texture/textureXX.jpg # 0...14
        # images/texture/texture0.png
        # models/
        #   model_normalized.json
        #   model_normalized.mtl
        #   model_normalized.obj
        #   model_normalized.solid.binvox
        #   model_normalized.surface.binvox

    # def get_item(self, item):
    #     frame_meta = ShapeNet_FrameMeta.load_from_meta_with_name_unique(
    #         path_meta=self.path_meta,
    #         name_unique=self.list_frames_unique[item],
    #     )
    #     return OD3D_Frame(
    #         path_raw=self.path_raw,
    #         path_preprocess=self.path_preprocess,
    #         path_meta=self.path_meta,
    #         meta=frame_meta,
    #         modalities=self.modalities,
    #         categories=self.categories,
    #     )

    def get_frame_by_name_unique(self, name_unique):
        return self.frame_type(
            path_raw=self.path_raw,
            path_preprocess=self.path_preprocess,
            name_unique=name_unique,
            all_categories=self.categories,
            mask_type=self.mask_type,
            cam_tform4x4_obj_type=self.cam_tform_obj_type,
            mesh_type=self.mesh_type,
            mesh_feats_type=self.mesh_feats_type,
            mesh_feats_dist_reduce_type=self.mesh_feats_dist_reduce_type,
            modalities=self.modalities,
            pcl_type=self.pcl_type,
            sfm_type=self.sfm_type,
            tform_obj_type=self.tform_obj_type,
            depth_type=self.depth_type,
        )

    def get_sequence_by_name_unique(self, name_unique):
        return self.sequence_type(
            name_unique=name_unique,
            path_raw=self.path_raw,
            path_preprocess=self.path_preprocess,
            all_categories=self.categories,
            mask_type=self.mask_type,
            cam_tform4x4_obj_type=self.cam_tform_obj_type,
            mesh_type=self.mesh_type,
            modalities=self.modalities,
            mesh_feats_type=self.mesh_feats_type,
            mesh_feats_dist_reduce_type=self.mesh_feats_dist_reduce_type,
            pcl_type=self.pcl_type,
            sfm_type=self.sfm_type,
            tform_obj_type=self.tform_obj_type,
            depth_type=self.depth_type,
        )

    # add subsets: R2N2, v1, v2

    @staticmethod
    def extract_meta(config: DictConfig):
        path = Path(config.path_raw)
        path_preprocess = Path(config.path_preprocess)
        path_meta = ShapeNet.get_path_meta(config=config)
        path_raw = Path(config.path_raw)

        subsets = config.get("dict_nested_frames", {}).keys()
        if len(subsets) == 0:
            subsets = list(SHAPENET_SUBSETS)

        import math

        sequences_count_max_per_category = config.get(
            "sequences_count_max_per_category",
            None,
        )
        if sequences_count_max_per_category is None:
            sequences_count_max_per_category = math.inf

        subsets_rendering = [
            SHAPENET_SUBSETS.R2N2_V2_ICO_UNI32_THETA_UNI1,
            SHAPENET_SUBSETS.R2N2NOTXT_V2_ICO_UNI16_THETA_UNI1,
            SHAPENET_SUBSETS.R2N2NOTXT_V2_ICO_UNI32_THETA_UNI1,
            SHAPENET_SUBSETS.R2N2NOTXT_V2_ICO_UNI64_THETA_UNI1,
            SHAPENET_SUBSETS.R2N2NOTXT_V2_ICO_UNI32_THETA_UNI2,
            SHAPENET_SUBSETS.R2N2NOTXT_V2_ICO_UNI16_THETA_UNI4,
            SHAPENET_SUBSETS.R2N2NOTXT_V2_ICOTRAJ100_UNI64_THETA_UNI1,
            SHAPENET_SUBSETS.R2N2NOTXT_V2_ICOTRAJ50_UNI32_THETA_UNI1,
            SHAPENET_SUBSETS.R2N2NOTXT_V2_ICOTRAJ25_UNI16_THETA_UNI1,
            SHAPENET_SUBSETS.R2N2NOTXT_V2_ICOTRAJ10_UNI8_THETA_UNI1,
            SHAPENET_SUBSETS.R2N2NOTXT_V2_ICO_RAND16_THETA_RAND1,
            SHAPENET_SUBSETS.R2N2NOTXT_V2_ICO_RAND32_THETA_RAND1,
            SHAPENET_SUBSETS.R2N2NOTXT_V2_ICO_RAND64_THETA_RAND1,
            SHAPENET_SUBSETS.R2N2NOTXT_V2_ICOREALTRAJ100_UNI64_THETA_UNI1,
            SHAPENET_SUBSETS.R2N2NOTXT_V2_ICOREALTRAJ50_UNI32_THETA_UNI1,
            SHAPENET_SUBSETS.R2N2NOTXT_V2_ICOREALTRAJ25_UNI16_THETA_UNI1,
            SHAPENET_SUBSETS.R2N2NOTXT_V2_ICOREALTRAJ10_UNI8_THETA_UNI1,
            SHAPENET_SUBSETS.R2N2_V2_ICOREALTRAJ10_UNI8_THETA_UNI1,
            SHAPENET_SUBSETS.R2N2_V2_ICOREALTRAJ25_UNI16_THETA_UNI1,
            SHAPENET_SUBSETS.R2N2_V2_ICOREALTRAJ50_UNI32_THETA_UNI1,
            SHAPENET_SUBSETS.R2N2_V2_ICOREALTRAJ100_UNI64_THETA_UNI1,
            SHAPENET_SUBSETS.R2N2_V2_ICOTRAJ10_UNI8_THETA_UNI1,
            SHAPENET_SUBSETS.R2N2_V2_ICOTRAJ25_UNI16_THETA_UNI1,
            SHAPENET_SUBSETS.R2N2_V2_ICOTRAJ50_UNI32_THETA_UNI1,
            SHAPENET_SUBSETS.R2N2_V2_ICOTRAJ100_UNI64_THETA_UNI1,
        ]

        for subset in subsets:
            if subset in subsets_rendering:
                logger.info(f"render and extract meta for subset {subset}...")
                # subset = SHAPENET_SUBSETS.R2N2_V2_ICO

                categories = list(SHAPENET_CATEGORIES)
                if config.get("dict_nested_frames", None) is not None:
                    if config.dict_nested_frames.get(subset, None) is not None:
                        categories = list(config.dict_nested_frames.get(subset).keys())

                import re

                match = re.match(
                    r"([a-z0-9]+)_v([0-9]+)_([a-z0-9]+)_([a-z]+)([0-9]+)_theta_([a-z]+)([0-9]+)",
                    subset,
                    re.I,
                )
                if match and len(match.groups()) == 7:
                    (
                        subset_type,
                        version,
                        viewpoints_type,
                        viewpoints_sample_type,
                        viewpoints_count,
                        theta_sample_type,
                        theta_count,
                    ) = match.groups()
                    viewpoints_count = int(viewpoints_count)
                    viewpoints_sample_uni = viewpoints_sample_type == "uni"
                    theta_sample_uni = theta_sample_type == "uni"
                    theta_count = int(theta_count)
                    version = int(version)
                else:
                    msg = f"could not retrieve subset_type, version, viewpoints_type, viewpoints_sample_type, viewpoints_count, theta_sample_type, theta_count from subset {subset}"
                    raise Exception(msg)

                # v2 subset R2
                # R2N2
                path_r2n2 = path_raw.joinpath("ShapeNetRendering")
                # categories_r2n2_id = list(MAP_CATEGORIES_SHAPENET_R2N2_ID_TO_CATEGORY.keys())

                for category_shp in LIST_CATEGORIES_SHAPENET_CANOC:
                    category = category_shp.value
                    category_id = MAP_CATEGORIES_SHAPENET_CATEGORY_TO_ID[category]
                    # category = MAP_CATEGORIES_SHAPENET_ID_TO_CATEGORY[category_id]
                    sequences_count = 0
                    if category not in categories:
                        logger.info(f"skip category {category}")
                        continue
                    logger.info(f"extract meta category {category}")
                    path_r2n2_category = path_r2n2.joinpath(category_id)
                    if path_r2n2_category.exists():
                        sequences_ids = [
                            path.name for path in list(path_r2n2_category.iterdir())
                        ]
                    else:
                        sequences_ids = [
                            path.name
                            for path in list(
                                path_raw.joinpath(f"{category_id}").iterdir(),
                            )
                        ]

                    for sequence_id in sequences_ids:
                        if sequences_count == sequences_count_max_per_category:
                            logger.info(
                                f"reached maximum count of sequences {sequences_count_max_per_category}",
                            )
                            break

                        logger.info(f"extract meta sequence {sequence_id}")
                        frame_fpath_pcl = path_raw.joinpath(
                            f"pcl/{subset}/{category}/{sequence_id}.ply",
                        )
                        frame_fpath_mesh_orig = path_raw.joinpath(
                            f"{category_id}/{sequence_id}/models/model_normalized.obj",
                        )
                        frame_fpath_mesh = frame_fpath_mesh_orig
                        # frame_fpath_mesh = path_raw.joinpath(f"{category_id}/{sequence_id}/models/model_single.glb")
                        # frame_fpath_mesh_orig = path_raw.joinpath(f"{category_id}/{sequence_id}/models/model.glb")

                        if not frame_fpath_mesh_orig.exists():
                            logger.info(
                                f"skip sequence {sequence_id} in {category} because does not exist in v2.",
                            )
                            continue

                        sequence_meta = ShapeNet_SequenceMeta(
                            subset=subset,
                            category=category,
                            name=sequence_id,
                            rfpath_pcl=frame_fpath_pcl.relative_to(path_raw),
                            rfpath_mesh=frame_fpath_mesh.relative_to(path_raw),
                        )

                        if sequence_meta.get_fpath(path_meta=path_meta).exists():
                            logger.info(
                                f"already extracted sequence {sequence_id} in {category}",
                            )
                            sequences_count = sequences_count + 1
                            continue

                        # pip install "pyglet<2"
                        import trimesh

                        _mesh_trimesh = trimesh.load(
                            frame_fpath_mesh_orig,
                            force="mesh",
                        )

                        if subset_type == "r2n2notxt":
                            import numpy as np

                            # Define a single grey color (RGBA format)
                            grey_color = np.array([64, 64, 64, 255], dtype=np.uint8)
                            # Apply the color to all faces/vertices
                            _mesh_trimesh.visual = trimesh.visual.color.ColorVisuals(
                                _mesh_trimesh,
                                vertex_colors=grey_color,
                            )

                        logger.info(
                            f"sequence {sequence_id} in {category}, {len(_mesh_trimesh.vertices)} vertices",
                        )
                        # _mesh_trimesh.show()

                        if len(_mesh_trimesh.vertices) > 0:
                            pass
                            # _ = _mesh_trimesh.export(file_obj=frame_fpath_mesh)
                        else:
                            logger.info(
                                f"skip sequence {sequence_id} in {category} because could not convert to single mesh.",
                            )

                        from od3d.cv.geometry.transform import (
                            get_ico_cam_tform4x4_obj_for_viewpoints_count,
                            get_ico_traj_cam_tform4x4_obj_for_viewpoints_count,
                            get_cam_tform4x4_obj_for_viewpoints_count,
                            tform4x4_broadcast,
                        )
                        from od3d.cv.visual.show import (
                            get_default_camera_intrinsics_from_img_size,
                        )
                        import torch
                        from od3d.cv.geometry.objects3d.meshes import Meshes
                        from od3d.cv.visual.show import OBJ_TFORM_OBJ_SHAPENET
                        from od3d.cv.io import (
                            write_image,
                            write_mask_image,
                            write_depth_image,
                            write_pts3d_with_colors_and_normals,
                        )
                        from od3d.cv.visual.show import render_trimesh_to_tensor
                        from od3d.cv.geometry.transform import (
                            depth2pts3d_grid,
                            inv_tform4x4,
                        )
                        from od3d.cv.geometry.downsample import random_sampling, fps

                        # random_sampling(pts3d_cls, pts3d_max_count, return_mask=False)
                        # pts3d_ids, pts3d = fps(pts3d=pts3d, K=1024, fill=True)
                        W = 512
                        H = 512
                        device = "cuda"

                        imgs_sizes = torch.LongTensor([W, H])

                        if viewpoints_type.startswith(
                            "icotraj",
                        ) or viewpoints_type.startswith("icorealtraj"):
                            if viewpoints_type.startswith("icotraj"):
                                prefix = "icotraj"
                                real = False
                            elif viewpoints_type.startswith("icorealtraj"):
                                prefix = "icorealtraj"
                                real = True

                            traj_length = (
                                float(viewpoints_type.removeprefix(prefix))
                                / 100
                                * math.pi
                                * 2
                            )
                            cams_tform4x4_obj = (
                                get_ico_traj_cam_tform4x4_obj_for_viewpoints_count(
                                    viewpoints_count=viewpoints_count,
                                    radius=2.5,
                                    theta_count=theta_count,
                                    geodesic_distance=traj_length,
                                    real=real,
                                ).to(device)
                            )
                            # from od3d.cv.visual.show import show_scene
                            # show_scene(cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=cam_intr4x4[None,].repeat(64, 1, 1))
                        else:
                            cams_tform4x4_obj = (
                                get_ico_cam_tform4x4_obj_for_viewpoints_count(
                                    viewpoints_count=viewpoints_count,
                                    radius=2.5,
                                    theta_count=theta_count,
                                    viewpoints_uniform=viewpoints_sample_uni,
                                    theta_uniform=theta_sample_uni,
                                ).to(device)
                            )

                        # front, top, right, bottom
                        # cams_tform4x4_obj = get_cam_tform4x4_obj_for_viewpoints_count(viewpoints_count=3, dist=2.).to(device)

                        cams_tform4x4_obj = tform4x4_broadcast(
                            cams_tform4x4_obj,
                            OBJ_TFORM_OBJ_SHAPENET.to(device),
                        )
                        cam_intr4x4 = get_default_camera_intrinsics_from_img_size(
                            W=W,
                            H=H,
                        ).to(device)
                        try:
                            mesh = Meshes.read_from_ply_file(
                                fpath=frame_fpath_mesh,
                                device=device,
                                load_texts=True,
                            )
                        except Exception:
                            logger.warning(f"could not load texture {frame_fpath_mesh}")
                            continue

                        logger.info(
                            f"sequence {sequence_id} in {category}, {len(mesh.verts)} vertices",
                        )
                        logger.info(f"{frame_fpath_mesh}")

                        mods = mesh.render(
                            cams_tform4x4_obj=cams_tform4x4_obj,
                            cams_intr4x4=cam_intr4x4,
                            imgs_sizes=imgs_sizes,
                            modalities=["mask"],
                            broadcast_batch_and_cams=True,
                            rgb_bg=[0.0, 0.0, 0.0],
                            rgb_diffusion_alpha=0.3,
                        )
                        # rgbs = mods['rgb'][0]
                        masks = mods["mask"][0]

                        # from od3d.cv.visual.show import show_imgs
                        # show_imgs(rgbs)
                        pts3d = []
                        for i, cam_tform4x4_obj in enumerate(cams_tform4x4_obj):
                            frame_name = f"{int(i):03}"
                            frame_fpath_rgb = path_raw.joinpath(
                                f"rgb/{subset}/{category}/{sequence_id}/{frame_name}.png",
                            )
                            frame_fpath_mask = path_raw.joinpath(
                                f"mask/{subset}/{category}/{sequence_id}/{frame_name}.png",
                            )
                            frame_fpath_depth = path_raw.joinpath(
                                f"depth/{subset}/{category}/{sequence_id}/{frame_name}.png",
                            )
                            frame_fpath_depth_mask = path_raw.joinpath(
                                f"depth_mask/{subset}/{category}/{sequence_id}/{frame_name}.png",
                            )

                            rgb, depth = render_trimesh_to_tensor(
                                mesh_trimesh=_mesh_trimesh,
                                cam_tform4x4_obj=cam_tform4x4_obj,
                                cam_intr4x4=cam_intr4x4,
                                H=H,
                                W=W,
                                rgb_bg=[0.8, 0.8, 0.8],
                            )
                            depth_mask = depth > 0.0

                            _pts3d = depth2pts3d_grid(
                                depth=depth.to(device),
                                cam_intr4x4=cam_intr4x4,
                            ).permute(1, 2, 0)[depth_mask[0]]
                            _pts3d = random_sampling(
                                pts3d_cls=_pts3d,
                                pts3d_max_count=1024,
                            )
                            _pts3d = transf3d_broadcast(
                                pts3d=_pts3d,
                                transf4x4=inv_tform4x4(cam_tform4x4_obj),
                            )
                            pts3d.append(_pts3d)
                            write_image(img=rgb, path=frame_fpath_rgb)
                            write_image(img=masks[i], path=frame_fpath_mask)
                            write_depth_image(img=depth, path=frame_fpath_depth)
                            write_mask_image(
                                img=depth_mask,
                                path=frame_fpath_depth_mask,
                            )

                            frame_meta = ShapeNet_FrameMeta(
                                name=frame_name,
                                sequence_name=sequence_id,
                                rfpath_rgb=frame_fpath_rgb.relative_to(path_raw),
                                rfpath_mask=frame_fpath_mask.relative_to(path_raw),
                                rfpath_depth=frame_fpath_depth.relative_to(path_raw),
                                rfpath_depth_mask=frame_fpath_depth_mask.relative_to(
                                    path_raw,
                                ),
                                # rfpath_mesh=frame_fpath_mesh.relative_to(path_raw),
                                category=category,
                                l_size=imgs_sizes.tolist(),
                                l_cam_tform4x4_obj=cam_tform4x4_obj.tolist(),
                                l_cam_intr4x4=cam_intr4x4.tolist(),
                                subset=subset,
                            )

                            if frame_meta is not None:
                                frame_meta.save(path_meta=path_meta)

                        pts3d = torch.cat(pts3d, dim=0)
                        pts3d_ids, pts3d = fps(pts3d=pts3d, K=1024, fill=True)
                        write_pts3d_with_colors_and_normals(
                            pts3d,
                            pts3d_colors=None,
                            pts3d_normals=None,
                            fpath=frame_fpath_pcl,
                        )

                        if sequence_meta is not None:
                            sequence_meta.save(path_meta=path_meta)
                        sequences_count = sequences_count + 1

        if SHAPENET_SUBSETS.R2N2 in subsets:
            # R2N2
            path_r2n2 = path_raw.joinpath("ShapeNetRendering")
            categories_r2n2_id = list(
                MAP_CATEGORIES_SHAPENET_R2N2_ID_TO_CATEGORY.keys(),
            )

            for category_id in categories_r2n2_id:
                category = MAP_CATEGORIES_SHAPENET_ID_TO_CATEGORY[category_id]
                logger.info(f"extract meta category {category}")
                path_r2n2_category = path_r2n2.joinpath(category_id)
                sequences_ids = [
                    path.name for path in list(path_r2n2_category.iterdir())
                ]
                for sequence_id in sequences_ids:
                    logger.info(f"extract meta sequence {sequence_id}")
                    path_r2n2_sequence = path_r2n2_category.joinpath(
                        f"{sequence_id}/rendering",
                    )
                    path_renderings_txt = path_r2n2_sequence.joinpath("renderings.txt")
                    path_renderings_meta_txt = path_r2n2_sequence.joinpath(
                        "rendering_metadata.txt",
                    )
                    from od3d.io import read_str_from_file

                    frames_fnames = read_str_from_file(fpath=path_renderings_txt).split(
                        "\n",
                    )
                    frames_names = [Path(fname).stem for fname in frames_fnames]
                    import torch
                    from od3d.cv.geometry.objects3d.meshes import Meshes

                    frames_metas_strs = read_str_from_file(
                        fpath=path_renderings_meta_txt,
                    ).split("\n")
                    frames_metas_cams = torch.Tensor(
                        [
                            [float(number) for number in frame_meta_str.split(" ")]
                            for frame_meta_str in frames_metas_strs
                        ],
                    )
                    # N x 5 {yaw, pitch, 0, radius, fov}
                    # N x 5 {azimuth, elevation, 0, depth_ratio, 25}
                    MAX_CAMERA_DIST = 1.75
                    device = "cuda"

                    frame_fpath_mesh = path_raw.joinpath(
                        f"ShapeNetCore.v1/{category_id}/{sequence_id}/model.obj",
                    )

                    # mesh = Meshes.read_from_ply_file(fpath=frame_fpath_mesh, device=device, load_texts=False)

                    sequence_meta = ShapeNet_SequenceMeta(
                        subset=SHAPENET_SUBSETS.R2N2.value,
                        category=category,
                        name=sequence_id,
                        rfpath_mesh=frame_fpath_mesh.relative_to(path_raw),
                    )
                    if sequence_meta is not None:
                        sequence_meta.save(path_meta=path_meta)

                    for f, frame_name in enumerate(frames_names):
                        frame_fpath_rgb = path_r2n2_sequence.joinpath(frames_fnames[f])

                        frame_meta_cam = frames_metas_cams[f]

                        import numpy as np
                        from od3d.cv.io import read_image
                        from od3d.cv.visual.show import (
                            get_default_camera_intrinsics_from_img_size,
                        )
                        from od3d.cv.visual.show import show_img, show_imgs
                        import math

                        # v1
                        azimuth = torch.Tensor(
                            [-frame_meta_cam[0] / 180 * math.pi - math.pi / 2],
                        )
                        # fov = frame_meta_cam[4]
                        fov = 2 * math.atan(35 / (2 * 38)) * 180 / math.pi
                        # (field of view corresponding to sensor width 32mm, focal length 35mm - the blender default.
                        # fov = 2 * math.atan(36 / (2 * 50)) * 180 / math.pi

                        # v2
                        # azimuth = torch.Tensor([-frame_meta_cam[0] / 180 * math.pi])

                        elevation = torch.Tensor([frame_meta_cam[1] / 180 * math.pi])
                        depth_ratio = (
                            torch.Tensor([frame_meta_cam[3]]) * MAX_CAMERA_DIST
                        )
                        theta = torch.Tensor([0.0])

                        # fov = frame_meta_cam[4]
                        # fov = 2 * math.atan(35 / (2*32)) * 180 / math.pi
                        # fov = 2 * math.atan(36 / (2*50)) * 180 / math.pi
                        # (field of view corresponding to sensor width 32mm, focal length 35mm - the blender default.

                        rgb = read_image(frame_fpath_rgb)

                        from od3d.cv.visual.show import (
                            DEFAULT_CAM_TFORM_OBJ,
                            OBJ_TFORM_OBJ_SHAPENET,
                        )
                        from od3d.cv.geometry.transform import (
                            tform4x4,
                            transf4x4_from_spherical,
                        )

                        cams_tform4x4_obj = transf4x4_from_spherical(
                            azim=azimuth,
                            elev=elevation,
                            theta=theta,
                            dist=depth_ratio,
                        ).to(device)

                        cams_tform4x4_obj = tform4x4(
                            cams_tform4x4_obj,
                            OBJ_TFORM_OBJ_SHAPENET.clone().to(device)[None,],
                        )
                        # cams_tform4x4_obj[:, :3, :3] = rotation_matrix[None,]
                        # cams_tform4x4_obj = tform4x4(DEFAULT_CAM_TFORM_OBJ[None,].to(device=device), cams_tform4x4_obj)
                        # cams_tform4x4_obj[:, 2, 3] = 1. * radius

                        imgs_sizes = torch.LongTensor([137, 137]).to(device)
                        # imgs_sizes = torch.LongTensor([256, 256]).to(device)

                        cams_intr4x4 = torch.eye(4)[None,].to(device)
                        cams_intr4x4 = get_default_camera_intrinsics_from_img_size(
                            fov_x=fov,
                            H=imgs_sizes[0],
                            W=imgs_sizes[1],
                            device=device,
                        )

                        cams_intr4x4 = cams_intr4x4[None,]

                        # rgb_rendered = mesh.render(cams_tform4x4_obj=cams_tform4x4_obj, cams_intr4x4=cams_intr4x4,
                        #                           imgs_sizes=imgs_sizes, modalities='rgb')

                        # show_imgs([rgb[:3].to(device), rgb_rendered[0].to(device) * 255.])

                        # from od3d.cv.visual.show import show_scene
                        # show_scene(meshes=mesh, show_coordinate_frame=True)

                        frame_meta = ShapeNet_FrameMeta(
                            name=frame_name,
                            sequence_name=sequence_id,
                            rfpath_rgb=frame_fpath_rgb.relative_to(path_raw),
                            rfpath_mask=frame_fpath_rgb.relative_to(path_raw),
                            # rfpath_mesh=frame_fpath_mesh.relative_to(path_raw),
                            category=category,
                            l_size=imgs_sizes.tolist(),
                            l_cam_tform4x4_obj=cams_tform4x4_obj[0].tolist(),
                            l_cam_intr4x4=cams_intr4x4[0].tolist(),
                            subset=SHAPENET_SUBSETS.R2N2.value,
                        )

                        if frame_meta is not None:
                            frame_meta.save(path_meta=path_meta)

    @staticmethod
    def setup(config: DictConfig):
        # logger.info(OmegaConf.to_yaml(config))
        path_raw = Path(config.path_raw)
        logger.info(f"setup ShapeNet at {path_raw}")
        if path_raw.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous ShapeNet")
            shutil.rmtree(path_raw)

        path_raw.mkdir(parents=True, exist_ok=True)

        hf_name = "ShapeNet/ShapeNetCore"

        # download R2N2
        if not path_raw.joinpath("ShapeNetRendering").exists():
            od3d.io.run_cmd(
                f"cd {path_raw} && wget ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz",
                logger=logger,
                live=True,
            )
            time.sleep(5)
            od3d.io.run_cmd(
                f"cd {path_raw} && tar -xvzf ShapeNetRendering.tgz",
                logger=logger,
                live=True,
            )
            time.sleep(5)
            # these splits should equal the splits used in r2n2
            # # wget https://dl.fbaipublicfiles.com/meshrcnn/shapenet/pix2mesh_splits_val05.json
            od3d.io.run_cmd(
                f'cd {path_raw.joinpath("ShapeNetRendering")} && wget https://dl.fbaipublicfiles.com/meshrcnn/shapenet/pix2mesh_splits_val05.json',
                logger=logger,
                live=True,
            )

        from od3d.io import read_json, write_dict_as_yaml
        from od3d.datasets.shapenet.enum import (
            MAP_CATEGORIES_SHAPENET_V2_ID_TO_CATEGORY,
        )

        splits = read_json(
            path_raw.joinpath("ShapeNetRendering/pix2mesh_splits_val05.json"),
        )
        for split in splits.keys():
            logger.info(f"creating split yaml file {split}")
            category_id_split_dict = splits[split]

            category_split_dict = {}
            for category_id in splits[split].keys():
                category = MAP_CATEGORIES_SHAPENET_V2_ID_TO_CATEGORY[category_id]
                category_split_dict[category] = category_id_split_dict[category_id]
                for sequence_id in category_split_dict[category].keys():
                    category_split_dict[category][sequence_id] = [
                        f"{int(n):02}"
                        for n in category_split_dict[category][sequence_id]
                    ]
            fpath_split = path_raw.joinpath(f"ShapeNetRendering/{split}.yaml")
            split_dict = {"dict_nested_frames": {"r2n2": category_split_dict}}
            write_dict_as_yaml(_dict=split_dict, fpath=fpath_split)

        # download v1
        if not path_raw.joinpath("ShapeNetCore.v1").exists():
            # https://huggingface.co/datasets/ShapeNet/ShapeNetCore-archive/resolve/main/ShapeNetCore.v1.zip?download=true
            logger.info("ShapeNet downloading v1...")
            hf_token = "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"
            cmd = f'cd {path_raw} && wget --header="Authorization: Bearer {hf_token}" https://huggingface.co/datasets/ShapeNet/ShapeNetCore-archive/resolve/main/ShapeNetCore.v1.zip'
            od3d.io.run_cmd(
                cmd,
                logger=logger,
                live=True,
            )
            time.sleep(5)
            od3d.io.run_cmd(
                f"cd {path_raw} && unzip ShapeNetCore.v1.zip",
                logger=logger,
                live=True,
            )
            time.sleep(5)
            od3d.io.run_cmd(
                f"cd {path_raw} && rm ShapeNetCore.v1.zip",
                logger=logger,
                live=True,
            )

        # note: v2 has normaliztion: v_norm = (v - centroid) / norm(diag(boundingbox))
        categories_ids = MAP_CATEGORIES_SHAPENET_V2_ID_TO_CATEGORY.keys()
        for category_id in categories_ids:
            if path_raw.joinpath(category_id).exists():
                logger.info(
                    f"already there, category: {MAP_CATEGORIES_SHAPENET_V2_ID_TO_CATEGORY[category_id]}, "
                    f"{category_id}",
                )
            else:
                # bicycle
                # table, 04379243
                logger.info(
                    f"get category: {MAP_CATEGORIES_SHAPENET_V2_ID_TO_CATEGORY[category_id]}, {category_id}",
                )
                # $HF_TOKEN
                hf_token = "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"
                cmd = f'cd {path_raw} && wget --header="Authorization: Bearer {hf_token}" https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/{category_id}.zip'
                od3d.io.run_cmd(
                    cmd,
                    logger=logger,
                    live=True,
                )
                time.sleep(5)
                od3d.io.run_cmd(
                    f"cd {path_raw} && unzip {category_id}.zip",
                    logger=logger,
                    live=True,
                )

            category = MAP_CATEGORIES_SHAPENET_V2_ID_TO_CATEGORY[category_id]
            sequences_ids = [
                path.stem for path in list(path_raw.joinpath(category_id).iterdir())
            ]
            for sequence_id in sequences_ids:
                path_sequence_id = path_raw.joinpath(category_id, sequence_id, "models")
                # $HF_TOKEN
                # https://huggingface.co/datasets/ShapeNet/shapenetcore-glb/resolve/main/airplane/10155655850468db78d106ce0a280f87.glb?download=true
                hf_token = "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"
                logger.info(f"get object: {path_sequence_id}")
                cmd = f'cd {path_sequence_id} && wget --header="Authorization: Bearer {hf_token}" https://huggingface.co/datasets/ShapeNet/shapenetcore-glb/resolve/main/{category}/{sequence_id}.glb'
                od3d.io.run_cmd(
                    cmd,
                    logger=logger,
                    live=True,
                )
                cmd = f"cd {path_sequence_id} && mv {sequence_id}.glb model.glb"
                od3d.io.run_cmd(
                    cmd,
                    logger=logger,
                    live=True,
                )

        # from huggingface_hub import login
        # login("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", add_to_git_credential=True)
        # from datasets import load_dataset, get_dataset_split_names # , list_datasets
        # dataset = load_dataset("ShapeNet/shapenetcore-glb", data_files= ,cache_dir=str(path_raw)) # .parent
        # https://huggingface.co/datasets/ShapeNet/shapenetcore-glb/resolve/main/airplane/10155655850468db78d106ce0a280f87.glb?download=true
        # categories_ids = MAP_CATEGORIES_SHAPENET_V2_ID_TO_CATEGORY.keys()
        # for category_id in categories_ids:
        #     category = MAP_CATEGORIES_SHAPENET_V2_ID_TO_CATEGORY[category_id]
        #     sequences_ids = [path.stem for path in list(path_raw.joinpath(category_id).iterdir())]
        #     for sequence_id in sequences_ids:
        #         # $HF_TOKEN
        #         # https://huggingface.co/datasets/ShapeNet/shapenetcore-glb/resolve/main/airplane/10155655850468db78d106ce0a280f87.glb?download=true
        #         hf_token = "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"
        #         cmd = f'cd {path_raw} && wget --recursive --no-parent --header="Authorization: Bearer {hf_token}" https://huggingface.co/datasets/ShapeNet/shapenetcore-glb/resolve/main/{category}/{sequence_id}.glb'
        #         od3d.io.run_cmd(
        #             cmd,
        #             logger=logger,
        #             live=True,
        #         )

        # wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/02691156.zip

        # export HF_TOKEN=<your_hf_token>

        # from datasets import load_dataset, get_dataset_split_names # , list_datasets
        # dataset = load_dataset(hf_name, cache_dir=str(path_raw.parent))

        # dataset = load_dataset(hf_name, Token=True, cache_dir=str(path_raw.parent))
        # logger.info(get_dataset_split_names(hf_name))

        # MAP_CATEGORIES_SHAPENET_ID_TO_CATEGORY

        # "wget ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz"
        # "tar -xvzf ShapeNetRendering.tgz"

        # git config --global credential.helper store
        # huggingface-cli login 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
        # "https://huggingface.co/datasets/ShapeNet/ShapeNetCore-archive/blob/main/ShapeNetCore.v2.zip"

        # dataset = load_dataset("ShapeNet/ShapeNetCore")
        # from datasets import load_dataset
        # --filter=blob:none
        # dataset = load_dataset('ShapeNet/ShapeNetCore', cache_dir="/scratch/sommerl/repos/NeMo/ShapeNetCore")
        # git clone --filter=blob:none https://huggingface.co/datasets/ShapeNet/ShapeNetCore
        # https://huggingface.co/datasets/ShapeNet/ShapeNetCore/tree/main
        # from datasets import get_dataset_split_names
        # from datasets import load_dataset_builder
        # from datasets import load_dataset
        # from datasets import get_dataset_config_names
        # "ShapeNet/ShapeNetCore"  "rotten_tomatoes"
        # ds_builder = load_dataset_builder("ShapeNet/ShapeNetCore")
        # ds_builder.info.description
        # ds_builder.info.features
        # get_dataset_config_names("ShapeNet/ShapeNetCore")
        # # load_dataset('LOADING_SCRIPT', cache_dir="PATH/TO/MY/CACHE/DIR")

        # path_raw.mkdir(parents=True, exist_ok=True)
        # od3d.io.run_cmd("pip install opendatalab", logger=logger, live=True)
        # od3d.io.run_cmd(
        #     'please signup and login first with "odl login"',
        #     logger=logger,
        #     live=True,
        # )
        # #  -u {config.credentials.opendatalab.username} -p {config.credentials.opendatalab.password}
        # od3d.io.run_cmd(
        #     "odl info   OpenXD-OmniObject3D-New",
        #     logger=logger,
        #     live=True,
        # )  # View dataset metadata
        # od3d.io.run_cmd(
        #     "odl ls     OpenXD-OmniObject3D-New",
        #     logger=logger,
        #     live=True,
        # )  # View a list of dataset files
        # od3d.io.run_cmd(
        #     f"cd {path_raw} && odl get    OpenXD-OmniObject3D-New",
        #     logger=logger,
        #     live=True,
        # )

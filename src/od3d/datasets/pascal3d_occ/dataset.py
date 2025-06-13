import logging

logger = logging.getLogger(__name__)
from od3d.datasets.dataset import OD3D_Dataset
from omegaconf import DictConfig
from pathlib import Path
from od3d.datasets.object import OD3D_FRAME_DEPTH_TYPES
from od3d.datasets.frame import OD3D_FRAME_KPTS2D_ANNOT_TYPES
from od3d.datasets.pascal3d.enum import (
    PASCAL3D_CATEGORIES,
    MAP_CATEGORIES_OD3D_TO_PASCAL3D,
)
from od3d.datasets.frame import OD3D_FRAME_MODALITIES
from typing import Dict, List
import shutil
import od3d.io
from od3d.datasets.pascal3d.frame import Pascal3DFrameMeta, Pascal3DFrame
from od3d.datasets.object import OD3D_MESH_TYPES, OD3D_SCALE_TYPES
from od3d.datasets.pascal3d.frame import Pascal3DFrameMeta
from od3d.datasets.pascal3d_occ.frame import Pascal3D_OccFrameMeta, Pascal3D_OccFrame
from od3d.cv.geometry.objects3d.meshes import Meshes
from tqdm import tqdm
from od3d.datasets.pascal3d.enum import (
    PASCAL3D_SCALE_NORMALIZE_TO_REAL,
    PASCAL3D_CATEGORIES,
)
from od3d.data.ext_enum import ExtEnum


class PASCAL3D_OCC_SUBSETS(str, ExtEnum):
    LVL1 = "lvl1"
    LVL2 = "lvl2"
    LVL3 = "lvl3"


class Pascal3D_Occ(OD3D_Dataset):
    all_categories = list(PASCAL3D_CATEGORIES)
    map_od3d_categories = MAP_CATEGORIES_OD3D_TO_PASCAL3D
    frame_type = Pascal3D_OccFrame
    mesh_type = OD3D_MESH_TYPES.META

    def __init__(
        self,
        name: str,
        modalities: List[OD3D_FRAME_MODALITIES],
        path_raw: Path,
        path_preprocess: Path,
        path_pascal3d_raw: Path,
        categories: List[PASCAL3D_CATEGORIES] = None,
        dict_nested_frames: Dict[str, Dict[str, List[str]]] = None,
        dict_nested_frames_ban: Dict[str, Dict[str, List[str]]] = None,
        transform=None,
        subset_fraction=1.0,
        index_shift=0,
        mesh_type: OD3D_MESH_TYPES = OD3D_MESH_TYPES.META,
        scale_type: OD3D_SCALE_TYPES = OD3D_SCALE_TYPES.NORM,
    ):
        if categories is not None:
            if self.map_od3d_categories is not None:
                self.categories = [
                    self.map_od3d_categories.get(category, category)
                    if category not in self.all_categories
                    else category
                    for category in categories
                ]
                print(f"categories: {self.categories}")
            else:
                self.categories = categories
        else:
            self.categories = self.all_categories
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
            scale_type=scale_type,
        )
        self.mesh_type = mesh_type
        self.path_pascal3d_raw = Path(path_pascal3d_raw)

    def filter_dict_nested_frames(self, dict_nested_frames):
        dict_nested_frames = super().filter_dict_nested_frames(dict_nested_frames)
        logger.info("filtering frames categorical...")
        dict_nested_frames_filtered = {}
        for subset, dict_category_frames in dict_nested_frames.items():
            dict_nested_frames_filtered[subset] = {}
            for category, list_frames in dict_category_frames.items():
                if category in self.categories:
                    dict_nested_frames_filtered[subset][category] = list_frames

        dict_nested_frames = dict_nested_frames_filtered
        return dict_nested_frames

    def get_subset_with_dict_nested_frames(self, dict_nested_frames):
        return Pascal3D_Occ(
            name=self.name,
            modalities=self.modalities,
            path_raw=self.path_raw,
            path_preprocess=self.path_preprocess,
            categories=self.categories,
            dict_nested_frames=dict_nested_frames,
            transform=self.transform,
            index_shift=self.index_shift,
            path_pascal3d_raw=self.path_pascal3d_raw,
        )

    # def get_item(self, item):
    #     frame_meta = Pascal3D_OccFrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta,
    #                                                                    name_unique=self.list_frames_unique[item])
    #     return Pascal3D_OccFrame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
    #                              path_meshes=self.path_meshes, meta=frame_meta, modalities=self.modalities,
    #                              categories=self.categories)
    @staticmethod
    def setup(config: DictConfig):
        path_raw = Path(config.path_raw)

        if path_raw.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous Pascal3D_Occ")
            shutil.rmtree(path_raw)

        if path_raw.exists():
            logger.info(f"Found Pascal3D_Occ dataset at {path_raw}")
        else:
            logger.info(f"Downloading Pascal3D_Occ dataset at {path_raw}")
            path_raw.mkdir(parents=True, exist_ok=True)
            fpath = path_raw.joinpath("pascal3d_occ.sh")

            od3d.io.download(url=config.url_pascal3d_occ_script, fpath=fpath)
            od3d.io.run_cmd(cmd=f"chmod +x {fpath}", logger=logger, live=True)
            od3d.io.run_cmd(cmd=f"cd {path_raw} && {fpath}", logger=logger, live=True)

    def preprocess(self, config_preprocess: DictConfig):
        logger.info("preprocess")
        for key in config_preprocess.keys():
            if key == "cuboid" and config_preprocess.cuboid.get("enabled", False):
                override = config_preprocess.cuboid.get("override", False)
                remove_previous = config_preprocess.cuboid.get("remove_previous", False)
                self.preprocess_cuboid(
                    override=override,
                    remove_previous=remove_previous,
                )
            elif key == "mask" and config_preprocess.mask.get("enabled", False):
                override = config_preprocess.mask.get("override", False)
                remove_previous = config_preprocess.mask.get("remove_previous", False)
                self.preprocess_mask(override=override, remove_previous=remove_previous)
            elif key == "depth" and config_preprocess.depth.get("enabled", False):
                override = config_preprocess.depth.get("override", False)
                remove_previous = config_preprocess.depth.get("remove_previous", False)
                self.preprocess_depth(
                    override=override,
                    remove_previous=remove_previous,
                )

    def preprocess_cuboid(self, override=False, remove_previous=False, quantile=0.95):
        logger.info("preprocess cuboid...")

        scale_pascal3d_to_od3d = {}
        for category in self.categories:
            if category not in self.all_categories:
                continue
            mesh_types = [
                OD3D_MESH_TYPES.SPHERE250,
                OD3D_MESH_TYPES.SPHERE500,
                OD3D_MESH_TYPES.SPHERE1000,
                OD3D_MESH_TYPES.CUBOID250,
                OD3D_MESH_TYPES.CUBOID500,
                OD3D_MESH_TYPES.CUBOID1000,
            ]
            for mesh_type in mesh_types:
                fpath_mesh_out = self.path_preprocess.joinpath(
                    Pascal3D_OccFrame.get_rfpath_pp_categorical_mesh(
                        mesh_type=mesh_type,
                        category=category,
                    ),
                )

                if fpath_mesh_out.exists() and not override:
                    logger.warning(f"mesh already exists {fpath_mesh_out}")
                    continue
                else:
                    logger.info(
                        f"preprocessing mesh for {category} with type {mesh_type}",
                    )

                import re

                match = re.match(r"([a-z]+)([0-9]+)", mesh_type, re.I)
                if match and len(match.groups()) == 2:
                    mesh_type, mesh_vertices_count = match.groups()
                    mesh_vertices_count = int(mesh_vertices_count)
                else:
                    msg = f"could not retrieve mesh type and vertices count from mesh name {mesh_type}"
                    raise Exception(msg)

                fpaths_meshes_category = [
                    fpath
                    for fpath in self.path_pascal3d_raw.joinpath(
                        Pascal3D_OccFrame.get_rpath_raw_categorical_meshes(
                            category=category,
                        ),
                    ).iterdir()
                ]
                meshes = Meshes.read_from_ply_files(fpaths_meshes_category)
                meshes.verts.data = meshes.verts
                pts3d = meshes.verts

                from od3d.cv.geometry.fit.cuboid import (
                    fit_cuboid_to_pts3d,
                    fit_sphere_to_pts3d,
                )
                from od3d.datasets.enum import OD3D_CATEGORIES_SIZES_IN_M

                if "cuboid" in mesh_type:
                    cuboids, tform_obj = fit_cuboid_to_pts3d(
                        pts3d=pts3d,
                        optimize_rot=False,
                        optimize_transl=False,
                        vertices_max_count=mesh_vertices_count,
                        optimize_steps=0,
                        q=0.95,
                        # size=OD3D_CATEGORIES_SIZES_IN_M[
                        #    MAP_CATEGORIES_PASCAL3D_TO_OD3D[category]
                        # ],
                    )

                    # scale_pascal3d_to_od3d[category] = (
                    #    tform_obj[:3, :3].norm(dim=-1).mean()
                    # )
                    # show:
                    # meshes.verts *= scale_pascal3d_to_od3d[category]
                    # Meshes.load_from_meshes([meshes.get_mesh_with_id(i) for i in range(meshes.meshes_count)] + [cuboids.get_mesh_with_id(0)]).show(meshes_add_translation=False)

                    obj_mesh = cuboids.get_mesh_with_id(0)

                elif "sphere" in mesh_type:
                    import torch

                    spheres, tform_obj = fit_sphere_to_pts3d(
                        pts3d=pts3d,
                        optimize_transl=False,
                        vertices_max_count=mesh_vertices_count,
                        q=0.95,
                        # size=OD3D_CATEGORIES_SIZES_IN_M[
                        #    MAP_CATEGORIES_PASCAL3D_TO_OD3D[category]
                        # ],
                    )

                    # scale_pascal3d_to_od3d[category] = (
                    #    tform_obj[:3, :3].norm(dim=-1).mean()
                    # )
                    # show:
                    # meshes.verts *= scale_pascal3d_to_od3d[category]
                    # Meshes.load_from_meshes([meshes.get_mesh_with_id(i) for i in range(meshes.meshes_count)] + [cuboids.get_mesh_with_id(0)]).show(meshes_add_translation=False)

                    obj_mesh = spheres.get_mesh_with_id(0)

                else:
                    raise ValueError(f"mesh type {mesh_type} not recognized")

                obj_mesh.write_to_file(fpath=fpath_mesh_out)

        log_str = "\n"
        for key, val in scale_pascal3d_to_od3d.items():
            log_str += str(key) + f": {val}, \n"
        logger.info(log_str)

    #### PREPROCESS META
    @staticmethod
    def extract_meta(config: DictConfig):
        subsets = config.get("subsets", None)
        if subsets is None:
            subsets = PASCAL3D_OCC_SUBSETS.list()
        categories = config.get("categories", None)
        if categories is None:
            categories = PASCAL3D_CATEGORIES.list()

        path_raw = OD3D_Dataset.get_path_raw(config=config)
        path_meta = OD3D_Dataset.get_path_meta(config=config)
        path_pascal3d_raw = Path(config.path_pascal3d_raw)
        rpath_meshes = Path("CAD")

        if config.extract_meta.remove_previous:
            if path_meta.exists():
                shutil.rmtree(path_meta)

        frames_names = []
        frames_categories = []
        frames_subsets = []
        for subset in subsets:
            for category in categories:
                # fpath_frame_names_partial = path_raw.joinpath("Image_sets", f"{category}_imagenet_{subset}.txt")
                subset_lvl = int(subset[-1:])
                fpath_frame_names_partial = path_raw.joinpath(
                    "lists",
                    f"{category}FGL{subset_lvl}_BGL{subset_lvl}.txt",
                )

                with fpath_frame_names_partial.open() as f:
                    frame_names_partial = f.read().splitlines()
                frames_names += [
                    Path(file_name).stem for file_name in frame_names_partial
                ]
                frames_categories += [category] * len(frame_names_partial)
                frames_subsets += [subset] * len(frame_names_partial)

        # frames_subsets, frames_categories, frames_names = Pascal3DFrameMeta.get_frames_names_from_subsets_and_cateogories_from_raw(path_pascal3d_raw=path_raw, subsets=subsets, categories=categories)

        if config.get("frames", None) is not None:
            frames_names = list(filter(lambda f: f in config.frames, frames_names))

        for i in tqdm(range(len(frames_names))):
            fpath = path_meta.joinpath(
                Pascal3D_OccFrameMeta.get_rfpath_from_name_unique(
                    name_unique=Pascal3DFrameMeta.get_name_unique_from_category_subset_name(
                        subset=frames_subsets[i],
                        category=frames_categories[i],
                        name=frames_names[i],
                    ),
                ),
            )
            if not fpath.exists() or config.extract_meta.override:
                pascal3d_frame_meta = Pascal3DFrameMeta.load_from_raw(
                    frame_name=frames_names[i],
                    subset="val",
                    category=frames_categories[i],
                    path_raw=path_pascal3d_raw,
                    rpath_meshes=rpath_meshes,
                )
                if pascal3d_frame_meta is None:
                    continue
                subset = frames_subsets[i]
                category = frames_categories[i]
                subset_lvl = int(subset[-1:])
                rfpath_rgb = Path(
                    "images",
                    f"{category}FGL{subset_lvl}_BGL{subset_lvl}",
                    f"{frames_names[i]}.JPEG",
                )
                rfpath_npz = Path(
                    "annotations",
                    f"{category}FGL{subset_lvl}_BGL{subset_lvl}",
                    f"{frames_names[i]}.npz",
                )
                fpath_npz = path_raw.joinpath(rfpath_npz)
                import numpy as np

                annots = np.load(fpath_npz)
                # 'source',
                # 'mask',
                # 'box',                # y0, y1, x0, x1, H, W
                # 'occluder_mask',
                # 'occluder_box',
                # 'category',
                # 'occluder_level'

                l_size = list(annots["mask"].shape)
                assert pascal3d_frame_meta.l_size == l_size

                l_bbox = [
                    int(annots["box"][2]),
                    int(annots["box"][0]),
                    int(annots["box"][3]),
                    int(annots["box"][1]),
                ]
                # assert pascal3d_frame_meta.l_bbox == l_bbox

                rfpath_mask = rfpath_npz

                frame_meta = Pascal3D_OccFrameMeta(
                    name=pascal3d_frame_meta.name,
                    l_size=l_size,
                    rfpath_rgb=rfpath_rgb,
                    l_cam_intr4x4=pascal3d_frame_meta.l_cam_intr4x4,
                    l_cam_tform4x4_obj=pascal3d_frame_meta.l_cam_tform4x4_obj,
                    category=pascal3d_frame_meta.category,
                    rfpath_mesh=pascal3d_frame_meta.rfpath_mesh,
                    subset=frames_subsets[i],
                    l_kpts3d=pascal3d_frame_meta.l_kpts3d,
                    l_kpts2d_annot=pascal3d_frame_meta.l_kpts2d_annot,
                    l_kpts2d_annot_vsbl=pascal3d_frame_meta.l_kpts2d_annot_vsbl,
                    kpts_names=pascal3d_frame_meta.kpts_names,
                    rfpath_mask=rfpath_mask,
                    l_bbox=l_bbox,
                )

                if frame_meta is not None:
                    frame_meta.save(path_meta=path_meta)

    ##### DATASET PROPERTIES
    def get_frame_by_name_unique(self, name_unique):
        from od3d.datasets.object import (
            OD3D_CAM_TFORM_OBJ_TYPES,
            OD3D_FRAME_MASK_TYPES,
            OD3D_MESH_TYPES,
            OD3D_MESH_FEATS_TYPES,
            OD3D_MESH_FEATS_DIST_REDUCE_TYPES,
            OD3D_TFROM_OBJ_TYPES,
        )

        return self.frame_type(
            path_raw=self.path_raw,
            path_preprocess=self.path_preprocess,
            path_meshes=self.path_meshes,
            name_unique=name_unique,
            all_categories=self.categories,
            mask_type=OD3D_FRAME_MASK_TYPES.MESH,
            cam_tform4x4_obj_type=OD3D_CAM_TFORM_OBJ_TYPES.META,
            mesh_type=self.mesh_type,
            mesh_feats_type=OD3D_MESH_FEATS_TYPES.M_DINOV2_VITB14_FROZEN_BASE_NO_NORM_T_CENTERZOOM512_R_ACC,
            mesh_feats_dist_reduce_type=OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN_AVG,
            modalities=self.modalities,
            tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW,
            depth_type=OD3D_FRAME_DEPTH_TYPES.MESH,
            kpts2d_annot_type=OD3D_FRAME_KPTS2D_ANNOT_TYPES.META,
            scale_type=self.scale_type,
        )

    @property
    def path_meshes(self):
        return self.path_pascal3d_raw

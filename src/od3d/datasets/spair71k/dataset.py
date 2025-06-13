import logging
import random

logger = logging.getLogger(__name__)
import torch.nn
from typing import Tuple, List, Dict
from od3d.datasets.frame import OD3D_FRAME_MODALITIES
from od3d.datasets.dataset import OD3D_Dataset
from od3d.datasets.frame import OD3D_FRAME_KPTS2D_ANNOT_TYPES
from od3d.datasets.object import OD3D_FRAME_DEPTH_TYPES
from omegaconf import DictConfig
from pathlib import Path
import od3d.io
import shutil
from tqdm import tqdm
from od3d.cv.geometry.objects3d.meshes import Meshes
from od3d.datasets.spair71k.frame import SPair71KFrame, SPair71KFrameMeta
from od3d.datasets.pascal3d.enum import (
    PASCAL3D_CATEGORIES,
    MAP_CATEGORIES_PASCAL3D_TO_OD3D,
    MAP_CATEGORIES_OD3D_TO_PASCAL3D,
)
from od3d.datasets.spair71k.enum import SPAIR71K_SUBSETS
from od3d.datasets.object import OD3D_MESH_TYPES


class SPair71K(OD3D_Dataset):
    map_od3d_categories = MAP_CATEGORIES_OD3D_TO_PASCAL3D
    all_categories = list(PASCAL3D_CATEGORIES)
    frame_type = SPair71KFrame

    def __init__(
        self,
        name: str,
        modalities: List[OD3D_FRAME_MODALITIES],
        path_raw: Path,
        path_preprocess: Path,
        path_pascal3d_raw: Path,
        categories: List[str] = None,
        transform=None,
        index_shift=0,
        subset_fraction=1.0,
        dict_nested_frames: Dict = None,
        dict_nested_frames_ban: Dict = None,
    ):
        # if categories is not None:
        #     if self.map_od3d_categories is not None:
        #         self.categories = [
        #             self.map_od3d_categories.get(category, category)
        #             if category not in self.all_categories
        #             else category
        #             for category in categories
        #         ]
        #         print(f"categories: {self.categories}")
        #     else:
        #         self.categories = categories
        # else:
        #     self.categories = self.all_categories

        if transform is None:
            from od3d.cv.transforms.rgb_uint8_to_float import RGB_UInt8ToFloat

            transform = RGB_UInt8ToFloat()

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
        )

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

    @staticmethod
    def setup(config):
        path_spair71k_raw = Path(config.path_raw)
        if path_spair71k_raw.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous SPair71k")
            shutil.rmtree(path_spair71k_raw)
        if path_spair71k_raw.exists():
            logger.info(f"Found SPair71k dataset at {path_spair71k_raw}")
        else:
            logger.info(f"Downloading SPair71k dataset at {path_spair71k_raw}")
            fpath = path_spair71k_raw.joinpath("spair71k.tar.gz")
            od3d.io.download(url=config.url_spair71k_raw, fpath=fpath)
            logger.info(f"Unzipping SPair71k dataset...")
            od3d.io.untar(fpath=fpath, dst=fpath.parent)
            logger.info(f"Moving SPair71k dataset...")
            od3d.io.move_dir(
                src=fpath.parent.joinpath(
                    Path(config.url_spair71k_raw).with_suffix("").with_suffix("").name,
                ),
                dst=fpath.parent,
            )

    # PREPROCESS META
    @staticmethod
    def extract_meta(config: DictConfig):
        subsets = config.get("subsets", None)
        if subsets is None:
            subsets = SPAIR71K_SUBSETS.list()
        categories = config.get("categories", None)
        if categories is None:
            categories = PASCAL3D_CATEGORIES.list()

        path_raw = SPair71K.get_path_raw(config=config)
        path_meta = SPair71K.get_path_meta(config=config)
        path_pascal3d_raw = Path(config.path_pascal3d_raw)
        # rpath_meshes = SPair71K.get_rpath_meshes()

        if config.extract_meta.remove_previous:
            if path_meta.exists():
                shutil.rmtree(path_meta)

        pair_names = []
        pair_categories = []
        pair_subsets = []
        for subset in subsets:
            fpath_pair_names_partial = path_raw.joinpath(
                "Layout",
                "large",
                f"{subset}.txt",
            )

            with fpath_pair_names_partial.open() as f:
                pair_names_partial = (
                    f.read().splitlines()
                )  # duplication for large and small todo : handle with subset
            pair_names_partial = list(
                filter(lambda x: x.split(":", 1)[1] in categories, pair_names_partial)
            )

            pair_names += [
                pair_partial_name.split(":", 1)[0]
                for pair_partial_name in pair_names_partial
            ]
            pair_categories += [
                pair_partial_name.split(":", 1)[1]
                for pair_partial_name in pair_names_partial
            ]
            pair_subsets += [subset] * len(pair_names_partial)

        # frames_subsets, frames_categories, frames_names = Pascal3DFrameMeta.get_frames_names_from_subsets_and_cateogories_from_raw(path_pascal3d_raw=path_raw, subsets=subsets, categories=categories)

        if config.get("frames", None) is not None:
            pair_names = list(filter(lambda f: f in config.frames, pair_names))

        for i in tqdm(range(len(pair_names))):
            fpath = path_meta.joinpath(
                SPair71KFrameMeta.get_rfpath_from_name_unique(
                    SPair71KFrameMeta.get_name_unique_from_category_subset_name(
                        subset=pair_subsets[i],
                        category=pair_categories[i],
                        name=pair_names[i],
                    ),
                ),
            )
            if not fpath.exists() or config.extract_meta.override:
                frame_meta = SPair71KFrameMeta.load_from_raw(
                    path_raw_pascal3d=path_pascal3d_raw,
                    pair_name=pair_names[i],
                    subset=pair_subsets[i],
                    category=pair_categories[i],
                    path_raw=path_raw,
                )

                if frame_meta is not None:
                    frame_meta.save(path_meta=path_meta)

    # PREPROCESS
    # No preprocess required

    # DATASET PROPERTIES
    def get_frame_by_name_unique(self, name_unique):
        from od3d.datasets.object import (
            OD3D_CAM_TFORM_OBJ_TYPES,
            OD3D_TFROM_OBJ_TYPES,
        )

        return self.frame_type(
            path_raw=self.path_raw,
            # path_pascal3d_raw = self.path_pascal3d_raw,
            path_preprocess=self.path_preprocess,
            name_unique=name_unique,
            all_categories=self.categories,
            modalities=self.modalities,
            kpts2d_annot_type=OD3D_FRAME_KPTS2D_ANNOT_TYPES.META,
            cam_tform4x4_obj_type=OD3D_CAM_TFORM_OBJ_TYPES.META,
            tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW,
        )

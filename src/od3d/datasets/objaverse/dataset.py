import logging

import trimesh

import od3d.io

logger = logging.getLogger(__name__)
from od3d.datasets.dataset import OD3D_Dataset
from od3d.datasets.frame_meta import (
    OD3D_FrameMeta,
    OD3D_FrameMetaRGBMixin,
    OD3D_FrameMetaCategoryMixin,
    OD3D_FrameMetaMaskMixin,
    OD3D_FrameMetaSizeMixin,
)
from od3d.datasets.frame import OD3D_FRAME_MODALITIES, OD3D_Frame

# from od3d.datasets.objectnet3d.enum import OBJECTNET3D_CATEOGORIES
from pathlib import Path
from typing import List, Dict
from omegaconf import DictConfig
import shutil


class ObjaVerse_FrameMeta(
    OD3D_FrameMetaMaskMixin,
    OD3D_FrameMetaRGBMixin,
    OD3D_FrameMetaSizeMixin,
    OD3D_FrameMetaCategoryMixin,
    OD3D_FrameMeta,
):
    @property
    def name_unique(self):
        return f"{self.category}/{self.name}"

    @staticmethod
    def load_from_raw():
        pass


class Objaverse(OD3D_Dataset):
    def __init__(
        self,
        name: str,
        modalities: List[OD3D_FRAME_MODALITIES],
        path_raw: Path,
        path_preprocess: Path,
        categories: List = None,
        dict_nested_frames: Dict = None,
        dict_nested_frames_ban: Dict = None,
        transform=None,
        index_shift=0,
        subset_fraction=1.0,
    ):
        categories = (
            categories if categories is not None else []
        )  # TODO: OBJECTNET3D_CATEOGORIES.list()
        super().__init__(
            categories=categories,
            dict_nested_frames=dict_nested_frames,
            dict_nested_frames_ban=dict_nested_frames_ban,
            name=name,
            modalities=modalities,
            path_raw=path_raw,
            path_preprocess=path_preprocess,
            transform=transform,
            index_shift=index_shift,
            subset_fraction=subset_fraction,
        )

        # directories
        # images/texture/textureXX.jpg # 0...14
        # images/texture/texture0.png
        # models/
        #   model_normalized.json
        #   model_normalized.mtl
        #   model_normalized.obj
        #   model_normalized.solid.binvox
        #   model_normalized.surface.binvox

    def get_item(self, item):
        frame_meta = ObjaVerse_FrameMeta.load_from_meta_with_name_unique(
            path_meta=self.path_meta,
            name_unique=self.list_frames_unique[item],
        )
        return OD3D_Frame(
            path_raw=self.path_raw,
            path_preprocess=self.path_preprocess,
            path_meta=self.path_meta,
            meta=frame_meta,
            modalities=self.modalities,
            categories=self.categories,
        )

    @staticmethod
    def setup(config: DictConfig):
        # logger.info(OmegaConf.to_yaml(config))
        path_raw = Path(config.path_raw)
        if path_raw.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous ObjaVerse")
            shutil.rmtree(path_raw)

        #import objaverse
        #objaverse.__version__
        #import objaverse.xl as oxl
        # 10M
        #annotations = oxl.get_annotations(download_dir=str(path_raw)) # pd.DataFrame
        #annotations["source"].value_counts()
        #annotations["fileType"].value_counts()

        # 1M: Annotations Alignment
        #annotations = oxl.get_alignment_annotations(download_dir=str(path_raw))

        #

        # 47K: Annotations LVIS
        #annotations = oxl.get_alignment_annotations(download_dir=str(path_raw))

        import objaverse
        # annotations = objaverse.load_annotations() # keys: UID, values: annotation
        # uids = objaverse.load_uids()
        # uids = [uid for uid, annotation in annotations.items()]

        lvis_annotations = objaverse.load_lvis_annotations() # keys: category (1156), values: List[UID]
        lvis_uids = []

        category = 'airplane' # 'car_(automobile)' 'airplane'
        category_counts = 20

        for lvis_cat, lvis_cat_uids in lvis_annotations.items():
            if lvis_cat == category:
                lvis_uids += lvis_cat_uids

        # 122 bicyle
        # 70 cup
        # 453 chair
        # 81 sofa
        # 102 car_(automobile)
        # 112 airplane

        import random
        import multiprocessing
        processes = multiprocessing.cpu_count()

        random.seed(42)

        uids = lvis_uids
        random_object_uids = random.sample(uids, k=category_counts)

        objects = objaverse.load_objects(
            uids=random_object_uids,
            download_processes=processes
        )
        #objects

        from od3d.cv.geometry.objects3d.meshes import Meshes
        import torch
        from od3d.cv.geometry.transform import get_cam_tform4x4_obj_for_viewpoints_count
        from od3d.cv.visual.show import show_scene, show_imgs
        from od3d.cv.visual.show import get_default_camera_intrinsics_from_img_size

        dtype = torch.float
        device = 'cuda:0'

        from od3d.io import read_json

        canon = read_json(fpath ="src/od3d/datasets/objaverse/canon.json")
        # from https://github.com/JinLi998/CanonObjaverseDataset/blob/master/data/CanonicalObjaverseDataset.json
        # - category (lower case): ->
        canon_airplane = {el[0]: el[1] for el in canon[category]}

        logger.info(f"found {len(canon_airplane)}:{len(lvis_uids)} labels for category {category}")
        from od3d.cv.visual.show import render_trimesh_to_tensor
        import trimesh
        for i in range(len(list(objects.values()))):
            uid = str(random_object_uids[i])
            fpath = list(objects.values())[i]
            if uid in canon_airplane.keys():
                tform = canon_airplane[uid]
                tform = torch.Tensor(tform)
                _mesh = trimesh.load_mesh(fpath)
                _mesh.vertices = (tform[None,] @ _mesh.vertices[..., None])[..., 0]
                _mesh.show()

        a = Meshes.read_from_ply_file(fpath=Path(list(objects.values())[0], device=device))

        H = 512
        W = 512

        cam_intr4x4 = get_default_camera_intrinsics_from_img_size(
            H=H,
            W=W,
            dtype=dtype,
            device=device,
        )
        img_size = torch.Tensor([H, W]).to(dtype=dtype, device=device)
        cam_tform4x4_obj = get_cam_tform4x4_obj_for_viewpoints_count(viewpoints_count=3, dist=120).to(dtype=dtype, device=device)
        imgs = a.render(cams_intr4x4=cam_intr4x4, cams_tform4x4_obj=cam_tform4x4_obj,
                        imgs_sizes=img_size, broadcast_batch_and_cams=True, modalities=['rgb'], )
        show_imgs(imgs['rgb'])
        # show_scene(meshes=a)
        #import trimesh
        #trimesh.load(list(objects.values())[0]).show()

        # DOWNLOAD
        # sample a single object from each source
        sampled_df = annotations.groupby('source').apply(lambda x: x.sample(1)).reset_index(drop=True)
        oxl.download_objects(objects=sampled_df)

        # git config --global credential.helper store
        # huggingface-cli login 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
        from huggingface_hub import login

        login("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", add_to_git_credential=True)

        "https://huggingface.co/datasets/ObjaVerse/ObjaVerseCore-archive/blob/main/ObjaVerseCore.v2.zip"

        # dataset = load_dataset("ObjaVerse/ObjaVerseCore")
        # from datasets import load_dataset
        # --filter=blob:none
        # dataset = load_dataset('ObjaVerse/ObjaVerseCore', cache_dir="/scratch/sommerl/repos/NeMo/ObjaVerseCore")
        # git clone --filter=blob:none https://huggingface.co/datasets/ObjaVerse/ObjaVerseCore
        # https://huggingface.co/datasets/ObjaVerse/ObjaVerseCore/tree/main
        # from datasets import get_dataset_split_names
        # from datasets import load_dataset_builder
        # from datasets import load_dataset
        # from datasets import get_dataset_config_names
        # "ObjaVerse/ObjaVerseCore"  "rotten_tomatoes"
        # ds_builder = load_dataset_builder("ObjaVerse/ObjaVerseCore")
        # ds_builder.info.description
        # ds_builder.info.features
        # get_dataset_config_names("ObjaVerse/ObjaVerseCore")
        # # load_dataset('LOADING_SCRIPT', cache_dir="PATH/TO/MY/CACHE/DIR")

        path_raw.mkdir(parents=True, exist_ok=True)
        od3d.io.run_cmd("pip install opendatalab", logger=logger, live=True)
        od3d.io.run_cmd(
            'please signup and login first with "odl login"',
            logger=logger,
            live=True,
        )
        #  -u {config.credentials.opendatalab.username} -p {config.credentials.opendatalab.password}
        od3d.io.run_cmd(
            "odl info   OpenXD-OmniObject3D-New",
            logger=logger,
            live=True,
        )  # View dataset metadata
        od3d.io.run_cmd(
            "odl ls     OpenXD-OmniObject3D-New",
            logger=logger,
            live=True,
        )  # View a list of dataset files
        od3d.io.run_cmd(
            f"cd {path_raw} && odl get    OpenXD-OmniObject3D-New",
            logger=logger,
            live=True,
        )

# setup dropbox
# https://www.dropbox.com/scl/fo/ixmai3d7uf4mzp3le8sz3/ALRxBZUhhaAs11xH56rJXnM?rlkey=sn7kyuart2i8ujeu1vygz4wcy&e=1&st=j6o0fe8l&dl=0
#
# from od3d.io import read_json
# from pathlib import Path
# # # real_obj_meta
# real = read_json(Path("/data/lmbraid21/jesslen/Omni6DPose/Meta/real_obj_meta.json"))
# syn = read_json(Path("/data/lmbraid21/jesslen/Omni6DPose/Meta/obj_meta.json"))
# box, shoe, doll, teapot, handbag, dinosaur, mug, flower_pot
# syn['instance_dict']['omniobject3d-whistle_025']
# 'object_id': e.g. 'omniobject3d-whistle_025'
# 'source': e.g. 'omniobject3d'
# 'name': e.g. 'whistle_025'
# 'class_label': e.g. 153
# 'class_name': e.g. 'whistle'
# 'dimensions': e.g. [30.647, 58.113, 30.219]
# 'tag':
#     'datatype': e.g. "train"
#     'materialOptions': e.g. ["raw", "diffuse"]
# for inst_name, inst_data in syn['instance_dict'].items():
#    print(inst_name, "  ", inst_data['tag']['datatype'])
# ['class_name']: e.g. 'whistle'
# ['tag']['datatype']: e.g. 'train' / 'test'
# # ['class_list', 'instance_dict']
# for cls_data in real['class_list']:
#     print(cls_data['name'], cls_data['label'], len(cls_data['instance_ids']))
#
#
# from od3d.io import read_json
# real = read_json(Path("/data/lmbraid21/jesslen/Omni6DPose/Meta/real_obj_meta.json"))
# for cls_data in real['class_list']:
#     print(cls_data['name'].upper(), f": \"{cls_data['name']}\"")
#
#
# for cls_data in syn['class_list']:
#     print(cls_data['name'], cls_data['label'], len(cls_data['instance_ids']))
import logging

import torch
from od3d.cv.geometry.objects3d.objects3d import PROJECT_MODALITIES
from od3d.cv.geometry.transform import transf3d_broadcast
from od3d.cv.io import write_depth_image
from od3d.cv.io import write_image
from od3d.data.ext_dicts import rollup_flattened_dict
from od3d.data.ext_dicts import unroll_nested_dict

logger = logging.getLogger(__name__)

from od3d.datasets.frame import OD3D_FRAME_MODALITIES
from pathlib import Path

from od3d.datasets.omni6dpose.frame import Omni6DPose_Frame, Omni6DPose_FrameMeta
from od3d.datasets.omni6dpose.sequence import Omni6DPose_Sequence
from od3d.datasets.omni6dpose.enum import (
    OMNI6DPOSE_CATEGORIES,
    OMNI6DPOSE_SUBSETS,
    MAP_CATEGORIES_OD3D_TO_OMNI6DPOSE,
)

from od3d.datasets.dataset import OD3D_SequenceDataset
from od3d.datasets.object import (
    OD3D_FRAME_MASK_TYPES,
    OD3D_CAM_TFORM_OBJ_TYPES,
    OD3D_MESH_TYPES,
    OD3D_PCL_TYPES,
    OD3D_SEQUENCE_SFM_TYPES,
    OD3D_TFROM_OBJ_TYPES,
    OD3D_MESH_FEATS_TYPES,
    OD3D_MESH_FEATS_DIST_REDUCE_TYPES,
    OD3D_FRAME_DEPTH_TYPES,
)
from pathlib import Path
from typing import List, Dict

from od3d.datasets.omni6dpose.frame import Omni6DPose_Frame, Omni6DPose_FrameMeta
from od3d.datasets.omni6dpose.sequence import (
    Omni6DPose_Sequence,
    Omni6DPose_SequenceMeta,
)
from od3d.io import run_cmd, read_json
from omegaconf import DictConfig
import shutil
from tqdm import tqdm


class Omni6DPose(OD3D_SequenceDataset):
    all_categories = list(OMNI6DPOSE_CATEGORIES)
    map_od3d_categories = MAP_CATEGORIES_OD3D_TO_OMNI6DPOSE
    sequence_type = Omni6DPose_Sequence
    frame_type = Omni6DPose_Frame
    tform_obj_type = OD3D_TFROM_OBJ_TYPES.RAW  # LABEL3D_CUBOID
    sfm_type = OD3D_SEQUENCE_SFM_TYPES.META
    cam_tform_obj_type = OD3D_CAM_TFORM_OBJ_TYPES.META
    mask_type = OD3D_FRAME_MASK_TYPES.META
    pcl_type = OD3D_PCL_TYPES.META
    mesh_type = OD3D_MESH_TYPES.ALPHA500
    dict_nested_sequences_filter_cat_w_name = False

    mesh_feats_type = (
        OD3D_MESH_FEATS_TYPES.M_DINOV2_VITB14_FROZEN_BASE_NO_NORM_T_CENTERZOOM512_R_ACC
    )
    mesh_feats_dist_reduce_type = OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN_AVG
    dict_nested_frames_struct = "subset/sequence/frame"

    def __init__(
        self,
        name: str,
        modalities: List[OD3D_FRAME_MODALITIES],
        path_raw: Path,
        path_preprocess: Path,
        categories: List[OMNI6DPOSE_CATEGORIES] = None,
        dict_nested_frames: Dict[str, Dict[str, List[str]]] = None,
        dict_nested_frames_ban: Dict[str, Dict[str, List[str]]] = None,
        frames_count_max_per_sequence=None,
        sequences_count_max_per_category=None,
        transform=None,
        index_shift=0,
        subset_fraction=1.0,
        mesh_type: OD3D_MESH_TYPES = OD3D_MESH_TYPES.ALPHA500,
        mesh_feats_type=OD3D_MESH_FEATS_TYPES.M_DINOV2_VITB14_FROZEN_BASE_NO_NORM_T_CENTERZOOM512_R_ACC,
        mesh_feats_dist_reduce_type=OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN_AVG,
        tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW,
        dict_nested_sequences_filter_cat_w_name=False,
        dict_nested_frames_filter_cat_w_name=False,
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
            dict_nested_frames_filter_cat_w_name=dict_nested_frames_filter_cat_w_name,
        )
        if (
            not dict_nested_sequences_filter_cat_w_name
            and not dict_nested_frames_filter_cat_w_name
        ):
            self.categories = self.all_categories
        self.mesh_type = mesh_type
        self.mesh_feats_type = mesh_feats_type
        self.mesh_feats_dist_reduce_type = mesh_feats_dist_reduce_type
        self.tform_obj_type = tform_obj_type

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
            pcl_type=self.pcl_type,
            sfm_type=self.sfm_type,
            modalities=self.modalities,
            tform_obj_type=self.tform_obj_type,
            depth_type=OD3D_FRAME_DEPTH_TYPES.META,
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
            mesh_feats_type=self.mesh_feats_type,
            mesh_feats_dist_reduce_type=self.mesh_feats_dist_reduce_type,
            pcl_type=self.pcl_type,
            sfm_type=self.sfm_type,
            modalities=self.modalities,
            tform_obj_type=self.tform_obj_type,
            depth_type=OD3D_FRAME_DEPTH_TYPES.META,
        )

    @staticmethod
    def setup(config: DictConfig):
        # logger.info(OmegaConf.to_yaml(config))
        path_raw = Path(config.path_raw)
        if path_raw.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous Omni6DPose")
            shutil.rmtree(path_raw)

        if path_raw.exists() and not config.setup.override:
            logger.info(f"Found Omni6DPose dataset at {path_raw}")
        else:
            # # setup dropbox
            run_cmd(
                cmd=f"cd {path_raw} && wget https://www.dropbox.com/scl/fo/ixmai3d7uf4mzp3le8sz3/ALRxBZUhhaAs11xH56rJXnM?rlkey=sn7kyuart2i8ujeu1vygz4wcy&e=1&st=j6o0fe8l&dl=0",
                live=True,
                logger=logger,
            )

    def preprocess(self, config_preprocess: DictConfig):
        logger.info("preprocess")
        for key in config_preprocess.keys():
            if key == "objcentric" and config_preprocess.objcentric.get(
                "enabled", False
            ):
                override = config_preprocess.objcentric.get("override", False)
                remove_previous = config_preprocess.objcentric.get(
                    "remove_previous", False
                )
                self.preprocess_objcentric(
                    override=override,
                    remove_previous=remove_previous,
                )
            if key == "bboxs" and config_preprocess.bboxs.get("enabled", False):
                override = config_preprocess.bboxs.get("override", False)
                remove_previous = config_preprocess.bboxs.get("remove_previous", False)
                self.preprocess_bboxs(
                    override=override,
                    remove_previous=remove_previous,
                )

            # if key == "pcl" and config_preprocess.pcl.get("enabled", False):
            #     override = config_preprocess.pcl.get("override", False)
            #     remove_previous = config_preprocess.pcl.get("remove_previous", False)
            #     self.preprocess_pcl(override=override)

            # if key == "tform_obj" and config_preprocess.tform_obj.get("enabled", False):
            #    override = config_preprocess.tform_obj.get("override", False)
            #    self.preprocess_tform_obj(override=override)

            if key == "mesh" and config_preprocess.mesh.get("enabled", False):
                override = config_preprocess.mesh.get("override", False)
                remove_previous = config_preprocess.mesh.get("remove_previous", False)
                self.preprocess_mesh(override=override, remove_previous=remove_previous)

    def preprocess_mesh(self, remove_previous=False, override=False):
        logger.info("preprocess mesh...")
        from od3d.datasets.sequence_meta import OD3D_SequenceMeta

        if remove_previous:
            logger.info("remove mesh...")
            shutil.rmtree(self.path_preprocess.joinpath("mesh"))

        from copy import copy
        from functools import partial

        # mesh_type = copy(self.mesh_type)
        # self.mesh_type = OD3D_MESH_TYPES.META

        modalities = [
            # OD3D_FRAME_MODALITIES.MESH,
            # OD3D_FRAME_MODALITIES.OBJ_TFORM4X4_OBJS,
            OD3D_FRAME_MODALITIES.CAM_INTR4X4,
            # OD3D_FRAME_MODALITIES.CAM_TFORM4X4_OBJ,
            OD3D_FRAME_MODALITIES.SIZE,
        ]

        self.modalities = modalities
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            collate_fn=partial(self.collate_fn, modalities=modalities),
        )
        logging.info(f"Dataset contains {len(self)} frames.")

        from od3d.cv.io import get_default_device

        device = get_default_device()
        for batch in iter(tqdm(dataloader)):
            logger.info(f"{batch.name_unique[0]}")  # sequence_name[0]}')

            frame = self.get_frame_by_name_unique(name_unique=batch.name_unique[0])
            fpaths_meshs_meta = frame.get_fpaths_meshs(mesh_type=OD3D_MESH_TYPES.META)
            fpaths_meshs = frame.get_fpaths_meshs(
                mesh_type=OD3D_MESH_TYPES.ALPHA500
            )  # self.mesh_type

            for i, fpath_mesh_meta in enumerate(tqdm(fpaths_meshs_meta)):
                fpath_mesh = fpaths_meshs[i]

                if fpath_mesh.exists() and not override:
                    logger.info(f"mesh exists at {fpath_mesh}, skipping...")
                    continue

                logger.info(f"{fpath_mesh_meta}, {fpath_mesh}")
                from od3d.cv.geometry.objects3d.meshes import Meshes

                mesh = Meshes.read_from_ply_file(fpath_mesh_meta)
                mesh_smpl = mesh.get_simplified_mesh()
                mesh_smpl.write_to_file(fpath=fpath_mesh)

        # fpaths_meshs_meta = []
        # fpaths_meshs = []
        # for sequence_name_unique in tqdm(OD3D_SequenceMeta.unroll_nested_metas(
        #     self.dict_nested_sequences,
        # )):
        #     sequence = self.get_sequence_by_name_unique(
        #         name_unique=sequence_name_unique,
        #     )
        #     fpaths_meshs_meta += sequence.first_frame.get_fpaths_meshs(mesh_type=OD3D_MESH_TYPES.META)
        #     fpaths_meshs += sequence.first_frame.get_fpaths_meshs(mesh_type=self.mesh_type)
        #
        # for i, fpath_mesh_meta in enumerate(tqdm(fpaths_meshs_meta)):
        #     fpath_mesh = fpaths_meshs[i]
        #
        #     if fpath_mesh.exists() and not override:
        #         logger.info(f"mesh exists at {fpath_mesh}, skipping...")
        #         continue
        #
        #     logger.info(f"{fpath_mesh_meta}, {fpath_mesh}")
        #     from od3d.cv.geometry.objects3d.meshes import Meshes
        #     mesh = Meshes.read_from_ply_file(fpath_mesh_meta)
        #     mesh_smpl = mesh.get_simplified_mesh()
        #
        #     mesh_smpl.write_to_file(fpath=fpath_mesh)

    def preprocess_objcentric(self, remove_previous=False, override=False):
        logger.info("preprocess objcentric...")
        from functools import partial
        import shutil

        subdirs = ["rgb", "depth", "depth_mask", "pxl_cat_id", "bboxs"]
        if remove_previous:
            logger.info("remove objcentric...")
            for subdir in subdirs:
                shutil.rmtree(self.path_preprocess.joinpath(subdir))

        modalities = [
            OD3D_FRAME_MODALITIES.MESH,
            OD3D_FRAME_MODALITIES.CAM_INTR4X4,
            OD3D_FRAME_MODALITIES.CAM_TFORM4X4_OBJ,
            OD3D_FRAME_MODALITIES.OBJ_TFORM4X4_OBJS,
            OD3D_FRAME_MODALITIES.SIZE,
        ]

        self.modalities = modalities
        from copy import copy

        mesh_type = copy(self.mesh_type)
        self.mesh_type = OD3D_MESH_TYPES.META

        if not override:
            # filter already preprocessed images
            dict_nested_frames_unrolled = unroll_nested_dict(self.dict_nested_frames)
            dict_nested_frames_unrolled_filtered = {}
            for frames_subdir, frames_names in tqdm(
                dict_nested_frames_unrolled.items()
            ):
                for frame_name in frames_names:
                    if (
                        not self.path_preprocess.joinpath(
                            f"bboxs/{frames_subdir}/{frame_name}/bboxs.pt"
                        ).exists()
                        or not self.path_preprocess.joinpath(
                            f"rgb/{frames_subdir}/{frame_name}.png"
                        ).exists()
                        or not self.path_preprocess.joinpath(
                            f"pxl_cat_id/{frames_subdir}/{frame_name}.png"
                        ).exists()
                        or not self.path_preprocess.joinpath(
                            f"depth/{frames_subdir}/{frame_name}.png"
                        ).exists()
                    ):
                        if frames_subdir not in dict_nested_frames_unrolled_filtered:
                            dict_nested_frames_unrolled_filtered[frames_subdir] = []
                        dict_nested_frames_unrolled_filtered[frames_subdir] += [
                            frame_name
                        ]
            dict_nested_frames_filtered = rollup_flattened_dict(
                dict_nested_frames_unrolled_filtered
            )
            self = self.get_subset_with_dict_nested_frames(
                dict_nested_frames=dict_nested_frames_filtered
            )

        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=2,
            num_workers=0,
            shuffle=False,
            collate_fn=partial(self.collate_fn, modalities=modalities),
        )
        logging.info(f"Dataset contains {len(self)} frames.")

        from od3d.cv.io import get_default_device

        device = get_default_device()
        for batch in iter(tqdm(dataloader)):
            logger.info(f"{batch.name_unique[0]}")  # sequence_name[0]}')
            batch.to(device=device)
            from od3d.cv.visual.show import (
                show_scene,
                show_img,
                render_trimesh_to_tensor,
                show_imgs,
            )

            obj_size = batch.mesh.verts.max() * 2

            mask = batch.mesh.render(
                cams_tform4x4_obj=batch.cam_tform4x4_obj,
                cams_intr4x4=batch.cam_intr4x4,
                imgs_sizes=batch.size,
                objects_ids=torch.LongTensor(batch.mesh_id_in_batch).to(device=device),
                modalities=PROJECT_MODALITIES.MASK,
                broadcast_batch_and_cams=False,
                obj_tform4x4_objs=torch.stack(batch.obj_tform4x4_objs).to(
                    device=device
                ),
                zfar=10 * obj_size,
            )

            # show_imgs(mask)
            for i, frame_name_unique in enumerate(batch.name_unique):
                frame = self.get_frame_by_name_unique(name_unique=frame_name_unique)

                import trimesh

                _mesh_trimesh = trimesh.load(
                    frame.get_fpaths_meshs(mesh_type=OD3D_MESH_TYPES.META)[0],
                    force="mesh",
                )
                obj_size = _mesh_trimesh.vertices.max() * 2
                H = batch.size[0]
                W = batch.size[1]
                rgb, depth = render_trimesh_to_tensor(
                    mesh_trimesh=_mesh_trimesh,
                    cam_tform4x4_obj=batch.cam_tform4x4_obj[i],
                    cam_intr4x4=batch.cam_intr4x4[i],
                    H=H,
                    W=W,
                    rgb_bg=[0.8, 0.8, 0.8],
                    zfar=obj_size * 10,
                )

                if frame.fpath_rgb.exists() and not override:
                    logger.info(f"rgb exists at {frame.fpath_rgb}, skipping...")
                else:
                    write_image(img=rgb, path=frame.fpath_rgb)

                if frame.fpath_depth.exists() and not override:
                    logger.info(f"depth exists at {frame.fpath_depth}, skipping...")
                else:
                    write_depth_image(img=depth, path=frame.fpath_depth)

                if frame.fpath_pxl_cat_id.exists() and not override:
                    logger.info(
                        f"pxl_cat_id exists at {frame.fpath_pxl_cat_id}, skipping..."
                    )
                else:
                    from od3d.datasets.omni6dpose.enum import (
                        MAP_CATEGORIES_OMNI6DPOSE_TO_ID,
                    )

                    dataset_specific_id = MAP_CATEGORIES_OMNI6DPOSE_TO_ID[
                        frame.categories[0]
                    ]
                    write_image(
                        img=(dataset_specific_id * (mask[i] > 0)).to(dtype=torch.uint8),
                        path=frame.fpath_pxl_cat_id,
                    )

                if frame.fpath_bboxs.exists() and not override:
                    logger.info(f"bboxs exists at {frame.fpath_bboxs}, skipping...")
                else:
                    from od3d.cv.visual.draw import get_bboxs_from_masks

                    bboxs = get_bboxs_from_masks(mask[i])
                    frame.write_bboxs(bboxs)

        self.mesh_type = mesh_type

        # from od3d.cv.io import read_image
        # b = read_image(path=frame.fpath_pxl_cat_id)

        # Path("/data/lmbraid21/jesslen/Omni6DPose/Meta/real_obj_meta.json")
        # Path("/data/lmbraid21/jesslen/Omni6DPose/Meta/obj_meta.json")
        # from od3d.io import read_json
        # meta = read_json(Path("/data/lmbraid21/jesslen/Omni6DPose/Meta/obj_meta.json"))['class_list']

        # if frame.fpath_rgb.exists() and not override:
        #     logger.info(f"rgbs exists at {frame.fpath_rgb}, skipping...")
        #     continue
        #
        # mods = mesh.render(cams_tform4x4_obj=cams_tform4x4_obj, cams_intr4x4=cam_intr4x4,
        #                    imgs_sizes=imgs_sizes, modalities=["mask"],
        #                    broadcast_batch_and_cams=True, rgb_bg=[0., 0., 0.],
        #                    rgb_diffusion_alpha=0.3)
        #
        # # rgbs = mods['rgb'][0]
        # masks = mods['mask'][0]
        #
        # # from od3d.cv.visual.show import show_imgs
        # # show_imgs(rgbs)
        # pts3d = []
        # for i, cam_tform4x4_obj in enumerate(cams_tform4x4_obj):
        #     # cam_tform4x4_obj_2 = cam_tform4x4_obj.clone()
        #     # cam_tform4x4_obj_2[2, 3] = 1000.
        #     rgb, depth = render_trimesh_to_tensor(mesh_trimesh=_mesh_trimesh,
        #                                           cam_tform4x4_obj=cam_tform4x4_obj,
        #                                           cam_intr4x4=cam_intr4x4, H=H, W=W,
        #                                           rgb_bg=[0.8, 0.8, 0.8], zfar=obj_scale * 10)
        #     depth_mask = depth > 0.
        #
        #     _pts3d = \
        #         depth2pts3d_grid(depth=depth.to(device), cam_intr4x4=cam_intr4x4).permute(1, 2, 0)[
        #             depth_mask[0]]
        #     _pts3d = random_sampling(pts3d_cls=_pts3d, pts3d_max_count=1024)
        #     _pts3d = transf3d_broadcast(pts3d=_pts3d, transf4x4=inv_tform4x4(cam_tform4x4_obj))
        #     pts3d.append(_pts3d)
        #     write_image(img=rgb, path=frame_fpath_rgb)
        #     write_image(img=masks[i], path=frame_fpath_mask)
        #     write_image(img=masks[i] * cat_id, path=frame_fpath_pxl_cat_id)
        #     write_depth_image(img=depth, path=frame_fpath_depth)
        #     write_mask_image(img=depth_mask, path=frame_fpath_depth_mask)
        #
        #
        #
        #
        #
        # scene = batch.mesh.render(cams_tform4x4_obj=batch.cam_tform4x4_obj,
        #                          cams_intr4x4=batch.cam_intr4x4,
        #                          imgs_sizes=batch.size,
        #                          objects_ids=batch.mesh_id_in_batch[0],
        #                          modalities=PROJECT_MODALITIES.MASK,
        #                          broadcast_batch_and_cams=True,
        #                          obj_tform4x4_objs=batch.obj_tform4x4_objs[0].to(device=device))
        #
        #

        # frame = self.get_frame_by_name_unique(name_unique=batch.name_unique[0])
        # if frame.fpath_bboxs.exists() and not override:
        #     logger.info(f"bboxs exists at {frame.fpath_bboxs}, skipping...")
        #     continue
        #
        #
        #
        # from od3d.cv.geometry.objects3d.meshes import PROJECT_MODALITIES
        # scene = batch.mesh.render(cams_tform4x4_obj=batch.cam_tform4x4_obj,
        #                          cams_intr4x4=batch.cam_intr4x4,
        #                          imgs_sizes=batch.size,
        #                          objects_ids=batch.mesh_id_in_batch[0],
        #                          modalities=PROJECT_MODALITIES.MASK,
        #                          broadcast_batch_and_cams=True,
        #                          obj_tform4x4_objs=batch.obj_tform4x4_objs[0].to(device=device))

    def preprocess_bboxs(self, remove_previous=False, override=False):
        logger.info("preprocess bboxs...")
        from functools import partial
        import shutil

        bboxs_subsets = ["bbox_mask_amodal", "bbox_mask_modal"]

        if remove_previous:
            # rgb/test_real_bbox_mask_modal
            # meta/frames/test_real_bbox_mask_modal
            # meta/sequences/test_real_bbox_mask_modal
            subdirs = [
                "meta/frames",
                "meta/sequences",
                "rgb",
                "depth",
                "pxl_cat_id",
                "bboxs",
                "mask",
                "mask_amodal",
            ]
            if self.dict_nested_frames is None:
                msg = f"subsets are not defined in dict nested frames cannot remove previous..."
                raise NotImplementedError(msg)
            subsets = self.dict_nested_frames.keys()
            for subset in subsets:
                if "bbox" in subset:  # no preprocessing required for these subsets
                    continue
                for subdir in subdirs:
                    if subdir == "bboxs":
                        logger.info(f"remove {subdir}/{subset}...")
                        if self.path_preprocess.joinpath(subdir, subset).exists():
                            shutil.rmtree(self.path_preprocess.joinpath(subdir, subset))
                    else:
                        for bbox_subset in bboxs_subsets:
                            logger.info(f"remove {subdir}/{subset}_{bbox_subset}...")
                            if self.path_preprocess.joinpath(
                                subdir, f"{subset}_{bbox_subset}"
                            ).exists():
                                shutil.rmtree(
                                    self.path_preprocess.joinpath(
                                        subdir, f"{subset}_{bbox_subset}"
                                    )
                                )

        if not override:
            # filter already preprocessed images
            dict_nested_frames_unrolled = unroll_nested_dict(self.dict_nested_frames)
            dict_nested_frames_unrolled_filtered = {}
            for frames_subdir, frames_names in tqdm(
                dict_nested_frames_unrolled.items()
            ):
                if (
                    "bbox" in frames_subdir
                ):  # no preprocessing required for these subsets
                    continue
                for frame_name in frames_names:
                    frames_subset = frames_subdir.split("/")[0]
                    frames_seq = frames_subdir.split("/")[1]

                    if (
                        not self.path_preprocess.joinpath(
                            f"meta/sequences/{frames_subset}_bbox_mask_amodal/{frames_seq}_{frame_name}.yaml"
                        ).exists()
                    ) or (
                        not self.path_preprocess.joinpath(
                            f"meta/sequences/{frames_subset}_bbox_mask_modal/{frames_seq}_{frame_name}.yaml"
                        ).exists()
                    ):
                        if frames_subdir not in dict_nested_frames_unrolled_filtered:
                            dict_nested_frames_unrolled_filtered[frames_subdir] = []
                        dict_nested_frames_unrolled_filtered[frames_subdir] += [
                            frame_name
                        ]
            dict_nested_frames_filtered = rollup_flattened_dict(
                dict_nested_frames_unrolled_filtered
            )
            self = self.get_subset_with_dict_nested_frames(
                dict_nested_frames=dict_nested_frames_filtered
            )

        from copy import copy

        mesh_type = copy(self.mesh_type)
        # self.mesh_type = OD3D_MESH_TYPES.META

        modalities = [
            OD3D_FRAME_MODALITIES.RGB,
            OD3D_FRAME_MODALITIES.DEPTH,
            OD3D_FRAME_MODALITIES.PXL_CAT_ID,
            OD3D_FRAME_MODALITIES.CATEGORIES,
            OD3D_FRAME_MODALITIES.CATEGORIES_IDS,
            OD3D_FRAME_MODALITIES.MESH,
            OD3D_FRAME_MODALITIES.CAM_INTR4X4,
            OD3D_FRAME_MODALITIES.CAM_TFORM4X4_OBJ,
            OD3D_FRAME_MODALITIES.OBJ_TFORM4X4_OBJS,
            OD3D_FRAME_MODALITIES.SIZE,
        ]

        self.modalities = modalities
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=1,
            num_workers=4,
            shuffle=False,
            collate_fn=partial(self.collate_fn, modalities=modalities),
        )
        logging.info(f"Dataset contains {len(self)} frames.")

        from od3d.cv.io import get_default_device

        device = get_default_device()
        for batch in iter(tqdm(dataloader)):
            logger.info(f"{batch.name_unique[0]}")  # sequence_name[0]}')

            frame = self.get_frame_by_name_unique(name_unique=batch.name_unique[0])

            batch.to(device=device)

            from od3d.cv.geometry.objects3d.meshes import PROJECT_MODALITIES

            mask_amodal = batch.mesh.render(
                cams_tform4x4_obj=batch.cam_tform4x4_obj,
                cams_intr4x4=batch.cam_intr4x4,
                imgs_sizes=batch.size,
                objects_ids=batch.mesh_id_in_batch[0],
                modalities=PROJECT_MODALITIES.MASK,
                broadcast_batch_and_cams=True,
                obj_tform4x4_objs=batch.obj_tform4x4_objs[0].to(device=device),
            )

            # scene: [16, 1, 1, 512, 512]) get bounding box per object in 16
            # from od3d.cv.visual.show import show_img, show_imgs
            # show_imgs(scene[:, 0, 0], height=512, width=512)

            from od3d.cv.visual.draw import get_bboxs_from_masks

            bboxs_mask_amodal = get_bboxs_from_masks(mask_amodal)[:, 0, 0]

            if frame.fpath_bboxs.exists() and not override:
                logger.info(f"bboxs exists at {frame.fpath_bboxs}, skipping...")
            else:
                frame.write_bboxs(bboxs_mask_amodal)

            mask = batch.mesh.render(
                cams_tform4x4_obj=batch.cam_tform4x4_obj,
                cams_intr4x4=batch.cam_intr4x4,
                imgs_sizes=batch.size,
                objects_ids=torch.LongTensor(batch.mesh_id_in_batch).to(device=device),
                modalities=PROJECT_MODALITIES.OBJ_IN_SCENE_ONEHOT,
                broadcast_batch_and_cams=False,
                obj_tform4x4_objs=torch.stack(batch.obj_tform4x4_objs).to(
                    device=device
                ),
            )

            # from od3d.cv.visual.show import show_img, show_imgs
            # show_imgs(mask[0], height=512, width=512)
            # mask: [1, 16, 512, 512]
            # from od3d.cv.visual.show import show_img, show_imgs
            # show_imgs([scene[:, 0, 0], mask[0]], height=512, width=512)

            from od3d.cv.visual.draw import get_bboxs_from_masks

            bboxs_mask_modal = get_bboxs_from_masks(mask)[0]

            H_out = 512
            W_out = 512
            # TODO: how to add bboxs from artur? load at this point
            # TYPES: bbox_mask_amodal, bbox_mask_modal, bbox_cubercnn
            for bbox_subset in bboxs_subsets:
                if bbox_subset == "bbox_mask_amodal":
                    bboxs = bboxs_mask_amodal
                elif bbox_subset == "bbox_mask_modal":
                    bboxs = bboxs_mask_modal
                else:
                    raise NotImplementedError

                for bbox_id in range(len(bboxs)):
                    bbox = bboxs[bbox_id]
                    from od3d.cv.visual.crop import crop_with_bbox

                    rgb, cam_crop_tform_cam = crop_with_bbox(
                        img=batch.rgb[0],
                        bbox=bbox,
                        H_out=H_out,
                        W_out=W_out,
                    )
                    # from od3d.cv.visual.show import show_img, show_imgs
                    # show_imgs([batch.rgb[0], rgb])

                    pxl_cat_id, _ = crop_with_bbox(
                        img=batch.pxl_cat_id[0],
                        bbox=bbox,
                        mode="nearest_v2",
                        H_out=H_out,
                        W_out=W_out,
                    )

                    bbox_mask, _ = crop_with_bbox(
                        img=mask[:, bbox_id],
                        bbox=bbox,
                        H_out=H_out,
                        W_out=W_out,
                    )

                    bbox_mask_amodal, _ = crop_with_bbox(
                        img=mask_amodal[bbox_id, :, 0],
                        bbox=bbox,
                        H_out=H_out,
                        W_out=W_out,
                    )

                    # show_img(pxl_cat_id)
                    # show_img(batch.pxl_cat_id[0])

                    depth, _ = crop_with_bbox(
                        img=batch.depth[0, None],
                        bbox=bbox,
                        H_out=H_out,
                        W_out=W_out,
                    )

                    cam_intr4x4 = torch.bmm(
                        cam_crop_tform_cam[None,],
                        batch.cam_intr4x4.clone()[0, None],
                    )[0]

                    l_size = [rgb.shape[-2], rgb.shape[-1]]

                    subset = f"{self.subset}_{bbox_subset}"
                    frame_name = f"{frame.meta.categories[bbox_id]}_{int(bbox_id):03}"
                    sequence_name = f"{frame.meta.sequence_name}_{frame.meta.name}"

                    # frame_name = f"{int(i):03}"
                    frame_fpath_mask = self.path_preprocess.joinpath(
                        f"mask/{subset}/{sequence_name}/{frame_name}.png",
                    )
                    frame_fpath_mask_amodal = self.path_preprocess.joinpath(
                        f"mask_amodal/{subset}/{sequence_name}/{frame_name}.png",
                    )

                    frame_fpath_rgb = self.path_preprocess.joinpath(
                        f"rgb/{subset}/{sequence_name}/{frame_name}.png",
                    )
                    # frame_fpath_mask = path_raw.joinpath(
                    #    f"mask/{subset}/{sequence_name}/{frame_name}.png")
                    frame_fpath_pxl_cat_id = self.path_preprocess.joinpath(
                        f"pxl_cat_id/{subset}/{sequence_name}/{frame_name}.png",
                    )
                    frame_fpath_depth = self.path_preprocess.joinpath(
                        f"depth/{subset}/{sequence_name}/{frame_name}.png",
                    )
                    frame_fpath_depth_mask = self.path_preprocess.joinpath(
                        f"depth_mask/{subset}/{sequence_name}/{frame_name}.png",
                    )

                    if frame_fpath_rgb.exists() and not override:
                        logger.info(f"rgb exists at {frame_fpath_rgb}, skipping...")
                    else:
                        write_image(img=rgb, path=frame_fpath_rgb)

                    if frame_fpath_depth.exists() and not override:
                        logger.info(f"depth exists at {frame_fpath_depth}, skipping...")
                    else:
                        write_depth_image(img=depth, path=frame_fpath_depth)

                    if frame_fpath_mask.exists() and not override:
                        logger.info(f"mask exists at {frame_fpath_mask}, skipping...")
                    else:
                        write_image(img=bbox_mask, path=frame_fpath_mask)

                    if frame_fpath_mask_amodal.exists() and not override:
                        logger.info(
                            f"mask amodal exists at {frame_fpath_mask_amodal}, skipping..."
                        )
                    else:
                        write_image(img=bbox_mask_amodal, path=frame_fpath_mask_amodal)

                    if frame_fpath_pxl_cat_id.exists() and not override:
                        logger.info(
                            f"pxl_cat_id exists at {frame_fpath_pxl_cat_id}, skipping..."
                        )
                    else:
                        from od3d.datasets.omni6dpose.enum import (
                            MAP_CATEGORIES_OMNI6DPOSE_TO_ID,
                        )

                        # dataset_specific_id = MAP_CATEGORIES_OMNI6DPOSE_TO_ID[batch.categories[0][bbox_id]]
                        # pxl_cat_id[pxl_cat_id != dataset_specific_id] = 0
                        dataset_specific_id = MAP_CATEGORIES_OMNI6DPOSE_TO_ID[
                            frame.meta.categories[bbox_id]
                        ]
                        write_image(
                            img=(dataset_specific_id * (bbox_mask > 0)).to(
                                dtype=torch.uint8
                            ),
                            path=frame_fpath_pxl_cat_id,
                        )
                        # write_image(img=pxl_cat_id.to(dtype=torch.uint8), path=frame_fpath_pxl_cat_id)

                    fpaths_meshs = frame.get_fpaths_meshs()[bbox_id : bbox_id + 1]
                    frame_meta = Omni6DPose_FrameMeta(
                        name=frame_name,
                        sequence_name=sequence_name,
                        subset=subset,
                        rfpath_rgb=frame_fpath_rgb.relative_to(
                            self.path_preprocess
                        ),  # frame.meta.rfpath_rgb,
                        rfpath_pxl_cat_id=frame_fpath_pxl_cat_id.relative_to(
                            self.path_preprocess
                        ),
                        rfpath_depth=frame_fpath_depth.relative_to(
                            self.path_preprocess
                        ),
                        rfpath_depth_mask=frame_fpath_depth_mask.relative_to(
                            self.path_preprocess
                        ),
                        rfpaths_meshs=[
                            fpath.relative_to(self.path_raw) for fpath in fpaths_meshs
                        ],
                        l_size=l_size,
                        categories=frame.meta.categories[bbox_id : bbox_id + 1],
                        l_cam_intr4x4=cam_intr4x4.tolist(),
                        l_cam_tform4x4_obj=frame.meta.l_cam_tform4x4_obj,
                        l_obj_tform4x4_objs=frame.meta.l_obj_tform4x4_objs[
                            bbox_id : bbox_id + 1
                        ],
                        l_objs_name=frame.meta.l_objs_name[bbox_id : bbox_id + 1],
                        l_objs_valid=frame.meta.l_objs_valid[bbox_id : bbox_id + 1],
                    )

                    if (
                        frame_meta.get_fpath(path_meta=self.path_meta).exists()
                        and not override
                    ):
                        logger.info(
                            f"already extracted frame {frame_name} in scene {sequence_name} in subset {subset}"
                        )
                    else:
                        frame_meta.save(path_meta=self.path_meta)

                    if bbox_id == len(bboxs) - 1:
                        seq_meta = Omni6DPose_SequenceMeta(
                            name=sequence_name, subset=subset
                        )
                        if (
                            seq_meta.get_fpath(path_meta=self.path_meta).exists()
                            and not override
                        ):
                            logger.info(
                                f"already extracted sequence {sequence_name} in subset {subset}"
                            )
                        else:
                            seq_meta.save(path_meta=self.path_meta)

        self.mesh_type = mesh_type

    @staticmethod
    def extract_meta(config: DictConfig):
        path_raw = Path(config.path_raw)
        path_meta = Omni6DPose.get_path_meta(config=config)

        if config.extract_meta.remove_previous:
            if path_meta.exists():
                shutil.rmtree(path_meta)

        """
        TEST_REAL = "test_real"
        TEST_IKEA = "test_ikea"
        TEST_MATTERPORT3D = "test_matterport3d"
        TEST_SCANNETPP = "test_scannetpp"
        TRAIN_IKEA = "train_ikea" # "ikea", "matterport3d", "scannet++",
        TRAIN_MATTERPORT3D = "train_matterport3d"
        TRAIN_SCANNETPP = "train_scannetpp"
        """

        subsets = config.get("dict_nested_frames", {}).keys()
        if len(subsets) == 0:
            subsets = list(OMNI6DPOSE_SUBSETS)

        scenes_paths = {}
        scenes_names = {}
        sope_patch_ids = [path.stem for path in path_raw.joinpath("SOPE").iterdir()]
        for subset in subsets:
            logger.info(f"extracting meta for subset {subset}")
            if subset == OMNI6DPOSE_SUBSETS.TEST_REAL:
                scenes_paths[subset] = list(path_raw.joinpath("ROPE").iterdir())
                scenes_names[subset] = [
                    scene_path.stem for scene_path in scenes_paths[subset]
                ]
            elif (
                subset == OMNI6DPOSE_SUBSETS.TEST_IKEA
                or subset == OMNI6DPOSE_SUBSETS.TEST_MATTERPORT3D
                or subset == OMNI6DPOSE_SUBSETS.TEST_SCANNETPP
                or subset == OMNI6DPOSE_SUBSETS.TRAIN_IKEA
                or subset == OMNI6DPOSE_SUBSETS.TRAIN_MATTERPORT3D
                or subset == OMNI6DPOSE_SUBSETS.TRAIN_SCANNETPP
            ):
                sope_subsets_to_rpath = {
                    OMNI6DPOSE_SUBSETS.TEST_IKEA: "test/ikea",
                    OMNI6DPOSE_SUBSETS.TEST_MATTERPORT3D: "test/matterport3d",
                    OMNI6DPOSE_SUBSETS.TEST_SCANNETPP: "test/scannet++",
                    OMNI6DPOSE_SUBSETS.TRAIN_IKEA: "train/ikea",
                    OMNI6DPOSE_SUBSETS.TRAIN_MATTERPORT3D: "train/matterport3d",
                    OMNI6DPOSE_SUBSETS.TRAIN_SCANNETPP: "train/scannet++",
                }
                subset_rpath = sope_subsets_to_rpath[subset]
                scenes_paths[subset] = []
                scenes_names[subset] = []
                for patch_id in sope_patch_ids:
                    path_patch_subset = path_raw.joinpath(
                        "SOPE", patch_id, subset_rpath
                    )
                    if path_patch_subset.exists():
                        scenes_paths[subset] += list(path_patch_subset.iterdir())
                        scenes_names[subset] += [
                            f"{patch_id}_{scene_path.stem}"
                            for scene_path in scenes_paths[subset]
                        ]

            for s, scene_path in enumerate(tqdm(scenes_paths[subset])):
                frames_ids = [
                    fpath.stem.split("_")[0]
                    for fpath in filter(
                        lambda p: Path(p).stem.endswith("_color"), scene_path.iterdir()
                    )
                ]

                seq_meta = Omni6DPose_SequenceMeta(
                    name=scenes_names[subset][s], subset=subset
                )
                if (
                    seq_meta.get_fpath(path_meta=path_meta).exists()
                    and not config.extract_meta.override
                ):
                    logger.info(
                        f"already extracted sequence {scenes_names[subset][s]} in subset {subset}"
                    )
                else:
                    seq_meta.save(path_meta=path_meta)

                # FRAME META
                # 'objects':
                #   OBJ1_NAME:
                #       'id'
                #       'meta':
                #           'class_name' (str)
                #           'class_label' (int)
                #           'instance_path' (str)
                #           'scale' (List[float]): 3 floats
                #           'bbox_side_len' (List[float]): 3 floats
                #           'is_background' (bool)
                #           'oid' (str): identifier for object e.g. omniobject3d-toy_truck_072
                #        'quaternion_wxyz' (List[float]): 4 floats
                #        'translation' (List[float]): 3 floats
                #        'world_quaternion_wxyz' (List[float]): 4 floats
                #        'world_translation' (List[float]): 3 floats
                #        'is_valid' (bool)
                #        'material' (Tuple(str, str)): e.g. ['specular', 'metal_13'], or ['raw', '']
                # 'camera':
                #   'quaternion' (List[float]): 4 floats
                #   'translation' (List[float]): 3 floats
                #   'intrinsics', (Dict[str, float]): 'fx', 'fy', 'cx', 'cy', 'width', 'height'
                #   'scene_obj_path' (str): path to scene object
                #   'background_image_path' (str): path to background image, e.g. 'data/ikea_data/table30/0003_color.png'
                #   'background_depth_path' (str): path to background depth image
                #   'distances' (List[float]):
                #   'kind' (str): dataset subset e.g. 'ikea'
                # 'env_param':
                #   'env_map_id' (int)
                #   'rotation_euler_z' (float)
                #   'image_value' (float)
                #   'image_path' (str): path to environment at /data/huangweiyao/OmniObjectPose/data_generation/assets/envmap_lib/test/
                # 'face_up' (bool)
                # 'concentrated' (bool)
                # 'comments' (str): e.g. 'code=v1/46, time="2024-02-04 15:56:12.885443", commit=4365326, description="dataset v1; 100w"'
                # 'runtime_seed' (int)
                # 'baseline_dis' (float)
                # 'emitter_dist_l' (float)
                # 'scene_dataset' (str): e.g. 'ikea'
                # from od3d.io import read_json
                # meta = read_json(scene_path.joinpath(f"0000_meta.json"))
                # rfpath_meshes = [obj['meta']['instance_path'] for obj in meta['objects'].values()]
                # obj_ids = [obj['id'] for obj in meta['objects'].values()]

                for frame_id in tqdm(frames_ids):
                    from od3d.io import read_json

                    meta = read_json(scene_path.joinpath(f"{frame_id}_meta.json"))
                    from od3d.cv.geometry.transform import (
                        cam_intr_to_4x4,
                        transf4x4_from_rot3x3_and_transl3,
                        cam_intr4x4_downsample,
                    )

                    fpath_rgb = scene_path.joinpath(f"{frame_id}_color.png")
                    fpath_mask = scene_path.joinpath(f"{frame_id}_mask.exr")
                    fpath_depth = scene_path.joinpath(f"{frame_id}_depth.exr")
                    fpath_depth_mask = scene_path.joinpath(f"{frame_id}_depth.exr")

                    img_size = torch.Tensor(
                        [
                            meta["camera"]["intrinsics"]["height"],
                            meta["camera"]["intrinsics"]["width"],
                        ],
                    )

                    # note: this downsampling is required due to non-consistency in Omni6DPose, 5.7.25
                    #       -> at least for
                    #               matterport3d train/test subset
                    #               scannetpp train/test subset
                    #               real

                    from od3d.cv.io import read_image, read_image_exr

                    # mask = read_image_exr(fpath_mask)
                    rgb = read_image(fpath_rgb)

                    downsamplerate_H = (
                        meta["camera"]["intrinsics"]["height"] / rgb.shape[-2]
                    )
                    downsamplerate_W = (
                        meta["camera"]["intrinsics"]["width"] / rgb.shape[-1]
                    )

                    if downsamplerate_H != downsamplerate_W:
                        msg = f"rgb downsample for {fpath_rgb} is not equal for width and height"
                        raise ValueError(msg)

                    downsamplerate = downsamplerate_W
                    cam_intr4x4 = cam_intr_to_4x4(
                        fx=meta["camera"]["intrinsics"]["fx"],
                        fy=meta["camera"]["intrinsics"]["fy"],
                        cx=meta["camera"]["intrinsics"]["cx"],
                        cy=meta["camera"]["intrinsics"]["cy"],
                    )

                    cam_intr4x4, img_size = cam_intr4x4_downsample(
                        cams_intr4x4=cam_intr4x4,
                        imgs_sizes=img_size,
                        down_sample_rate=downsamplerate,
                    )

                    l_size = img_size.tolist()
                    l_cam_intr4x4 = cam_intr4x4.tolist()
                    from od3d.cv.geometry.transform import (
                        transf4x4_from_rot4_and_transl3,
                        inv_tform4x4,
                        tform4x4_broadcast,
                    )

                    world_tform4x4_cam = transf4x4_from_rot4_and_transl3(
                        rot4=meta["camera"]["quaternion"],
                        transl3=meta["camera"]["translation"],
                    )
                    cam_tform4x4_world = inv_tform4x4(world_tform4x4_cam)
                    # world_tform4x4_objs = torch.stack([transf4x4_from_rot4_and_transl3(
                    #    rot4=obj['world_quaternion_wxyz'], transl3=obj['world_translation'])
                    #    for obj in meta['objects'].values()], dim=0)
                    cam_tform4x4_objs = torch.stack(
                        [
                            transf4x4_from_rot4_and_transl3(
                                rot4=obj["quaternion_wxyz"],
                                transl3=obj["translation"],
                            )
                            for obj in meta["objects"].values()
                        ],
                        dim=0,
                    )

                    rfpath_scene_mesh = meta["camera"]["scene_obj_path"]
                    if subset == OMNI6DPOSE_SUBSETS.TEST_REAL:
                        rfpath_meshes = [
                            Path("PAM/object_meshes").joinpath(
                                obj["meta"]["oid"],
                                Path(obj["meta"]["instance_path"]).name,
                            )
                            for obj in meta["objects"].values()
                        ]
                    else:
                        rfpath_meshes = [
                            obj["meta"]["instance_path"]
                            for obj in meta["objects"].values()
                        ]
                        rfpath_meshes = ["PAM" + rfpath[4:] for rfpath in rfpath_meshes]

                    fpath_meshes = [
                        path_raw.joinpath(rfpath_mesh) for rfpath_mesh in rfpath_meshes
                    ]

                    objs_scale = torch.FloatTensor(
                        [obj["meta"]["scale"] for obj in meta["objects"].values()]
                    )

                    if subset == OMNI6DPOSE_SUBSETS.TEST_REAL:
                        # import trimesh
                        # objs_scale_real = []
                        # for fpath in fpath_meshes:
                        #     _mesh_trimesh = trimesh.load(fpath, force='mesh')
                        #     _mesh_trimesh_verts = torch.Tensor(_mesh_trimesh.vertices)
                        #     obj_scale = max(_mesh_trimesh_verts.max(dim=0)[0] - _mesh_trimesh_verts.min(dim=0)[0])[None,]
                        #     objs_scale_real.append(obj_scale)
                        # objs_scale_real = torch.stack(objs_scale_real)
                        # objs_scale = objs_scale_real * 9.
                        objs_scale[:] = 1.0
                    cam_tform4x4_objs[..., :3, :3] *= objs_scale[:, :, None]
                    world_tform4x4_objs = tform4x4_broadcast(
                        world_tform4x4_cam[None,], cam_tform4x4_objs
                    )
                    # cam_tform4x4_objs = tform4x4_broadcast(cam_tform4x4_world[None,], world_tform4x4_objs)
                    # cam_tform4x4_objs = torch.stack([transf4x4_from_rot4_and_transl3(rot4=obj['quaternion_wxyz'],
                    #                                                     transl3=obj['translation'])
                    #                       for obj in meta['objects'].values()], dim=0)

                    categories = [
                        obj["meta"]["class_name"] for obj in meta["objects"].values()
                    ]
                    # objs_visible = torch.BoolTensor([obj['is_valid'] for obj in meta['objects'].values()])
                    # objs_visible[:] = True
                    # objs_background = torch.BoolTensor(
                    #    [obj['meta']['is_background'] for obj in meta['objects'].values()])
                    # objs_visible_and_not_background = objs_visible * (~objs_background)
                    # categories = [cat for i, cat in enumerate(categories)]
                    # rfpath_meshes = [rfpath_mesh for i, rfpath_mesh in enumerate(rfpath_meshes)]

                    obj_tform4x4_objs = world_tform4x4_objs

                    l_cam_tform4x4_obj = cam_tform4x4_world.tolist()
                    l_cam_tform4x4_objs = cam_tform4x4_objs.tolist()
                    l_obj_tform4x4_objs = obj_tform4x4_objs.tolist()
                    l_objs_name = [
                        obj["meta"]["oid"] for obj in meta["objects"].values()
                    ]
                    l_objs_valid = torch.BoolTensor(
                        [obj["is_valid"] for obj in meta["objects"].values()]
                    ).tolist()

                    from od3d.cv.visual.show import (
                        show_scene,
                        show_img,
                        render_trimesh_to_tensor,
                        show_imgs,
                    )
                    from od3d.cv.geometry.objects3d.meshes.meshes import (
                        Meshes,
                        PROJECT_MODALITIES,
                        VERT_MODALITIES,
                    )

                    # from od3d.cv.io import read_image
                    # rgb = read_image(path_raw.joinpath(rfpath_rgb))
                    # fpath_meshes = [path_raw.joinpath(rfpath_mesh) for rfpath_mesh in rfpath_meshes]

                    # fpath_meshes = fpath_meshes[:18]
                    # meshes = Meshes.read_from_ply_files(fpaths_meshes=fpath_meshes, device='cuda')

                    # objs_ids = torch.LongTensor([[-1, 3, 4], [3, 2, -1]]).cuda()
                    # objs_ids = torch.arange(len(meshes))[None,].cuda()
                    # obj_tform4x4_objs = obj_tform4x4_objs.cuda()[objs_ids]

                    # show_scene(meshes=meshes, cams_intr4x4=[cam_intr4x4], cams_tform4x4_world=[cam_tform4x4_world])
                    # ncds = meshes.render(cams_tform4x4_obj=cam_tform4x4_world[None,].cuda().repeat(1, 1, 1),
                    #                      cams_intr4x4=cam_intr4x4[None,].cuda(),
                    #                      imgs_sizes=torch.Tensor(l_size),
                    #                      objects_ids=objs_ids,
                    #                      modalities=PROJECT_MODALITIES.RGB,
                    #                      obj_tform4x4_objs=obj_tform4x4_objs)
                    # show_imgs([rgb.cuda() / 255., ncds[0]]) # .repeat(3, 1, 1)
                    # verts_stacked = meshes.get_verts_stacked_mask_with_mesh_ids()
                    # import trimesh
                    # mesh_trimesh = trimesh.load(fpath_meshes[0], force='mesh')
                    # rgb = render_trimesh_to_tensor(mesh_trimesh=mesh_trimesh, cam_intr4x4=cam_intr4x4, cam_tform4x4_obj=cam_tform4x4_world)
                    # simple operations objects3d, tform4x4, fuse_to_single_mesh
                    # show_img(rgb)
                    # meshes.render_batch(cams_tform4x4_obj=, cams_intr4x4=)
                    # show_scene(meshes=meshes)
                    # meshes.verts = transf3d_broadcast(pts3d=meshes.verts, transf4x4=world_tform4x4_objs)
                    # show_scene(meshes=meshes)
                    # 'objects', 'camera', 'env_param', 'face_up', 'concentrated', 'comments', 'runtime_seed', 'baseline_dis', 'emitter_dist_l', 'scene_dataset'
                    # q1: what makes a scene a scene?
                    #   -> same objects
                    # q2: why is scene object changing from image to image?
                    #   -> because multiple background allowed per scene
                    frame_meta = Omni6DPose_FrameMeta(
                        name=frame_id,
                        sequence_name=scenes_names[subset][s],
                        subset=subset,
                        rfpath_rgb=fpath_rgb.relative_to(path_raw),
                        rfpath_pxl_cat_id=fpath_mask.relative_to(path_raw),
                        rfpath_depth=fpath_depth.relative_to(path_raw),
                        rfpath_depth_mask=fpath_depth_mask.relative_to(path_raw),
                        rfpaths_meshs=rfpath_meshes,
                        l_size=l_size,
                        categories=categories,
                        l_cam_intr4x4=l_cam_intr4x4,
                        l_cam_tform4x4_obj=l_cam_tform4x4_obj,
                        l_obj_tform4x4_objs=l_obj_tform4x4_objs,
                        l_objs_name=l_objs_name,
                        l_objs_valid=l_objs_valid,
                    )

                    if (
                        frame_meta.get_fpath(path_meta=path_meta).exists()
                        and not config.extract_meta.override
                    ):
                        logger.info(
                            f"already extracted frame {frame_id} in scene {scenes_names[subset][s]} in subset {subset}"
                        )
                    else:
                        frame_meta.save(path_meta=path_meta)

            if subset not in [
                OMNI6DPOSE_SUBSETS.TEST_REAL,
                OMNI6DPOSE_SUBSETS.TEST_IKEA,
                OMNI6DPOSE_SUBSETS.TEST_MATTERPORT3D,
                OMNI6DPOSE_SUBSETS.TEST_SCANNETPP,
                OMNI6DPOSE_SUBSETS.TRAIN_IKEA,
                OMNI6DPOSE_SUBSETS.TRAIN_MATTERPORT3D,
                OMNI6DPOSE_SUBSETS.TRAIN_SCANNETPP,
            ]:
                # else:
                # TRAIN_SYN_ALBEDO_ICO_UNI64_THETA_UNI1
                # TEST_SYN_ALBEDO_ICO_UNI64_THETA_UNI1
                # TEST_REAL_ALBEDO_ICO_UNI64_THETA_UNI1
                # train_syn_albedo_ico_uni64_theta_uni1
                # test_syn_albedo_ico_uni64_theta_uni1
                # test_real_albedo_ico_uni64_theta_uni1

                # what about category
                categories = Omni6DPose.get_class_specific_categories(
                    config.get("categories", None)
                )
                import re

                match = re.match(
                    r"([a-z]+)_([a-z]+)_([a-z]+)_([a-z0-9]+)_([a-z]+)([0-9]+)_theta_([a-z]+)([0-9]+)",
                    subset,
                    re.I,
                )
                if match and len(match.groups()) == 8:
                    (
                        train_or_test,
                        real_or_syn,
                        render_type,
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
                else:
                    msg = f"could not retrieve train_or_test, real_or_syn, render_type, viewpoints_type, viewpoints_sample_type, viewpoints_count, theta_sample_type, theta_count from subset {subset}"
                    raise Exception(msg)

                from od3d.io import read_json

                # # real_obj_meta
                # Path("/data/lmbraid21/jesslen/Omni6DPose/Meta/real_obj_meta.json")
                # Path("/data/lmbraid21/jesslen/Omni6DPose/Meta/obj_meta.json")

                if real_or_syn == "real":
                    if train_or_test == "train":
                        msg = f"no real assets available for training."
                        raise Exception(msg)
                    # ['class_list']['instance_dict']
                    assets_meta = read_json(
                        path_raw.joinpath("Meta/real_obj_meta.json")
                    )["instance_dict"]
                else:
                    # ['class_list']['instance_dict']
                    assets_meta = read_json(path_raw.joinpath("Meta/obj_meta.json"))[
                        "instance_dict"
                    ]
                    assets_meta = {
                        key: val
                        for key, val in assets_meta.items()
                        if val["tag"]["datatype"] == train_or_test
                    }

                # box, shoe, doll, teapot, handbag, dinosaur, mug, flower_pot

                # syn['instance_dict']['omniobject3d-whistle_025']
                # 'object_id': e.g. 'omniobject3d-whistle_025'
                # 'obj_path': e.g. 'whistle/whistle_005/Scan/Aligned.obj'
                # 'source': e.g. 'omniobject3d'
                # 'name': e.g. 'whistle_025'
                # 'class_label': e.g. 153
                # 'class_name': e.g. 'whistle'
                # 'dimensions': e.g. [30.647, 58.113, 30.219]
                # 'tag':
                #     'datatype': e.g. "train"
                #     'materialOptions': e.g. ["raw", "diffuse"]

                for category in categories:
                    # filter category
                    assets_meta_cat = {
                        key: val
                        for key, val in assets_meta.items()
                        if val["class_name"] == category
                    }

                    sequences_count_max_per_category = config.get(
                        "sequences_count_max_per_category", None
                    )
                    if sequences_count_max_per_category is not None:
                        assets_meta_cat = {
                            k: assets_meta_cat[k]
                            for k in list(assets_meta_cat)[
                                :sequences_count_max_per_category
                            ]
                        }

                    for asset_name, asset_data in assets_meta_cat.items():
                        sequence_name = f"{category}_{asset_data['object_id']}"
                        sequence_meta = Omni6DPose_SequenceMeta(
                            subset=subset,
                            name=sequence_name,
                        )

                        if (
                            sequence_meta.get_fpath(path_meta=path_meta).exists()
                            and not config.extract_meta.override
                        ):
                            logger.info(f"already extracted sequence {sequence_name}")
                            continue
                        # # /data/lmbraid21/jesslen/Omni6DPose/PAM/object_meshes/omniobject3d-whistle_025
                        fpath_mesh = path_raw.joinpath(
                            "PAM/object_meshes/"
                            + asset_data["object_id"]
                            + "/"
                            + Path(asset_data["obj_path"]).name
                        )
                        frame_fpath_mesh_orig = fpath_mesh
                        frame_fpath_mesh = fpath_mesh
                        cat_id = asset_data["class_label"]
                        obj_name = asset_data["object_id"]
                        if real_or_syn == "real":
                            import trimesh

                            _mesh_trimesh = trimesh.load(
                                frame_fpath_mesh_orig, force="mesh"
                            )
                            _mesh_trimesh_verts = torch.Tensor(_mesh_trimesh.vertices)
                            obj_scale = max(
                                _mesh_trimesh_verts.max(dim=0)[0]
                                - _mesh_trimesh_verts.min(dim=0)[0]
                            )
                        else:
                            obj_scale = max(asset_data["dimensions"])
                        # pip install "pyglet<2"
                        # import trimesh
                        # _mesh_trimesh = trimesh.load(frame_fpath_mesh_orig, force='mesh')
                        #
                        # if render_type == 'notxt':
                        #     import numpy as np
                        #     # Define a single grey color (RGBA format)
                        #     grey_color = np.array([64, 64, 64, 255], dtype=np.uint8)
                        #     # Apply the color to all faces/vertices
                        #     _mesh_trimesh.visual = trimesh.visual.color.ColorVisuals(_mesh_trimesh,
                        #                                                              vertex_colors=grey_color)
                        #
                        # logger.info(f"sequence {sequence_name}, {len(_mesh_trimesh.vertices)} vertices")
                        # #_mesh_trimesh.show()
                        #
                        # if len(_mesh_trimesh.vertices) > 0:
                        #     pass
                        #     # _ = _mesh_trimesh.export(file_obj=frame_fpath_mesh)
                        # else:
                        #     logger.info(
                        #         f"skip sequence {sequence_name} because could not convert to single mesh.")

                        from od3d.cv.geometry.transform import (
                            get_ico_cam_tform4x4_obj_for_viewpoints_count,
                            get_ico_traj_cam_tform4x4_obj_for_viewpoints_count,
                            get_cam_tform4x4_obj_for_viewpoints_count,
                            tform4x4_broadcast,
                        )
                        from od3d.cv.visual.show import (
                            get_default_camera_intrinsics_from_img_size,
                        )
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
                        import math

                        # random_sampling(pts3d_cls, pts3d_max_count, return_mask=False)
                        # pts3d_ids, pts3d = fps(pts3d=pts3d, K=1024, fill=True)
                        W = 512
                        H = 512
                        device = "cuda"

                        imgs_sizes = torch.LongTensor([W, H])

                        if viewpoints_type.startswith(
                            "icotraj"
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
                                    radius=4.0 * obj_scale,
                                    theta_count=theta_count,
                                    geodesic_distance=traj_length,
                                    real=real,
                                ).to(device)
                            )
                        else:
                            cams_tform4x4_obj = (
                                get_ico_cam_tform4x4_obj_for_viewpoints_count(
                                    viewpoints_count=viewpoints_count,
                                    radius=4.0 * obj_scale,
                                    theta_count=theta_count,
                                    viewpoints_uniform=viewpoints_sample_uni,
                                    theta_uniform=theta_sample_uni,
                                ).to(device)
                            )

                        # from od3d.cv.visual.show import show_scene
                        # show_scene(cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=cam_intr4x4[None,].repeat(64, 1, 1))

                        # front, top, right, bottom
                        # cams_tform4x4_obj = get_cam_tform4x4_obj_for_viewpoints_count(viewpoints_count=3, dist=2.).to(device)

                        cams_tform4x4_obj = tform4x4_broadcast(
                            cams_tform4x4_obj, OBJ_TFORM_OBJ_SHAPENET.to(device)
                        )
                        cam_intr4x4 = get_default_camera_intrinsics_from_img_size(
                            W=W, H=H
                        ).to(device)
                        try:
                            mesh = Meshes.read_from_ply_file(
                                fpath=frame_fpath_mesh, device=device, load_texts=True
                            )
                        except Exception:
                            # print(frame_fpath_mesh)
                            logger.warning(f"could not load texture {frame_fpath_mesh}")
                            continue

                        logger.info(
                            f"sequence {sequence_name}, {len(mesh.verts)} vertices"
                        )
                        logger.info(f"{frame_fpath_mesh}")

                        # from od3d.cv.visual.show import show_scene
                        # show_scene(cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=cam_intr4x4[None,].repeat(64, 1, 1), meshes=mesh)

                        # mods = mesh.render(cams_tform4x4_obj=cams_tform4x4_obj, cams_intr4x4=cam_intr4x4,
                        #                    imgs_sizes=imgs_sizes, modalities=["mask"],
                        #                    broadcast_batch_and_cams=True, rgb_bg=[0., 0., 0.],
                        #                    rgb_diffusion_alpha=0.3)
                        # rgbs = mods['rgb'][0]
                        # masks = mods['mask'][0]

                        # from od3d.cv.visual.show import show_imgs
                        # show_imgs(rgbs)
                        pts3d = []
                        for i, cam_tform4x4_obj in enumerate(cams_tform4x4_obj):
                            frame_name = f"{int(i):03}"
                            frame_fpath_rgb = path_raw.joinpath(
                                f"rgb/{subset}/{sequence_name}/{frame_name}.png",
                            )
                            # frame_fpath_mask = path_raw.joinpath(
                            #    f"mask/{subset}/{sequence_name}/{frame_name}.png")
                            frame_fpath_pxl_cat_id = path_raw.joinpath(
                                f"pxl_cat_id/{subset}/{sequence_name}/{frame_name}.png",
                            )
                            frame_fpath_depth = path_raw.joinpath(
                                f"depth/{subset}/{sequence_name}/{frame_name}.png",
                            )
                            frame_fpath_depth_mask = path_raw.joinpath(
                                f"depth_mask/{subset}/{sequence_name}/{frame_name}.png",
                            )

                            # cam_tform4x4_obj_2 = cam_tform4x4_obj.clone()
                            # cam_tform4x4_obj_2[2, 3] = 1000.
                            # rgb, depth = render_trimesh_to_tensor(mesh_trimesh=_mesh_trimesh,
                            #                                       cam_tform4x4_obj=cam_tform4x4_obj,
                            #                                       cam_intr4x4=cam_intr4x4, H=H, W=W,
                            #                                       rgb_bg=[0.8, 0.8, 0.8], zfar=obj_scale * 10)
                            # depth_mask = depth > 0.
                            #
                            # _pts3d = \
                            # depth2pts3d_grid(depth=depth.to(device), cam_intr4x4=cam_intr4x4).permute(1, 2, 0)[
                            #     depth_mask[0]]
                            # _pts3d = random_sampling(pts3d_cls=_pts3d, pts3d_max_count=1024)
                            # _pts3d = transf3d_broadcast(pts3d=_pts3d, transf4x4=inv_tform4x4(cam_tform4x4_obj))
                            # pts3d.append(_pts3d)
                            # write_image(img=rgb, path=frame_fpath_rgb)
                            # write_image(img=masks[i], path=frame_fpath_mask)
                            # write_image(img=masks[i] * cat_id, path=frame_fpath_pxl_cat_id)
                            # write_depth_image(img=depth, path=frame_fpath_depth)
                            # write_mask_image(img=depth_mask, path=frame_fpath_depth_mask)

                            l_obj_tform4x4_objs = torch.eye(4)[None,].tolist()
                            l_objs_valid = (
                                torch.ones((1,)).to(dtype=torch.bool).tolist()
                            )
                            l_objs_name = [obj_name]
                            frame_meta = Omni6DPose_FrameMeta(
                                name=frame_name,  # frame_id,
                                sequence_name=sequence_name,  # scenes_names[subset][s],
                                subset=subset,
                                rfpath_rgb=frame_fpath_rgb.relative_to(path_raw),
                                rfpath_pxl_cat_id=frame_fpath_pxl_cat_id.relative_to(
                                    path_raw
                                ),
                                rfpath_depth=frame_fpath_depth.relative_to(path_raw),
                                rfpath_depth_mask=frame_fpath_depth_mask.relative_to(
                                    path_raw,
                                ),
                                rfpaths_meshs=[frame_fpath_mesh.relative_to(path_raw)],
                                l_size=imgs_sizes.tolist(),
                                categories=[category],
                                l_cam_intr4x4=cam_intr4x4.tolist(),
                                l_cam_tform4x4_obj=cam_tform4x4_obj.tolist(),
                                l_obj_tform4x4_objs=l_obj_tform4x4_objs,
                                l_objs_name=l_objs_name,
                                l_objs_valid=l_objs_valid,
                            )

                            # frame_meta = ShapeNet_FrameMeta(name=frame_name,
                            #                                 sequence_name=sequence_id,
                            #                                 rfpath_rgb=frame_fpath_rgb.relative_to(path_raw),
                            #                                 rfpath_mask=frame_fpath_mask.relative_to(path_raw),
                            #                                 rfpath_depth=frame_fpath_depth.relative_to(path_raw),
                            #                                 rfpath_depth_mask=frame_fpath_depth_mask.relative_to(
                            #                                     path_raw),
                            #                                 # rfpath_mesh=frame_fpath_mesh.relative_to(path_raw),
                            #                                 category=category,
                            #                                 l_size=imgs_sizes.tolist(),
                            #                                 l_cam_tform4x4_obj=cam_tform4x4_obj.tolist(),
                            #                                 l_cam_intr4x4=cam_intr4x4.tolist(),
                            #                                 subset=subset, )

                            if frame_meta is not None:
                                frame_meta.save(path_meta=path_meta)

                        # pts3d = torch.cat(pts3d, dim=0)
                        # pts3d_ids, pts3d = fps(pts3d=pts3d, K=1024, fill=True)
                        # write_pts3d_with_colors_and_normals(pts3d, pts3d_colors=None, pts3d_normals=None,
                        #                                    fpath=frame_fpath_pcl)

                        if sequence_meta is not None:
                            sequence_meta.save(path_meta=path_meta)
                        # sequences_count = sequences_count + 1

                    # fpaths_color = filter(lambda p: Path(p).stem.endswith("_color"), scene_path.iterdir())

        # dict_nested_frames = config.get("dict_nested_frames", None)

        # if dict_nested_frames is None:
        # dict_nested_frames_banned = config.get("dict_nested_frames_ban", None)
        # preprocess_meta_override = config.get("extract_meta", False).get(
        #    "override",

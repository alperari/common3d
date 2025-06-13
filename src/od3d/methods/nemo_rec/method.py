import logging
import time

import numpy as np
import od3d.io
import pandas as pd
from od3d.benchmark.results import OD3D_Results
from od3d.cv.metric.pose import get_pose_diff_in_rad
from od3d.datasets.dataset import OD3D_Dataset
from od3d.methods.method import OD3D_Method
from omegaconf import DictConfig
from od3d.cv.select import batched_index_select
from od3d.datasets.frames import OD3D_Frames

from od3d.tasks.reconstruction import Reconstruction
from od3d.tasks.task import OD3D_Visuals, OD3D_Metrics

logger = logging.getLogger(__name__)
import torch
from od3d.cv.geometry.metrics.dist import batch_chamfer_distance, batch_point_face_distance, batch_point_face_distance_v2
from od3d.cv.geometry.transform import transf3d_broadcast
from od3d.datasets.frame import OD3D_FRAME_MODALITIES
from od3d.cv.geometry.transform import se3_exp_map, proj3d2d_broadcast
from torch.nn.utils.rnn import pad_sequence
from od3d.cv.visual.show import imgs_to_img, show_scene2d
from od3d.cv.geometry.objects3d.meshes import Meshes
from pathlib import Path
from od3d.cv.geometry.transform import (
    transf4x4_from_spherical,
    tform4x4,
    inv_tform4x4,
    tform4x4_broadcast,
)
from od3d.cv.visual.show import show_img
from od3d.cv.visual.show import show_bar_chart
from od3d.cv.visual.blend import blend_rgb
from tqdm import tqdm
from od3d.cv.geometry.objects3d.objects3d import PROJECT_MODALITIES

import tempfile
import open3d
import wandb
import copy

# note: math is actually used by config
import math  # noqa
from od3d.datasets.co3d import CO3D
from od3d.datasets.spair71k import SPair71K

from od3d.cv.io import image_as_wandb_image
from od3d.cv.visual.resize import resize
from od3d.models.model import OD3D_Model

from od3d.cv.geometry.grid import get_pxl2d_like
from od3d.cv.geometry.fit3d2d import batchwise_fit_se3_to_corresp_3d_2d_and_masks
from od3d.cv.transforms.transform import OD3D_Transform
from od3d.cv.transforms.sequential import SequentialTransform

from typing import Dict
from od3d.data.ext_enum import StrEnum

import matplotlib.pyplot as plt

plt.switch_backend("Agg")
from od3d.cv.visual.show import get_img_from_plot
from od3d.cv.visual.draw import draw_text_in_rgb


class VISUAL_MODALITIES(StrEnum):
    PRED_VERTS_NCDS_IN_RGB = "pred_verts_ncds_in_rgb"
    GT_VERTS_NCDS_IN_RGB = "gt_verts_ncds_in_rgb"
    PRED_VS_GT_VERTS_NCDS_IN_RGB = "pred_vs_gt_verts_ncds_in_rgb"
    NET_FEATS_NEAREST_VERTS = "net_feats_nearest_verts"
    SIM_PXL = "sim_pxl"
    SAMPLES = "samples"
    TSNE = "tsne"
    PCA = "pca"
    RECONSTRUCTION_MAP = "reconstruction_map"
    TSNE_PER_IMAGE = "tsne_per_image"
    MESH = "mesh"
    LATENT_INTERP = "latent_interp"


class NeMo_Rec(OD3D_Method):
    def setup(self):
        pass

    def __init__(
        self,
        config: DictConfig,
        logging_dir,
    ):
        super().__init__(config=config, logging_dir=logging_dir)

        self.device = "cuda:0"

        # init Network
        torch.multiprocessing.set_sharing_strategy("file_system")

        self.transform_train = SequentialTransform(
            [
                OD3D_Transform.subclasses[
                    config.train.transform.class_name
                ].create_from_config(config=config.train.transform),
            ],
        )
        self.transform_test = SequentialTransform(
            [
                OD3D_Transform.subclasses[
                    config.test.transform.class_name
                ].create_from_config(config=config.test.transform),
            ],
        )

        # self.path_shapenemo = Path(config.path_shapenemo)
        # self.fpaths_meshes_shapenemo = [self.path_shapenemo.joinpath(cls, '01.off') for cls in config.categories]
        self.fpaths_meshes = [
            self.config.fpaths_meshes[cls] for cls in config.categories
        ]
        fpaths_meshes_tform_obj = self.config.get("fpaths_meshes_tform_obj", None)
        if fpaths_meshes_tform_obj is not None:
            self.fpaths_meshes_tform_obj = [
                fpaths_meshes_tform_obj[cls] for cls in config.categories
            ]
        else:
            self.fpaths_meshes_tform_obj = [None for _ in config.categories]

        from od3d.cv.geometry.objects3d.objects3d import OD3D_Objects3D

        self.use_pbr_rendering = True
        if self.use_pbr_rendering:
            from od3d.cv.render.utils.util import latlong_to_cubemap, dot
            # latlong_img = torch.tensor(load_image("data/light_envs/neutral.hdr"), dtype=torch.float32, device='cuda')
            latlong_img = torch.randn((512, 1024, 3)).to(dtype=torch.float32, device='cuda') * 0.
            # latlong_img = latlong_img.clamp(0.2, 1.)
            # 6 x H x W x 3
            cubemap = latlong_to_cubemap(latlong_img, [512, 512])

            self._rgb_light_env = torch.nn.Parameter(cubemap, requires_grad=True)
            self._rgb_light_env = self._rgb_light_env.cuda()

        else:
            self._rgb_light_env = None

        from od3d.cv.geometry.objects3d.objects3d import FEATS_ACTIVATION
        self.config.objects3d.update({
            #"verts_requires_grad": False,
            "feats_requires_grad": True,
            "feat_dim": 3 if not self.use_pbr_rendering else 6,
            "fpaths_meshes": self.fpaths_meshes,
            "fpaths_meshes_tforms": self.fpaths_meshes_tform_obj,
            "feats_objects": True,
            "feat_clutter": True,
            "feats_activation": FEATS_ACTIVATION.SIGMOID,
        }
        )

        config_objects3d_without_class_name = self.config.objects3d.copy()
        del config_objects3d_without_class_name.class_name

        self.meshes = OD3D_Objects3D.subclasses[self.config.objects3d.class_name].read_from_ply_files(
            **config_objects3d_without_class_name,
        )

        self.task = Reconstruction(metrics={OD3D_Metrics.REC_RGB_MSE: 1.,
                                            OD3D_Metrics.REC_MASK_MSE: 1.,
                                            OD3D_Metrics.REC_MASK_DT_DOT: 1.,
                                            OD3D_Metrics.REC_RGB_PSNR: 1.},
                                   visuals=[OD3D_Visuals.PRED_VS_GT_RGB],
                                   apply_mask_rgb_pred=False,
                                   apply_mask_rgb_gt=True)

        self.down_sample_rate = 4.

        logger.info(f"loading meshes from following fpaths: {self.fpaths_meshes}...")
        # self.meshes.show()
        self.verts_count_max = self.meshes.verts_counts_max
        self.mem_verts_feats_count = len(config.categories) * self.verts_count_max
        self.mem_clutter_feats_count = config.num_noise * config.max_group
        self.mem_count = self.mem_verts_feats_count + self.mem_clutter_feats_count

        self.feats_bank_count = self.verts_count_max * len(self.meshes) + 1

        # dict to save estimated tforms, sequence : tform,
        self.seq_obj_tform4x4_est_obj = {}
        self.seq_obj_tform4x4_est_obj_sim = {}

        self.total_params_mesh_clutter = sum(
            p.numel() for p in self.meshes.parameters()
        )
        self.trainable_params_mesh_clutter = sum(
            p.numel() for p in self.meshes.parameters() if p.requires_grad
        )

        self.meshes.cuda()
        self.meshes.eval()

        self.back_propagate = True

        logger.info(
            f"total params mesh and clutter: {self.total_params_mesh_clutter}, trainable params mesh and clutter: {self.trainable_params_mesh_clutter}",
        )

        if not self.use_pbr_rendering:
            self.optim = od3d.io.get_obj_from_config(
                config=self.config.train.optimizer,
                params=list(set(self.meshes.parameters())),
            )
        else:
            self.optim = od3d.io.get_obj_from_config(
                config=self.config.train.optimizer,
                params=list(set(self.meshes.parameters())) + [self._rgb_light_env],
            )

        self.scheduler = od3d.io.get_obj_from_config(
            self.optim,
            config=self.config.train.scheduler,
        )

        #import wandb
        # wandb.watch(self.meshes, log="all", log_freq=1)
        # wandb.watch(self.net, log="all", log_freq=1)

    @property
    def rgb_light_env(self):
        if self._rgb_light_env is None:
            return self._rgb_light_env
        else:
            return torch.sigmoid(self._rgb_light_env * 100.)

    def train(
        self,
        datasets_train: Dict[str, OD3D_Dataset],
        datasets_val: Dict[str, OD3D_Dataset],
    ):
        # prevents inheriting broken CUDA context for each worker
        #import torch.multiprocessing as mp
        #mp.set_start_method('spawn', force=True)

        score_metric_neg = self.config.train.early_stopping_score.startswith('-')
        score_metric_name = self.config.train.early_stopping_score  # "pose/acc_pi18"  # 'pose/acc_pi18' 'pose/acc_pi6'
        if score_metric_neg:
            score_metric_name = score_metric_name[1:]

        score_ckpt_val = -np.inf
        score_latest = -np.inf

        if "main" in datasets_val.keys():
            dataset_train_sub = datasets_train["labeled"]
        else:
            dataset_train_sub, dataset_val_sub = datasets_train["labeled"].get_split(
                fraction1=1.0 - self.config.train.val_fraction,
                fraction2=self.config.train.val_fraction,
                split=self.config.train.split,
            )
            datasets_val["main"] = dataset_val_sub

        # first validation
        if self.config.train.val:
            for dataset_val_key, dataset_val in datasets_val.items():
                with torch.no_grad():
                    results_val = self.test(dataset_val, val=True)
                results_val.log_with_prefix(prefix=f"val/{dataset_val.name}")
                if dataset_val_key == "main":
                    score_latest = results_val[score_metric_name]
                    if score_metric_neg:
                        score_latest = -score_latest
            if not self.config.train.early_stopping or score_latest >= score_ckpt_val:
                score_ckpt_val = score_latest

        for epoch in range(self.config.train.epochs):
            results_epoch = self.train_epoch(dataset=dataset_train_sub)
            results_epoch.log_with_prefix("train")
            if (
                    self.config.train.val
                    and self.config.train.epochs_to_next_test > 0
                    and epoch % self.config.train.epochs_to_next_test == 0
            ):
                for dataset_val_key, dataset_val in datasets_val.items():
                    results_val = self.test(dataset_val, val=True)
                    results_val.log_with_prefix(prefix=f"val/{dataset_val.name}")
                    if dataset_val_key == "main":
                        score_latest = results_val[score_metric_name]
                        if score_metric_neg:
                            score_latest = -score_latest

                if (
                        not self.config.train.early_stopping
                        or score_latest >= score_ckpt_val
                ):
                    score_ckpt_val = score_latest


    # objects3d: basic:
    #   a) render
    #   b) get_verts3d, get_normals3d
    #   c) get_deform_verts3d_with_img, def get_deform_verts3d_with_latent
    #   d) forward, get deform (with/without require grad for verts, with/without grad for deform) + render
    # everytime render is called extract verts? or is this separate? -> separate
    # surface probability, add probability background which is dependent on th e
    # how to add texture? everytime get_verts3d -> automatic extract texture map
    # batch mesh, cat mesh

    # 1. render modalities: rgb, mask, surface_probability
    # 2. define losses/tasks: rgb, mask, mask_dt, mask_inv_dt
    # 3. visualize modalities: rgb, mask, surface_probability


    def test(self, dataset: OD3D_Dataset, val=False, return_results_epoch=False):
        logger.info(f"test dataset {dataset.name}")
        self.meshes.eval()

        dataset.transform = copy.deepcopy(self.transform_test)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.test.dataloader.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            num_workers=self.config.test.dataloader.num_workers,
            pin_memory=self.config.test.dataloader.pin_memory,
        )
        logger.info(f"Dataset contains {len(dataset)} frames.")

        results_epoch = OD3D_Results(logging_dir=self.logging_dir)
        for i, batch in enumerate(iter(dataloader)):
            with torch.no_grad():
                results_batch: OD3D_Results = self.train_batch(batch=batch)
            results_epoch += results_batch

        count_pred_frames = len(results_epoch["item_id"])
        logger.info(f"Predicted {count_pred_frames} frames.")

        results_visual = self.get_results_visual(
            results_epoch=results_epoch,
            dataset=dataset,
            config_visualize=self.config.test.visualize,
        )

        results_epoch_mean = results_epoch.mean()
        results_epoch_mean += results_visual

        if return_results_epoch:
            return results_epoch_mean, results_epoch
        else:
            return results_epoch_mean

    def train_epoch(self, dataset: OD3D_Dataset) -> OD3D_Results:
        logger.info(f"train dataset {dataset.name}")
        self.meshes.del_pre_rendered()
        self.meshes.train()

        self.optim.zero_grad()
        dataset.transform = copy.deepcopy(self.transform_train)
        dataloader_train = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.train.dataloader.batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
            num_workers=self.config.train.dataloader.num_workers,
            pin_memory=self.config.train.dataloader.pin_memory,
        )

        results_epoch = OD3D_Results(logging_dir=self.logging_dir)
        accumulate_steps = 0
        for i, batch in enumerate(iter(dataloader_train)):
            results_batch: OD3D_Results = self.train_batch(batch=batch)
            results_batch.log_with_prefix("train")
            accumulate_steps += 1
            if accumulate_steps % self.config.train.batch_accumulate_to_next_step == 0:
                if self.back_propagate:
                    self.optim.step()
                    self.optim.zero_grad()

            results_epoch += results_batch

        self.scheduler.step()
        self.optim.zero_grad()

        results_visual = self.get_results_visual(
            results_epoch=results_epoch,
            dataset=dataset,
            config_visualize=self.config.train.visualize,
        )
        results_epoch = results_epoch.mean()
        results_epoch += results_visual
        return results_epoch

    def train_batch(self, batch) -> OD3D_Results:
        results_batch = OD3D_Results(logging_dir=self.logging_dir)
        B = len(batch)

        logger.info(f"categories before cuda {batch.category_id}")

        batch.to(device=self.device)

        if not self.use_pbr_rendering:


            mods = self.meshes.update_and_render_with_batch_and_imgs_feats(batch,
                                                                           modalities=[PROJECT_MODALITIES.FEATS,
                                                                                       PROJECT_MODALITIES.MASK],
                                                                           add_clutter=True,
                                                                           down_sample_rate=self.down_sample_rate,
                                                                           broadcast_batch_and_cams=True)
            rgb_pred = mods[PROJECT_MODALITIES.FEATS][0]


        else:
            mods = self.meshes.update_and_render_with_batch_and_imgs_feats(batch,
                                                                           modalities=[PROJECT_MODALITIES.FEATS_PBR,
                                                                                       PROJECT_MODALITIES.MASK],
                                                                           down_sample_rate=self.down_sample_rate,
                                                                           add_clutter=True,
                                                                           broadcast_batch_and_cams=True,
                                                                           rgb_light_env=self.rgb_light_env)


            rgb_pred = mods[PROJECT_MODALITIES.FEATS_PBR][0]

        #from od3d.cv.visual.show import show_imgs
        #show_imgs(rgb_pred)

        mask_pred = mods[PROJECT_MODALITIES.MASK][0]
        frames_pred = OD3D_Frames(rgb=rgb_pred, mask=mask_pred,
                                  modalities=[OD3D_FRAME_MODALITIES.RGB, OD3D_FRAME_MODALITIES.MASK], length=len(batch),
                                  name=batch.name, name_unique=batch.name_unique, item_id=batch.item_id)

        task_metrics = self.task.eval(frames_gt=batch, frames_pred=frames_pred)

        loss_geo_sdf_reg = self.meshes.get_geo_sdf_reg_loss(objects_ids=batch.category_id)

        losses_names = ['rec_rgb_mse', 'rec_mask_mse', 'rec_mask_dt_dot', 'sdf_reg']
        loss_batch = task_metrics.rec_rgb_mse + task_metrics.rec_mask_mse  + task_metrics.rec_mask_dt_dot + loss_geo_sdf_reg
        losses = [task_metrics.rec_rgb_mse.mean() * 1., task_metrics.rec_mask_mse.mean() * 10.,
                  -task_metrics.rec_mask_dt_dot.mean() * 100., loss_geo_sdf_reg.mean() * 0.01]

        if len(losses) > 0:
            for l in range(len(losses)):
                if losses[l].isnan().any():
                    logger.warning(f'Loss {losses_names[l]} contains NaNs. Setting it to zero.')
                    losses[l][:] = 0.
                results_batch[losses_names[l]] = losses[l].detach()

            loss = torch.stack(losses).sum(dim=0)
            loss = loss / self.config.train.batch_accumulate_to_next_step
            if self.back_propagate and loss.requires_grad:
                logger.info("backward pass")
                loss.backward()
        else:
            loss = torch.zeros([]).to(device=self.device)

        logger.info(f"loss {loss.item()}")

        results_batch["psnr"] = task_metrics.rec_rgb_psnr.detach()

        results_batch["loss"] = loss[None,].detach()
        results_batch["loss_batch"] = loss_batch.detach()
        results_batch["rgb"] = frames_pred.rgb.detach() # note: this needs to be detached
        results_batch["item_id"] = batch.item_id
        results_batch["name_unique"] = batch.name_unique

        return results_batch

    def get_results_visual_batch(
        self,
        batch,
        results_batch: OD3D_Results,
        config_visualize: DictConfig,
        dict_name_unique_to_sel_name=None,
        dict_name_unique_to_result_id=None,
        caption_metrics=["sim", "rot_diff_rad"],
    ):
        results_batch_visual = OD3D_Results(logging_dir=self.logging_dir)
        modalities = config_visualize.modalities
        if len(modalities) == 0:
            return results_batch_visual

        down_sample_rate = config_visualize.down_sample_rate
        samples_sorted = config_visualize.samples_sorted
        samples_scores = config_visualize.samples_scores
        live = config_visualize.live

        with torch.no_grad():
            batch.to(device=self.device)
            B = len(batch)
            if dict_name_unique_to_result_id is not None:
                batch_result_ids = torch.LongTensor(
                    [
                        dict_name_unique_to_result_id[batch.name_unique[b]]
                        for b in range(B)
                    ],
                ).to(device=self.device)
            else:
                batch_result_ids = torch.LongTensor(range(B)).to(device=self.device)

            if dict_name_unique_to_sel_name is not None:
                batch_sel_names = [
                    dict_name_unique_to_sel_name[batch.name_unique[b]] for b in range(B)
                ]
            else:
                batch_sel_names = [batch.name_unique[b] for b in range(B)]

            batch_names = [batch.name_unique[b] for b in range(B)]
            batch_sel_scores = []
            for b in range(B):
                batch_sel_scores.append(
                    "\n".join(
                        [
                            f"{metric}={results_batch[metric].to(device=self.device)[batch_result_ids[b]].cpu().detach().item():.3f}"
                            for metric in caption_metrics
                            if metric in results_batch.keys()
                        ],
                    ),
                )

            frames_pred = OD3D_Frames(rgb=results_batch["rgb"].to(self.device)[batch_result_ids],
                                      modalities=[OD3D_FRAME_MODALITIES.RGB], length=len(batch),
                                      name=batch.name, name_unique=batch.name_unique, item_id=batch.item_id)

            self.task.visualize(frames_gt=batch, frames_pred=frames_pred)
            B = len(frames_pred.pred_vs_gt_rgb)

            for b in range(B):
                results_batch_visual[
                    f"visual/{VISUAL_MODALITIES.PRED_VS_GT_VERTS_NCDS_IN_RGB}/{batch_sel_names[b]}"
                ] = image_as_wandb_image(
                    frames_pred.pred_vs_gt_rgb[b],
                    caption=f"{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}",
                )

            if self.rgb_light_env is not None: # 6 x H x W x 3
                results_batch_visual["visual/rgb_light_env"] = image_as_wandb_image(
                    resize(imgs_to_img(self.rgb_light_env.detach().permute(0, 3, 1, 2)), scale_factor=1. / config_visualize.down_sample_rate),
                    caption=f"rgb light env",
                )

        return results_batch_visual

    def get_results_visual(
        self,
        results_epoch,
        dataset: OD3D_Dataset,
        config_visualize: DictConfig,
        filter_name_unique=True,
        caption_metrics=["sim", "rot_diff_rad"],
    ):
        results = OD3D_Results(logging_dir=self.logging_dir)
        count_best = config_visualize.count_best
        count_worst = config_visualize.count_worst
        count_rand = config_visualize.count_rand
        modalities = config_visualize.modalities
        if len(modalities) == 0:
            return results

        if "rot_diff_rad" in results_epoch.keys():
            rank_metric_name = "rot_diff_rad"
            # sorts values ascending
            epoch_ranked_ids = results_epoch[rank_metric_name].sort(dim=0)[1]
        elif "loss_batch" in results_epoch.keys():
            rank_metric_name = "loss_batch"
            # sorts values ascending
            epoch_ranked_ids = results_epoch[rank_metric_name].sort(dim=0)[1]
        elif "sim" in results_epoch.keys():
            rank_metric_name = "sim"
            # sorts values descending
            epoch_ranked_ids = results_epoch[rank_metric_name].sort(
                dim=0,
                descending=True,
            )[1]
        else:
            logger.warning(
                f"Could not find a suitable rank metric in results {results_epoch.keys()}",
            )
            return results

        if (
            filter_name_unique
            and "name_unique" in results_epoch.keys()
            and len(results_epoch["name_unique"]) > 0
        ):
            # this only groups the ranked elements depending on their category / sequence etc.
            # https://stackoverflow.com/questions/51408344/pandas-dataframe-interleaved-reordering
            group_names = list(
                {
                    "/".join(name_unique.split("/")[:-1])
                    for name_unique in results_epoch["name_unique"]
                },
            )
            group_ids = [
                group_id
                for result_id in range(len(results_epoch["name_unique"]))
                for group_id, group_name in enumerate(group_names)
                if results_epoch["name_unique"][epoch_ranked_ids[result_id]].startswith(
                    group_name,
                )
            ]
            df = pd.DataFrame(
                np.stack([epoch_ranked_ids, np.array(group_ids)], axis=-1),
                columns=["rank", "group"],
            )
            epoch_ranked_ids = torch.from_numpy(
                df.loc[
                    df.groupby("group").cumcount().sort_values(kind="mergesort").index
                ]["rank"].values,
            )

            df = df[::-1]
            epoch_ranked_ids_worst = torch.from_numpy(
                df.loc[
                    df.groupby("group").cumcount().sort_values(kind="mergesort").index
                ]["rank"].values,
            )
        else:
            epoch_ranked_ids_worst = epoch_ranked_ids.flip(dims=(0,))

        epoch_best_ids = epoch_ranked_ids[:count_best]
        epoch_best_names = [f"best/{i+1}" for i in range(len(epoch_best_ids))]
        epoch_worst_ids = epoch_ranked_ids_worst[:count_worst]
        epoch_worst_names = [
            f"worst/{len(epoch_worst_ids) - i}" for i in range(len(epoch_worst_ids))
        ]
        epoch_rand_ids = epoch_ranked_ids[
            torch.randperm(len(epoch_ranked_ids))[:count_rand]
        ]
        epoch_rand_names = [f"rand/{i+1}" for i in range(len(epoch_rand_ids))]
        epoch_sel_ids = torch.LongTensor(config_visualize.get("selected", []))
        epoch_sel_ids = epoch_sel_ids[epoch_sel_ids < len(results_epoch["item_id"])]
        epoch_sel_names = [f"sel/{i + 1}" for i in range(len(epoch_sel_ids))]

        sel_rank_ids = torch.cat(
            [epoch_best_ids, epoch_worst_ids, epoch_rand_ids, epoch_sel_ids],
            dim=0,
        )
        sel_item_ids = results_epoch["item_id"][sel_rank_ids]
        sel_names = epoch_best_names + epoch_worst_names + epoch_rand_names + epoch_sel_names
        sel_name_unique = [results_epoch["name_unique"][id] for id in sel_rank_ids]
        dict_name_unique_to_result_id = dict(zip(sel_name_unique, sel_rank_ids))
        dict_name_unique_to_sel_name = dict(zip(sel_name_unique, sel_names))
        logger.info("create dataset ...")
        dataset_visualize = dataset.get_subset_with_item_ids(item_ids=sel_item_ids)

        logger.info("create dataloader ...")
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset_visualize,
            batch_size=self.config.test.dataloader.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            num_workers=self.config.test.dataloader.num_workers,
            pin_memory=self.config.test.dataloader.pin_memory,
        )

        for i, batch in tqdm(enumerate(iter(dataloader))):
            results += self.get_results_visual_batch(
                batch,
                results_epoch,
                config_visualize=config_visualize,
                dict_name_unique_to_sel_name=dict_name_unique_to_sel_name,
                caption_metrics=caption_metrics,
                dict_name_unique_to_result_id=dict_name_unique_to_result_id,
            )

        return results

    def get_nearest_corresp2d3d(
        self,
        feats2d_net,
        meshes_ids,
        feats2d_net_mask: torch.Tensor = None,
    ):
        (
            nearest_verts3d,
            sim_texture,
            sim_clutter,
        ) = self.get_nearest_verts3d_to_feats2d_net(
            feats2d_net,
            meshes_ids,
            return_sim_texture_and_clutter=True,
        )
        nearest_verts2d = get_pxl2d_like(nearest_verts3d.permute(0, 2, 3, 1)).permute(
            0,
            3,
            1,
            2,
        )
        # H=sim_clutter.shape[1], W=sim_clutter.shape[2], dtype=sim_nearest_texture_verts.dtype, device=sim_nearest_texture_verts.device)[None,].expand()
        # prob_corresp2d3d = ((sim_texture + 1) / 2) * (1-((sim_clutter+1)/ 2)) #  (sim_clutter < sim_texture) * sim_texture
        prob_corresp2d3d = torch.exp(sim_texture) / (
            torch.exp(sim_texture) + torch.exp(sim_clutter)
        )

        prob_corresp2d3d *= feats2d_net_mask

        return nearest_verts3d, nearest_verts2d, prob_corresp2d3d


    def get_samples(
        self,
        config_sample: DictConfig,
        cam_intr4x4: torch.Tensor,
        cam_tform4x4_obj: torch.Tensor,
        feats2d_net: torch.Tensor,
        categories_ids: torch.LongTensor,
        feats2d_net_mask: torch.Tensor = None,
        multiview=False,
        instance_deform=None,
        pred_cam_tform4x4_objs=None,
    ):
        if multiview:
            if instance_deform is not None:
                instance_deform_first = instance_deform[:1]
            else:
                instance_deform_first = None

            if pred_cam_tform4x4_objs is not None:
                pred_cam_tform4x4_objs_first = pred_cam_tform4x4_objs[:1]
            else:
                pred_cam_tform4x4_objs_first = None

            b_cams_multiview_tform4x4_obj, b_cams_multiview_intr4x4 = self.get_samples(
                config_sample=config_sample,
                cam_intr4x4=cam_intr4x4[:1],
                cam_tform4x4_obj=cam_tform4x4_obj[:1],
                feats2d_net=feats2d_net[:1],
                categories_ids=categories_ids[:1],
                feats2d_net_mask=feats2d_net_mask[:1],
                instance_deform=instance_deform_first,
                pred_cam_tform4x4_objs=pred_cam_tform4x4_objs_first
            )
            # scale = (
            #     b_cams_multiview_tform4x4_obj[..., 2, 3]
            #     / cam_tform4x4_obj[:, None, 2, 3]
            # )
            # cam_tform4x4_obj_scaled = (
            #     cam_tform4x4_obj[:, None]
            #     .clone()
            #     .expand(
            #         cam_tform4x4_obj.shape[0],
            #         *b_cams_multiview_tform4x4_obj.shape[1:],
            #     )
            #     .clone()
            # )
            #
            # # use rot. scaling
            # # cam_tform4x4_obj_scaled[:, :, :3, :3] = cam_tform4x4_obj_scaled[:, :, :3, :3] * scale[:, :, None, None]
            # # use transl. scaling
            # # cam_tform4x4_obj_scaled[:, :, :3, 3] = cam_tform4x4_obj_scaled[:, :, :3, 3] * scale[:, :, None]
            # # use depth scaling
            # cam_tform4x4_obj_scaled[:, :, 2, 3] = (
            #     cam_tform4x4_obj_scaled[:, :, 2, 3] * scale[:, :]
            # )
            #
            # objs_multiview_tform4x4_cuboid_front = tform4x4_broadcast(
            #     inv_tform4x4(cam_tform4x4_obj_scaled[:1]),
            #     b_cams_multiview_tform4x4_obj,
            # )
            # b_cams_multiview_tform4x4_obj = tform4x4_broadcast(
            #     cam_tform4x4_obj_scaled,
            #     objs_multiview_tform4x4_cuboid_front,
            # )

            b_cams_multiview_tform4x4_obj = tform4x4_broadcast(
                cam_tform4x4_obj[:, None].clone(),
                tform4x4_broadcast(inv_tform4x4(cam_tform4x4_obj[:1, None]), b_cams_multiview_tform4x4_obj),
            )

            b_cams_multiview_intr4x4 = cam_intr4x4[:, None].expand(
                *b_cams_multiview_tform4x4_obj.shape,
            )

        else:
            B = len(feats2d_net)
            if "pred" in config_sample.method:
                C = pred_cam_tform4x4_objs.shape[1]
                b_cams_multiview_tform4x4_obj = pred_cam_tform4x4_objs.clone()
                # assumption 1: distance translation to object is known
                # b_cams_multiview_tform4x4_obj[:, :, 2, 3] = cam_tform4x4_obj[:, None].repeat(1, C, 1, 1)[:, :, 2, 3]
                # logger.info(f'dist {batch.cam_tform4x4_obj[:, 2, 3]}')
                # assumption 2: translation to object is known
                #b_cams_multiview_tform4x4_obj[:, :, :4, 3] = cam_tform4x4_obj[
                #    :,
                #    None,
                #].repeat(1, C, 1, 1)[:, :, :4, 3]

                b_cams_multiview_intr4x4 = cam_intr4x4[:, None].repeat(1, C, 1, 1)

            elif "uniform" in config_sample.method:
                azim = torch.linspace(
                    start=eval(config_sample.uniform.azim.min),
                    end=eval(config_sample.uniform.azim.max),
                    steps=config_sample.uniform.azim.steps,
                ).to(
                    device=self.device,
                )  # 12
                elev = torch.linspace(
                    start=eval(config_sample.uniform.elev.min),
                    end=eval(config_sample.uniform.elev.max),
                    steps=config_sample.uniform.elev.steps,
                ).to(
                    device=self.device,
                )  # start=-torch.pi / 6, end=torch.pi / 3, steps=4
                theta = torch.linspace(
                    start=eval(config_sample.uniform.theta.min),
                    end=eval(config_sample.uniform.theta.max),
                    steps=config_sample.uniform.theta.steps,
                ).to(
                    device=self.device,
                )  # -torch.pi / 6, end=torch.pi / 6, steps=3

                # dist = torch.linspace(start=eval(config_sample.uniform.dist.min), end=eval(config_sample.uniform.dist.max), steps=config_sample.uniform.dist.steps).to(
                #    device=self.device)
                dist = torch.linspace(start=1.0, end=1.0, steps=1).to(
                    device=self.device,
                )

                azim_shape = azim.shape
                elev_shape = elev.shape
                theta_shape = theta.shape
                dist_shape = dist.shape
                in_shape = azim_shape + elev_shape + theta_shape + dist_shape
                azim = azim[:, None, None, None].expand(in_shape).reshape(-1)
                elev = elev[None, :, None, None].expand(in_shape).reshape(-1)
                theta = theta[None, None, :, None].expand(in_shape).reshape(-1)
                dist = dist[None, None, None, :].expand(in_shape).reshape(-1)
                cams_multiview_tform4x4_cuboid = transf4x4_from_spherical(
                    azim=azim,
                    elev=elev,
                    theta=theta,
                    dist=dist,
                )

                C = len(cams_multiview_tform4x4_cuboid)

                b_cams_multiview_tform4x4_obj = cams_multiview_tform4x4_cuboid[
                    None,
                ].repeat(B, 1, 1, 1)

                # assumption 1: distance translation to object is known
                # b_cams_multiview_tform4x4_obj[:, :, 2, 3] = cam_tform4x4_obj[:, None].repeat(1, C, 1, 1)[:, :, 2, 3]
                # logger.info(f'dist {batch.cam_tform4x4_obj[:, 2, 3]}')
                # assumption 2: translation to object is known
                b_cams_multiview_tform4x4_obj[:, :, :3, 3] = cam_tform4x4_obj[
                    :,
                    None,
                ].repeat(1, C, 1, 1)[:, :, :3, 3]

                b_cams_multiview_intr4x4 = cam_intr4x4[:, None].repeat(1, C, 1, 1)

            elif "epnp3d2d" in config_sample.method:
                coarse_labels = config_sample.epnp3d2d.get("coarse_labels", False)
                # B x F x H x W
                sim_feats = self.meshes.get_sim_feats2d_img_to_all(
                    feats2d_img=feats2d_net,
                    imgs_sizes=None,
                    cams_tform4x4_obj=None,
                    cams_intr4x4=None,
                    objects_ids=categories_ids,
                    broadcast_batch_and_cams=False,
                    down_sample_rate=self.down_sample_rate,
                    add_clutter=True,
                    add_other_objects=False,
                    dense=True,
                    sim_temp=self.config.train.T,
                    clutter_pxl2d=0,
                    return_feats=False,
                    instance_deform=None,
                    coarse_labels=coarse_labels
                )

                H, W = feats2d_net.shape[-2:]
                # sim_feats[sim_feats == 0.] = -torch.inf
                sim_feats_top_vals, sim_feats_top_ids = sim_feats[:, :].sort(dim=1, descending=True)
                sim_feats_selected = torch.ones_like(sim_feats)
                sim_feats_selected[:] = - torch.inf

                # add top
                pt_nn_top_k = config_sample.epnp3d2d.pt_nn_top_k
                sim_feats_selected = torch.scatter(input=sim_feats_selected, dim=1, index=sim_feats_top_ids[:, :pt_nn_top_k], src=sim_feats_top_vals[:, :pt_nn_top_k])
                # add clutter
                sim_feats_selected[:, -1:] = sim_feats[:, -1:]

                modality_pt3d = PROJECT_MODALITIES.PT3D if not coarse_labels else PROJECT_MODALITIES.PT3D_COARSE
                pts3d = self.meshes.sample(objects_ids=categories_ids, modalities=modality_pt3d, add_clutter=False,
                                           add_other_objects=False, instance_deform=instance_deform)
                pts3d_prob = torch.softmax(sim_feats_selected, dim=1)[:, :-1].flatten(2)

                #from od3d.cv.visual.show import show_imgs
                #show_imgs(feats2d_net_mask)

                # from od3d.cv.visual.show import show_imgs
                # show_imgs(pts3d_prob.sum(dim=1).reshape(B, 1, H, W))

                if feats2d_net_mask is not None:
                    pts3d_prob = pts3d_prob * feats2d_net_mask.flatten(2)


                # B x F x N
                pts3d_prob[pts3d_prob.sum(dim=-1).sum(dim=-1) == 0.] = 1.
                K = config_sample.epnp3d2d.count_cams
                N = config_sample.epnp3d2d.count_pts
                if N == -1:
                    pts3d_prob[(pts3d_prob.sum(dim=-1) > 0).sum(dim=-1) < 4] = 1.
                else:
                    pts3d_prob[(pts3d_prob.sum(dim=-1) > 0).sum(dim=-1) < N] = 1.


                # rgb = torch.zeros_like(feats2d_net)[:, :3]
                # from od3d.cv.select import batched_indexMD_fill, batched_index_select
                # pts3d_plus_zero = torch.cat([pts3d, torch.zeros_like(pts3d[:, :1])], dim=1)
                # rgb = batched_index_select(input=pts3d_plus_zero, index=sim_feats_top_ids[:, 0].flatten(1), dim=1).clone().permute(0, 2, 1).reshape(-1, 3, *feats2d_net.shape[-2:])
                # #rgb = batched_indexMD_fill(inputMD=rgb, indexMD=sim_feats_top_ids[:, :1].permute(0, 2, 1)[..., None, :, :].expand(4, 3, 200, 2).long(), value=pts3d_pairs, dims=[2, 3])
                # #rgb = rgb.reshape(4, 3, 64, 64)
                # rgb = (rgb / (rgb.norm(dim=1, keepdim=True) + 1e-10)).clamp(0, 1)
                # from od3d.cv.visual.show import show_imgs
                # show_imgs(rgb)

                pxl2d = get_pxl2d_like(sim_feats.permute(0, 2, 3, 1)).permute(
                    0,
                    3,
                    1,
                    2,
                ).flatten(2)


                count_pts3d = pts3d_prob.shape[-2]
                count_pxl2d = pts3d_prob.shape[-1]
                # SAMPLING SEQUENTIAL
                # sample 1st 2D
                # N x K (where N never has includes the same pxl id)
                # pxl2d_ids = torch.multinomial(
                #     pts3d_prob.sum(dim=-2)[:, None].expand(pts3d_prob.shape[0], K, count_pxl2d).reshape(-1, count_pxl2d),
                #     num_samples=N,
                #     replacement=False,
                # ).reshape(pts3d_prob.shape[0], K, N)
                # from od3d.cv.select import batched_index_select
                # pts3d_prob_3d = batched_index_select(input=pts3d_prob, index=pxl2d_ids.reshape(-1, K*N), dim=2)
                # pts3d_prob_3d = pts3d_prob_3d.reshape(-1, K, N, count_pts3d)
                # pts3d_prob_3d[pts3d_prob_3d.sum(dim=-1) == 0] = 1.
                # pts3d_prob_3d[(pts3d_prob_3d.sum(dim=-2) > 0).sum(dim=-1) < N] = 1.
                #
                # # sample 2nd 2D
                # pts3d_ids = torch.multinomial(pts3d_prob_3d.reshape(-1, count_pts3d), num_samples=1, replacement=False)
                # pts3d_ids = pts3d_ids.reshape(pts3d_prob.shape[0], K, N)

                #SAMPLING SEQUENTIAL: First 3D
                #sample 1st 3D
                # N x K (where N never has includes the same pxl id)
                # pts3d_ids = torch.multinomial(
                #     pts3d_prob.sum(dim=-1)[:, None].expand(pts3d_prob.shape[0], K, count_pts3d).reshape(-1, count_pts3d),
                #     num_samples=N,
                #     replacement=False,
                # ).reshape(pts3d_prob.shape[0], K, N)
                # from od3d.cv.select import batched_index_select
                # pts3d_prob_2d = batched_index_select(input=pts3d_prob, index=pts3d_ids.reshape(-1, K*N), dim=1)
                # pts3d_prob_2d = pts3d_prob_2d.reshape(-1, K, N, count_pxl2d)
                # pts3d_prob_2d[pts3d_prob_2d.sum(dim=-1) == 0] = 1.
                # #pts3d_prob_2d[(pts3d_prob_2d.sum(dim=-2) > 0).sum(dim=-1) < N] = 1.
                #
                # # sample 2nd 2D
                # pxl2d_ids = torch.multinomial(pts3d_prob_2d.reshape(-1, count_pxl2d), num_samples=1, replacement=False)
                # pxl2d_ids = pxl2d_ids.reshape(pts3d_prob.shape[0], K, N)

                # SAMPLING ALL TOGETHER
                # N x K (where N never has includes the same pxl id)

                # H, W = feats2d_net.shape[2:]
                # grid_pxl2d = torch.cartesian_prod(torch.arange(W), torch.arange(H)).to(device=feats2d_net.device) * 1.
                # grid_pxl2d_dist = torch.cdist(grid_pxl2d, grid_pxl2d)
                # grid_pxl2d_prob = 1. - torch.exp(- grid_pxl2d_dist / ( W / 10 )) # , dim=-1)

                #from od3d.cv.visual.show import show_imgs
                #show_imgs(255 * grid_pxl2d_prob[:3].reshape(3, 1, H, W))

                # pts3d2d_ids = []
                # for n in range(N):
                #     _pts3d2d_ids = torch.multinomial(
                #         # pts3d_prob.flatten(1),
                #         pts3d_prob[:, None].expand(pts3d_prob.shape[0], K, *pts3d_prob.shape[1:]).reshape(
                #             pts3d_prob.shape[0] * K, -1),  # BxVxH*W
                #         num_samples=1,
                #         replacement=False,
                #     ).reshape(pts3d_prob.shape[0], K, 1)
                #     pts3d2d_ids.append(_pts3d2d_ids)
                #
                #     pxl2d_ids = _pts3d2d_ids % count_pxl2d  # BxKxN
                #     pts3d_prob *= grid_pxl2d_prob[pxl2d_ids][..., 0, :]
                # pts3d2d_ids = torch.cat(pts3d2d_ids, dim=-1)

                if N != -1:
                    pts3d2d_ids = torch.multinomial(
                        #pts3d_prob.flatten(1),
                        pts3d_prob[:, None].expand(pts3d_prob.shape[0], K, *pts3d_prob.shape[1:]).reshape(pts3d_prob.shape[0] * K, -1), # BxVxH*W
                        num_samples=N,
                        replacement=False,
                    ).reshape(pts3d_prob.shape[0], K, N)

                    pts3d_ids = pts3d2d_ids // count_pxl2d # BxKxN
                    pxl2d_ids = pts3d2d_ids % count_pxl2d # BxKxN
                    device = pts3d.device
                    dtype = pts3d.dtype
                    pts3d_pairs = torch.zeros(B, 3, K, N).to(device=device, dtype=dtype)
                    pts3d_pairs_prob = torch.zeros(B, K, K, N).to(device=device, dtype=dtype)
                    pxl2d_pairs = torch.zeros(B, 2, K, N).to(device=device, dtype=dtype)
                    pts3d2d_pairs_mask = torch.eye(K).to(device=device, dtype=bool)[None, :, :, None].expand(B, K, K, N) # (B, K, K*N)
                    pts3d2d_pairs_mask = pts3d2d_pairs_mask.reshape(B, K, K*N)
                    for b in range(B):
                        pts3d_pairs[b] = pts3d[b, pts3d_ids[b], :].permute(2, 0, 1)
                        pxl2d_pairs[b] = pxl2d[b, :, pxl2d_ids[b]]
                        pts3d_pairs_prob[b] = pts3d_prob[b].flatten()[pts3d2d_ids[b]][None,].expand(K, K, N)

                    pts3d_pairs_prob = pts3d_pairs_prob.reshape(B, K, K*N)
                    pts3d_pairs = pts3d_pairs.reshape(B, 3, -1)
                    pxl2d_pairs = pxl2d_pairs.reshape(B, 2, -1)

                    if not config_sample.epnp3d2d.pt_nn_weights:
                        pts3d_pairs_prob = None

                    # rgb = torch.zeros_like(feats2d_net)[:, :3]
                    # from od3d.cv.select import batched_indexMD_fill
                    # rgb = batched_indexMD_fill(inputMD=rgb, indexMD=pxl2d_pairs.permute(0, 2, 1)[..., None, :, :].expand(rgb.shape[0], 3, K * N, 2).long().flip(dims=(-1,)), value=pts3d_pairs, dims=[2, 3])
                    # rgb = rgb.reshape(rgb.shape[0], 3, *feats2d_net.shape[-2:])
                    # rgb = (rgb / (rgb.norm(dim=1, keepdim=True) + 1e-10)).clamp(0, 1)
                    # from od3d.cv.visual.show import show_imgs
                    # show_imgs(rgb)
                else:
                    pxl2d_pairs = pxl2d.reshape(*pxl2d.shape[:2], H, W)
                    pts3d_pairs_prob, pts3d_pairs_ids = pts3d_prob.max(dim=1)
                    pts3d_pairs = batched_index_select(input=pts3d, index=pts3d_pairs_ids, dim=1)
                    pts3d_pairs = pts3d_pairs.permute(0, 2, 1)
                    pts3d_pairs = pts3d_pairs.reshape(*pts3d_pairs.shape[:2], H, W)
                    pts3d_pairs_prob = pts3d_pairs_prob.reshape(*pts3d_pairs_prob.shape[:1], 1, H, W)
                    pts3d2d_pairs_mask = pts3d_pairs_prob > 0
                    if not config_sample.epnp3d2d.pt_nn_weights:
                        pts3d_pairs_prob = None

                b_cams_multiview_tform4x4_obj = (
                    batchwise_fit_se3_to_corresp_3d_2d_and_masks(
                        masks_in=pts3d2d_pairs_mask,
                        pts1=pts3d_pairs,
                        pxl2=pxl2d_pairs,
                        proj_mat=cam_intr4x4[
                            :,
                            :2,
                            :3,
                        ]
                        / self.down_sample_rate,
                        method="cpu-epnp",
                        weights=pts3d_pairs_prob,
                    )
                )

                if b_cams_multiview_tform4x4_obj.isnan().any():
                    logger.info(f'NaNs in cam_tform_obj')
                    b_nans = b_cams_multiview_tform4x4_obj.flatten(2).isnan().any(dim=-1)
                    b_cams_multiview_tform4x4_obj[b_nans] = torch.eye(4).to(device=b_nans.device)[None]
                if b_cams_multiview_tform4x4_obj.abs().max() > 1e+5:
                    logger.info(f'Too large values in cam_tform_obj')
                    b_too_largs = (b_cams_multiview_tform4x4_obj.flatten(2).abs() > 1e+5).any(dim=-1)
                    b_cams_multiview_tform4x4_obj[b_too_largs] = torch.eye(4).to(device=b_too_largs.device)[None]

                b_cams_multiview_intr4x4 = cam_intr4x4[:, None].repeat(1, K, 1, 1)
                b_cams_multiview_tform4x4_obj[
                    b_cams_multiview_tform4x4_obj.flatten(2).isinf().any(dim=2),
                    :,
                    :,
                ] = torch.eye(4, device=b_cams_multiview_tform4x4_obj.device)
                b_cams_multiview_tform4x4_obj[
                    (b_cams_multiview_tform4x4_obj[:, :, 3, :3] != 0.0).any(dim=-1),
                    :,
                    :,
                ] = torch.eye(4, device=b_cams_multiview_tform4x4_obj.device)

            else:
                raise NotImplementedError

        if config_sample.depth_from_box:
            from od3d.cv.geometry.fit.depth_from_mesh_and_box import (
                depth_from_mesh_and_box,
            )

            b_cams_multiview_tform4x4_obj[..., 2, 3] = depth_from_mesh_and_box(
                b_cams_multiview_intr4x4=cam_intr4x4,
                b_cams_multiview_tform4x4_obj=b_cams_multiview_tform4x4_obj,
                meshes=self.meshes,
                labels=categories_ids,
                mask=feats2d_net_mask,
                downsample_rate=self.down_sample_rate,
                multiview=multiview,
            )
        return b_cams_multiview_tform4x4_obj, b_cams_multiview_intr4x4

    def visualize_test(self):
        pass

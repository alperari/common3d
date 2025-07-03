import logging
import time

import numpy as np
import od3d.io
import pandas as pd
from od3d.benchmark.results import OD3D_Results
from od3d.cv.geometry.objects3d.meshes.meshes import VERT_MODALITIES
from od3d.cv.metric.pose import get_pose_diff_in_rad
from od3d.cv.select import batched_index_select
from od3d.datasets.dataset import OD3D_Dataset
from od3d.methods.method import OD3D_Method
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
import torch
from od3d.cv.geometry.metrics.dist import (
    batch_chamfer_distance,
    batch_point_face_distance,
    batch_point_face_distance_v2,
)
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
from od3d.datasets.omni6dpose import Omni6DPose
from od3d.datasets.spair71k import SPair71K

from od3d.cv.io import image_as_wandb_image
from od3d.cv.visual.resize import resize
from od3d.models.model import OD3D_Model

from od3d.cv.geometry.grid import get_pxl2d_like
from od3d.cv.geometry.fit3d2d import batchwise_fit_se3_to_corresp_3d_2d_and_masks
from od3d.cv.transforms.transform import OD3D_Transform
from od3d.cv.transforms.sequential import SequentialTransform

from typing import Dict

import matplotlib.pyplot as plt

plt.switch_backend("Agg")
from od3d.cv.visual.show import get_img_from_plot
from od3d.cv.visual.draw import draw_text_in_rgb
from od3d.data.ext_enum import StrEnum


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
    MESH_FEATS = "mesh_feats"
    LATENT_INTERP = "latent_interp"


class SIM_FEATS_MESH_WITH_IMAGE(StrEnum):
    VERTS2D = "verts2d"
    RENDERED = "rendered"


class NeMo(OD3D_Method):
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
        self.net = OD3D_Model(config.model)
        torch.multiprocessing.set_sharing_strategy("file_system")

        self.kpts3d = None
        self.kpts3d_id = None
        self.transform_train = SequentialTransform(
            [
                OD3D_Transform.subclasses[
                    config.train.transform.class_name
                ].create_from_config(config=config.train.transform),
                self.net.transform,
            ],
        )
        self.transform_test = SequentialTransform(
            [
                OD3D_Transform.subclasses[
                    config.test.transform.class_name
                ].create_from_config(config=config.test.transform),
                self.net.transform,
            ],
        )

        # init Meshes / Features
        self.total_params = sum(p.numel() for p in self.net.parameters())
        self.trainable_params_net = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )

        # self.path_shapenemo = Path(config.path_shapenemo)
        # self.fpaths_meshes_shapenemo = [self.path_shapenemo.joinpath(cls, '01.off') for cls in config.categories]
        self.fpaths_meshes = [
            self.config.fpaths_meshes[cls]
            if (
                self.config.fpaths_meshes is not None
                and cls in self.config.fpaths_meshes
            )
            else None
            for cls in config.categories
        ]
        fpaths_meshes_tform_obj = self.config.get("fpaths_meshes_tform_obj", None)
        if fpaths_meshes_tform_obj is not None:
            self.fpaths_meshes_tform_obj = [
                fpaths_meshes_tform_obj[cls] if cls in fpaths_meshes_tform_obj else None
                for cls in config.categories
            ]
        else:
            self.fpaths_meshes_tform_obj = [None for _ in config.categories]

        from od3d.cv.geometry.objects3d.objects3d import OD3D_Objects3D

        self.feat_dim = self.net.out_dim

        if self.config.get("pose_net_config", None) is not None:
            pose_net_in_dim = self.net.backbone.out_dims[-1] + 3

            self.net_pose_translations = self.config.pose_net_config.translations
            self.net_pose_rotation_dim = 6
            self.net_pose_rotations_othant = torch.BoolTensor(
                self.config.pose_net_config.rot6d_othant,
            ).to(self.device)
            self.net_pose_rotations_othant_signs = (
                torch.stack(
                    torch.meshgrid(
                        [
                            torch.arange(1, -2, -2) if othant else torch.arange(1, 2, 1)
                            for othant in self.net_pose_rotations_othant
                        ],
                    ),
                    -1,
                )
                .view(-1, self.net_pose_rotation_dim)
                .to(device=self.device)
            )  # 8x6

            self.net_pose_rotations = 2 ** int(
                self.net_pose_rotations_othant.sum().item(),
            )

            pose_net_out_dim = (
                self.net_pose_translations
                * self.net_pose_rotations
                * (self.net_pose_rotation_dim + 3)
            )  # translations * rotations * se3 params
            self.config.pose_net_config.head.update({"in_dims": [pose_net_in_dim]})
            self.config.pose_net_config.head.update({"in_upsample_scales": []})
            self.config.pose_net_config.head.fully_connected.update(
                {"out_dim": pose_net_out_dim},
            )

            self.net_pose = OD3D_Model(self.config.pose_net_config)
            self.net_pose.cuda()

            self.net_pose_params = list(self.net_pose.parameters())
        else:
            self.net_pose = None
            self.net_pose_params = []

        self.config.objects3d.update(
            {
                "feats_requires_grad": self.config.train.bank_feats_update
                != "moving_average"
                and self.config.train.bank_feats_update != "average",
                "feat_dim": self.feat_dim,
                "fpaths_meshes": self.fpaths_meshes,
                "fpaths_meshes_tforms": self.fpaths_meshes_tform_obj,
                "feats_objects": True,
                "feat_clutter": True,
            },
        )
        if self.config.objects3d.instance_deform_net_config is not None:
            instance_deform_net_nearest_pt3d = (
                self.config.objects3d.instance_deform_net_config.get(
                    "nearest_pt3d",
                    "cat",
                )
            )
            instance_deform_net_in_feats = (
                self.config.objects3d.instance_deform_net_config.get(
                    "in_feats",
                    "backbone",
                )
            )
            if instance_deform_net_nearest_pt3d == "cat":
                if instance_deform_net_in_feats == "backbone":
                    instance_deform_net_in_dim = self.net.backbone.out_dims[-1] + 3
                else:
                    instance_deform_net_in_dim = self.feat_dim
            elif instance_deform_net_nearest_pt3d == "cat_harmonics":
                if instance_deform_net_in_feats == "backbone":
                    instance_deform_net_in_dim = (
                        self.net.backbone.out_dims[-1] + 3 * 2 * 10
                    )
                else:
                    instance_deform_net_in_dim = self.feat_dim + 3 * 2 * 10
            elif instance_deform_net_nearest_pt3d == "sole":
                instance_deform_net_in_dim = 3
            elif instance_deform_net_nearest_pt3d == "sole_harmonics":
                instance_deform_net_in_dim = 3 * 2 * 10
            else:
                if instance_deform_net_in_feats == "backbone":
                    instance_deform_net_in_dim = self.net.backbone.out_dims[-1]
                else:
                    instance_deform_net_in_dim = self.feat_dim
            self.config.objects3d.instance_deform_net_config.head.update(
                {"in_dim": instance_deform_net_in_dim},
            )

        config_objects3d_without_class_name = self.config.objects3d.copy()
        del config_objects3d_without_class_name.class_name

        self.meshes = OD3D_Objects3D.subclasses[
            self.config.objects3d.class_name
        ].read_from_ply_files(
            **config_objects3d_without_class_name,
        )

        self.meshes_ranges = self.meshes.get_ranges().detach().cuda()
        self.refine_update_max = (
            torch.Tensor(self.config.inference.refine.dims_grad_max)
            .cuda()[None,]
            .expand(self.meshes_ranges.shape[0], 6)
            .clone()
        )
        self.refine_update_max[:, :3] = (
            self.refine_update_max[:, :3] * self.meshes_ranges
        )

        # self.meshes.rgb = (self.meshes.geodesic_prob[3, :, None].repeat(1, 3)).clamp(0, 1)
        # self.meshes.show()
        # watch_model_in_wandb(self.net, log="all")

        logger.info(f"loading meshes from following fpaths: {self.fpaths_meshes}...")
        # self.meshes.show()
        self.verts_count_max = self.meshes.verts_counts_max
        self.mem_verts_feats_count = len(config.categories) * self.verts_count_max
        self.mem_clutter_feats_count = config.num_noise * config.max_group
        self.mem_count = self.mem_verts_feats_count + self.mem_clutter_feats_count

        self.feats_bank_count = self.verts_count_max * len(self.meshes) + 1

        # self.meshes.set_feats_cat_with_pad(torch.nn.Parameter(torch.randn(size=(self.verts_count_max * len(self.meshes), self.net.feat_dim), device=self.device), requires_grad=True))

        # dict to save estimated tforms, sequence : tform,
        self.seq_obj_tform4x4_est_obj = {}
        self.seq_obj_tform4x4_est_obj_sim = {}

        self.total_params_mesh_clutter = sum(
            p.numel() for p in self.meshes.parameters()
        )
        self.trainable_params_mesh_clutter = sum(
            p.numel() for p in self.meshes.parameters() if p.requires_grad
        )

        if (
            self.config.train.loss.appear.type == "cross_entropy"
            or self.config.train.loss.appear.type == "cross_entropy_smooth"
            or self.config.train.loss.appear.type == "cross_entropy_coarse"
        ):
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        elif self.config.train.loss.appear.type == "nll_softmax":
            self.softmax = torch.nn.LogSoftmax(dim=1)
            self.criterion = torch.nn.NLLLoss().cuda()
        elif self.config.train.loss.appear.type == "nll_clip":
            self.criterion = torch.nn.NLLLoss().cuda()
        elif self.config.train.loss.appear.type == "nll_affine_to_prob":
            self.criterion = torch.nn.NLLLoss().cuda()
        elif self.config.train.loss.appear.type == "l2":
            self.criterion = torch.nn.MSELoss().cuda()
        elif self.config.train.loss.appear.type == "l2_squared":
            self.criterion = torch.nn.MSELoss().cuda()

        self.loss_appear_dropout = torch.nn.Dropout1d(
            p=self.config.train.loss.appear.dropout,
        )

        # self.net = torch.nn.DataParallel(self.net).cuda()
        self.net.cuda()
        self.meshes.cuda()

        self.pseudo_labels_fraction = (
            -self.config.train.pseudo_labels_fraction_per_epoch
        )
        self.pseudo_tform4x4_obj = {}

        # load checkpoint
        self.load_checkpoint_params(
            path_checkpoint=config.get("checkpoint", None),
            path_checkpoint_old=config.get("checkpoint", None),
        )
        # load_mesh(config.path_shapenemo)

        # from od3d.cv.visual.show import show_scene
        # show_scene(meshes=self.meshes)

        self.net.eval()
        if self.net_pose is not None:
            self.net_pose.eval()
        self.meshes.eval()
        self.back_propagate = True

        logger.info(
            f"total params: {self.total_params}, trainable params: {self.trainable_params_net}",
        )
        logger.info(
            f"total params mesh and clutter: {self.total_params_mesh_clutter}, trainable params mesh and clutter: {self.trainable_params_mesh_clutter}",
        )

        if (
            self.config.train.bank_feats_update == "moving_average"
            or self.config.train.bank_feats_update == "average"
        ):
            if self.trainable_params_net == 0:
                logger.info("no trainable params, no optimizer needed.")
                self.optim = od3d.io.get_obj_from_config(
                    config=self.config.train.optimizer,
                    params=list(self.net.parameters()) + self.net_pose_params,
                )
                self.back_propagate = False
            else:
                self.optim = od3d.io.get_obj_from_config(
                    config=self.config.train.optimizer,
                    params=list(self.net.parameters()) + self.net_pose_params,
                )
        else:
            if self.trainable_params_net == 0:
                self.optim = od3d.io.get_obj_from_config(
                    config=self.config.train.optimizer,
                    params=list(set(self.meshes.parameters())) + self.net_pose_params,
                )
            else:
                self.optim = od3d.io.get_obj_from_config(
                    config=self.config.train.optimizer,
                    params=list(set(self.meshes.parameters()))
                    + list(self.net.parameters())
                    + self.net_pose_params,
                )

        self.scheduler = od3d.io.get_obj_from_config(
            self.optim,
            config=self.config.train.scheduler,
        )

        # load checkpoint
        self.load_checkpoint_optim(
            path_checkpoint=config.get("checkpoint", None),
            path_checkpoint_old=config.get("checkpoint", None),
        )

        # self.meshes.show()

        # self.verts_feats = checkpoint["memory"][:self.mem_verts_feats_count].clone().detach().cpu()
        # note: somehow vertices are stored in wrong order of classes (starting with last class tvmonitor until first class aeroplane
        # self.verts_feats = self.verts_feats.reshape(len(self.meshes), self.verts_count_max, -1).flip(dims=(0,)).reshape(len(self.meshes) * self.verts_count_max, -1)
        self.down_sample_rate = self.net.downsample_rate

        # import wandb
        # wandb.watch(self.meshes, log="all", log_freq=1)
        # wandb.watch(self.net, log="all", log_freq=1)

    def calc_sim(self, comb, featsA, featsB):
        """
        Expand and permute a tensor based on the einsum equation.

        Parameters:
            tensor (torch.Tensor): Input tensor.
            equation (str): Einsum equation specifying the dimensions.

        Returns:
            torch.Tensor: Expanded and permuted tensor.
        """
        if self.config.bank_feats_distribution == "von-mises-fisher":
            return torch.einsum(comb, featsA, featsB)
        elif self.config.bank_feats_distribution == "gaussian":
            from od3d.cv.geometry.dist import einsum_cdist

            return -einsum_cdist(comb, featsA, featsB)
        else:
            msg = f"Unknown distribution {self.config.distribution}"
            raise NotImplementedError(msg)

    def save_checkpoint(self, path_checkpoint: Path):
        torch.save(
            {
                "net_state_dict": self.net.state_dict(),
                "net_pose_state_dict": self.net_pose.state_dict()
                if self.net_pose is not None
                else None,
                "optimizer_state_dict": self.optim.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "meshes_feats": self.meshes.state_dict(),
            },
            path_checkpoint,
        )

    def load_checkpoint(self, path_checkpoint=None, path_checkpoint_old=None):
        self.load_checkpoint_params(
            path_checkpoint=path_checkpoint,
            path_checkpoint_old=path_checkpoint_old,
        )
        self.load_checkpoint_optim(
            path_checkpoint=path_checkpoint,
            path_checkpoint_old=path_checkpoint_old,
        )

    def load_checkpoint_optim(self, path_checkpoint=None, path_checkpoint_old=None):
        if path_checkpoint is None and self.fpath_checkpoint.exists():
            path_checkpoint = self.fpath_checkpoint

        if path_checkpoint is not None:
            checkpoint = torch.load(Path(path_checkpoint))
            # try:
            #     self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
            # except Exception as e:
            #     logger.warning(f'could not load optimizer state dict {e}')
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                logger.warning(f"could not load net state dict {e}")
        elif path_checkpoint_old is not None:
            pass

    def load_checkpoint_params(self, path_checkpoint=None, path_checkpoint_old=None):
        if path_checkpoint is None and self.fpath_checkpoint.exists():
            path_checkpoint = self.fpath_checkpoint

        if path_checkpoint is not None:
            checkpoint = torch.load(Path(path_checkpoint))
            try:
                self.net.load_state_dict(checkpoint["net_state_dict"], strict=True)
                if self.net is not None and self.net.backbone is not None:
                    pca_enabled = False
                    if (
                        self.config.model is not None
                        and self.config.model.backbone is not None
                        and self.config.model.backbone is not None
                    ):
                        pca_enabled = self.config.model.backbone.pca.get(
                            "enable",
                            False,
                        )
                    self.net.backbone.pca_enabled = pca_enabled
            except Exception as e:
                logger.warning(
                    f"could not load net state dict {e} (or perhaps could only not load pca)",
                )
            if self.net_pose is not None:
                try:
                    self.net_pose.load_state_dict(
                        checkpoint["net_pose_state_dict"],
                        strict=True,
                    )
                except Exception as e:
                    logger.warning(f"could not load net pose state dict {e}")
            try:
                self.meshes.load_state_dict(checkpoint["meshes_feats"], strict=False)
            except Exception as e:
                logger.warning(f"could not load meshes state dict {e}")
        elif path_checkpoint_old is not None:
            fpaths_meshes_old = list(self.config.fpaths_meshes.values())
            meshes_old = Meshes.load_from_files(fpaths_meshes=fpaths_meshes_old)
            verts_count_max = meshes_old.verts_counts_max
            mem_verts_feats_count = len(fpaths_meshes_old) * verts_count_max
            checkpoint = torch.load(Path(path_checkpoint_old), map_location="cuda:0")
            self.net.backbone.net = torch.nn.DataParallel(self.net.backbone.net).cuda()
            self.net.backbone.net.load_state_dict(checkpoint["state"], strict=False)
            self.net.backbone.net = self.net.backbone.net.module
            self.clutter_feats = (
                checkpoint["memory"][mem_verts_feats_count:].clone().detach().cpu()
            )
            # self.clutter_feats = self.clutter_feats.mean(dim=0, keepdim=True)
            self.clutter_feats = torch.nn.Parameter(
                self.clutter_feats.to(device=self.device),
                requires_grad=True,
            )

            verts_feats = []
            map_mesh_id_to_old_id = [
                fpaths_meshes_old.index(fpath_mesh) for fpath_mesh in self.fpaths_meshes
            ]
            for i in range(len(self.fpaths_meshes)):
                mesh_old_id = map_mesh_id_to_old_id[i]
                verts_feats.append(
                    checkpoint["memory"][
                        mesh_old_id
                        * verts_count_max : (mesh_old_id + 1)
                        * verts_count_max
                    ]
                    .clone()
                    .detach()
                    .cpu(),
                )
            self.meshes.set_feats_cat_with_pad(torch.cat(verts_feats, dim=0))

        from od3d.cv.geometry.objects3d.dmtet_x_gaussians import DMTet_x_Gaussians
        from od3d.cv.geometry.objects3d.dmtet import DMTet
        from od3d.cv.geometry.objects3d.flexicubes import Flexicubes

        if (
            isinstance(self.meshes, DMTet_x_Gaussians)
            or isinstance(self.meshes, DMTet)
            or isinstance(self.meshes, Flexicubes)
        ):
            self.meshes.update_dmtet(require_grad=False)

    @property
    def fpath_checkpoint(self):
        return self.logging_dir.joinpath(self.rfpath_checkpoint)

    @property
    def fpath_checkpoint_cat(self):
        return self.logging_dir.joinpath(self.rfpath_checkpoint_cat)

    @property
    def rfpath_checkpoint_cat(self):
        return Path("nemo_cat.ckpt")

    @property
    def rfpath_checkpoint(self):
        return Path("nemo.ckpt")

    def train(
        self,
        datasets_train: Dict[str, OD3D_Dataset],
        datasets_val: Dict[str, OD3D_Dataset],
    ):
        # prevents inheriting broken CUDA context for each worker
        import torch.multiprocessing as mp

        mp.set_start_method("spawn", force=True)

        score_metric_neg = self.config.train.early_stopping_score.startswith("-")
        score_metric_name = (
            self.config.train.early_stopping_score
        )  # "pose/acc_pi18"  # 'pose/acc_pi18' 'pose/acc_pi6'
        if score_metric_neg:
            score_metric_name = score_metric_name[1:]

        score_ckpt_val = -np.inf
        score_latest = -np.inf

        self.save_checkpoint(path_checkpoint=self.fpath_checkpoint)

        if "main" in datasets_val.keys():
            dataset_train_sub = datasets_train["labeled"]
        else:
            dataset_train_sub, dataset_val_sub = datasets_train["labeled"].get_split(
                fraction1=1.0 - self.config.train.val_fraction,
                fraction2=self.config.train.val_fraction,
                split=self.config.train.split,
            )
            datasets_val["main"] = dataset_val_sub
        if (
            self.config.model.backbone.get(
                "pca",
                None,
            )
            is not None
            and self.config.model.backbone.pca.get("enable", False)
            and self.config.model.backbone.pca.get("recalc", True)
        ):
            logger.info("calc pca ...")
            from od3d.cv.cluster.embed import pca
            from od3d.datasets.dataset import OD3D_DATASET_SPLITS

            dataset_pca, _ = dataset_train_sub.get_split(
                fraction1=self.config.model.backbone.pca.get("subset_fraction", 1.0),
                fraction2=1.0
                - self.config.model.backbone.pca.get("subset_fraction", 1.0),
                split=OD3D_DATASET_SPLITS.RANDOM,
            )

            self.net.backbone.set_pca(
                dataset=dataset_pca,
                transform=self.transform_train,
                batch_size=self.config.train.dataloader.batch_size,
                pin_memory=self.config.train.dataloader.pin_memory,
                num_workers=self.config.train.dataloader.num_workers,
                device=self.device,
            )

        # first validation
        if self.config.train.val:
            for dataset_val_key, dataset_val in datasets_val.items():
                results_val = self.test(dataset_val, val=True, multiview=False)
                with torch.no_grad():
                    results_val += self.train_epoch(dataset_val, val=True)
                results_val.log_with_prefix(prefix=f"val/{dataset_val.name}")
                if dataset_val_key == "main":
                    score_latest = results_val[score_metric_name]
                    if score_metric_neg:
                        score_latest = -score_latest
            if not self.config.train.early_stopping or score_latest >= score_ckpt_val:
                score_ckpt_val = score_latest
                self.save_checkpoint(path_checkpoint=self.fpath_checkpoint)
        else:
            self.save_checkpoint(path_checkpoint=self.fpath_checkpoint)

        # self.scheduler.last_epoch
        for epoch in range(self.scheduler.last_epoch, self.config.train.epochs):
            # pseudo
            if self.config.train.pseudo_labels_fraction_per_epoch > 0.0:
                # from od3d.datasets.dataset import OD3D_DATASET_SPLITS
                # dataset_train_sub_sub, _ = dataset_train_sub.get_split(
                #     fraction1=0.1,
                #     fraction2=0.9,
                #     split=OD3D_DATASET_SPLITS.RANDOM,
                # )
                results_train_pseudo_mean, results_train_pseudo = self.test(
                    dataset_train_sub,
                    val=True,
                    multiview=True,
                    return_results_epoch=True,
                )
                results_train_pseudo_mean.log_with_prefix(
                    prefix=f"train_pseudo/{dataset_train_sub.name}",
                )

                tform_obj = tform4x4(
                    inv_tform4x4(results_train_pseudo["cam_tform4x4_obj"]),
                    results_train_pseudo["cam_tform4x4_obj_gt"],
                ).detach()

                del self.pseudo_tform4x4_obj
                self.pseudo_tform4x4_obj = {}
                for b, name_unique in enumerate(results_train_pseudo["name_unique"]):
                    name_seq = "/".join(name_unique.split("/")[:-1])
                    if name_seq not in self.pseudo_tform4x4_obj.keys():
                        self.pseudo_tform4x4_obj[name_seq] = tform_obj[b]

                # results_train_pseudo['cam_tform4x4_obj']
                # results_train_pseudo['cam_tform4x4_obj_gt']
                # results_train_pseudo['name_unique']
                self.pseudo_labels_fraction += (
                    self.config.train.pseudo_labels_fraction_per_epoch
                )
                self.pseudo_labels_fraction = min(self.pseudo_labels_fraction, 1.0)

            results_epoch = self.train_epoch(dataset=dataset_train_sub)
            results_epoch.log_with_prefix("train")
            if (
                self.config.train.val
                and self.config.train.epochs_to_next_test > 0
                and epoch % self.config.train.epochs_to_next_test == 0
            ):
                for dataset_val_key, dataset_val in datasets_val.items():
                    results_val = self.test(dataset_val, val=True, multiview=False)
                    with torch.no_grad():
                        results_val += self.train_epoch(dataset_val, val=True)
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
                    self.save_checkpoint(path_checkpoint=self.fpath_checkpoint)

            elif not self.config.train.val:
                self.save_checkpoint(path_checkpoint=self.fpath_checkpoint)

            if (
                self.config.train.get("epoch_cat_verts_requires_grad_stop", 9999999)
                == self.scheduler.last_epoch
            ):
                self.save_checkpoint(path_checkpoint=self.fpath_checkpoint_cat)

        self.load_checkpoint(path_checkpoint=self.fpath_checkpoint)

    def test(
        self,
        dataset: OD3D_Dataset,
        val=False,
        multiview=True,
        return_results_epoch=False,
    ):
        # note: ensure that checkpoint is saved for checkpointed runs
        if not self.fpath_checkpoint.exists():
            self.save_checkpoint(path_checkpoint=self.fpath_checkpoint)

        logger.info(f"test dataset {dataset.name}")
        self.net.eval()
        if self.net_pose is not None:
            self.net_pose.eval()
        self.meshes.eval()

        if OD3D_FRAME_MODALITIES.KPTS2D_ANNOT in dataset.modalities:
            self.set_kpts(dataset)

        multiview = (
            (isinstance(dataset, CO3D) or isinstance(dataset, Omni6DPose))
            and multiview
            and self.config.multiview.get("enabled", True)
        )

        if not isinstance(dataset, SPair71K):
            dataset.transform = copy.deepcopy(self.transform_test)
        else:
            # dataset.transform = self.net.transform
            dataset.transform = SequentialTransform(
                [
                    OD3D_Transform.create_by_name("crop512"),  # 840 512 504
                    copy.deepcopy(self.net.transform),
                ],
            )
        if not multiview:
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.config.test.dataloader.batch_size,
                shuffle=False,
                collate_fn=dataset.collate_fn,
                num_workers=self.config.test.dataloader.num_workers,
                pin_memory=self.config.test.dataloader.pin_memory,
            )
            logger.info(f"Dataset contains {len(dataset)} frames.")

        else:
            dict_category_sequences = {
                category: list(sequence_dict.keys())
                for category, sequence_dict in dataset.dict_nested_frames.items()
            }
            dataset_sub = dataset.get_subset_by_sequences(
                dict_category_sequences=dict_category_sequences,
                frames_count_max_per_sequence=self.config.multiview.batch_size,
            )

            dataloader = torch.utils.data.DataLoader(
                dataset=dataset_sub,
                batch_size=self.config.multiview.batch_size,
                shuffle=False,
                collate_fn=dataset_sub.collate_fn,
                num_workers=self.config.test.dataloader.num_workers,
                pin_memory=self.config.test.dataloader.pin_memory,
            )
            logger.info(f"Dataset contains {len(dataset_sub)} frames.")

        results_epoch = OD3D_Results(logging_dir=self.logging_dir)
        for i, batch in tqdm(enumerate(iter(dataloader))):
            batch.to(device=self.device)

            if not isinstance(dataset, SPair71K):
                results_batch = self.inference_batch(
                    batch=batch,
                    multiview=multiview,
                    val=val,
                )
            else:
                results_batch = self.inference_batch_corresp(batch=batch)

            results_epoch += results_batch

            if not val and self.config.test.save_results:
                # latent = results_batch['latent'][0]
                f_in_b_id = batch.rgb.shape[0] // 2
                backbone_out, net_out = self.net(
                    batch.rgb[f_in_b_id : f_in_b_id + 1],
                    return_backbone_output=True,
                )
                feats2d_img = net_out.featmap
                logger.info(f"categories {batch.category_id}")
                mesh = self.meshes.get_meshes_with_ids(
                    meshes_ids=batch.category_id[f_in_b_id : f_in_b_id + 1],
                    clone=True,
                )

                if (
                    self.config.train.get("epoch_inst_def_start", 0)
                    <= self.scheduler.last_epoch
                ):
                    instance_deform = self.meshes.get_instance_deform(
                        backbone_out,
                        img_feats_canonical=feats2d_img,
                        objects_ids=batch.category_id[f_in_b_id : f_in_b_id + 1],
                    )
                    mesh.verts += instance_deform.verts_deform[0]
                else:
                    instance_deform = None

                # category -name
                # asset_name = "_".join("/".join(batch.name_unique[f_in_b_id].split("/")[1:-1]).split("_")[1:])
                asset_name = str(
                    batch.sequence[f_in_b_id].first_frame.get_fpaths_meshs("meta")[0],
                ).split("/")[-2]
                fpath_mesh_alpha500 = dataset.path_preprocess.joinpath(
                    "mesh/alpha500/PAM/object_meshes",
                    asset_name,
                    "Aligned.ply",
                )
                mesh_alpha500 = Meshes.read_from_ply_file(fpath=fpath_mesh_alpha500)
                mesh.verts *= (
                    mesh_alpha500.get_range1d().to(mesh.verts.device)
                    / mesh.get_range1d()
                )
                fpath_mesh = dataset.path_preprocess.joinpath(
                    "mesh/common3d/PAM/object_meshes",
                    asset_name,
                    "Aligned.ply",
                )
                fpath_mesh.parent.mkdir(exist_ok=True, parents=True)
                mesh.write_to_file(fpath_mesh)

                results_visual_batch = self.get_results_visual_batch(
                    batch=batch,
                    results_batch=results_batch,
                    config_visualize=self.config.test.visualize,
                )
                results_visual_batch.save_visual(prefix=f"test/{dataset.name}")

        count_pred_frames = len(results_epoch["item_id"])
        logger.info(f"Predicted {count_pred_frames} frames.")
        if not val and self.config.test.save_results:
            results_epoch.save_with_dataset(prefix="test", dataset=dataset)

        if not multiview:
            results_visual = self.get_results_visual(
                results_epoch=results_epoch,
                dataset=dataset,
                config_visualize=self.config.test.visualize,
            )
        else:
            results_visual = self.get_results_visual(
                results_epoch=results_epoch,
                dataset=dataset_sub,
                config_visualize=self.config.test.visualize,
            )

        results_epoch_mean = results_epoch.mean()
        results_epoch_mean += results_visual

        if return_results_epoch:
            return results_epoch_mean, results_epoch
        else:
            return results_epoch_mean

    def train_epoch(self, dataset: OD3D_Dataset, val=False) -> OD3D_Results:
        if not val:
            self.net.train()
            if self.net_pose is not None:
                self.net_pose.train()
            self.meshes.del_pre_rendered()
            self.meshes.train()
        else:
            logger.info(f"test dataset {dataset.name}")
            self.net.eval()
            if self.net_pose is not None:
                self.net_pose.eval()
            self.meshes.eval()

        if (
            self.config.train.get("epoch_cat_verts_requires_grad_stop", 9999999)
            <= self.scheduler.last_epoch
        ):
            self.meshes.set_verts_requires_grad(False)

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
            if not val:
                results_batch.log_with_prefix("train")
            if not val:
                accumulate_steps += 1
                if (
                    accumulate_steps % self.config.train.batch_accumulate_to_next_step
                    == 0
                ):
                    if self.back_propagate:
                        self.optim.step()
                        self.optim.zero_grad()

            results_epoch += results_batch

        if not val:
            self.scheduler.step()
        self.optim.zero_grad()

        # results_epoch.log_dict_to_dir(name=f'train_frames/{dataset.name}')

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

        from od3d.cv.geometry.objects3d.dmtet_x_gaussians import DMTet_x_Gaussians
        from od3d.cv.geometry.objects3d.dmtet import DMTet
        from od3d.cv.geometry.objects3d.flexicubes import Flexicubes

        if (
            isinstance(self.meshes, DMTet_x_Gaussians)
            or isinstance(self.meshes, DMTet)
            or isinstance(self.meshes, Flexicubes)
        ):
            self.meshes.update_dmtet(require_feats_grad=True)

        elif "coarse" in self.config.train.loss.appear.type:
            self.meshes.update_verts_coarse()
        logger.info(f"categories before cuda {batch.category_id}")

        batch.to(device=self.device)

        cam_tform4x4_obj = batch.cam_tform4x4_obj.detach().clone()
        if self.pseudo_labels_fraction > 0.0:
            seq_names = [
                "/".join(name_unique.split("/")[:-1])
                for name_unique in batch.name_unique
            ]
            for b in range(B):
                if torch.rand(1)[0] < self.pseudo_labels_fraction:
                    pseudo_tform_obj_b = (
                        self.pseudo_tform4x4_obj[seq_names[b]]
                        .detach()
                        .clone()
                        .to(cam_tform4x4_obj.device)
                    )

                    if OD3D_FRAME_MODALITIES.MESH in batch.modalities:
                        mesh_verts = (
                            batch.mesh.get_verts_with_mesh_id(
                                mesh_id=batch.category_id[b],
                            )
                            .to(
                                device=self.device,
                            )
                            .clone()
                        )
                        mesh_verts = transf3d_broadcast(
                            pts3d=mesh_verts,
                            transf4x4=pseudo_tform_obj_b,
                        )
                        transl = (
                            -(mesh_verts.max(dim=0)[0] + mesh_verts.min(dim=0)[0]) / 2.0
                        )
                        pseudo_tform_obj_b[:3, 3] += transl

                        mesh_verts = batch.mesh.get_verts_with_mesh_id(
                            mesh_id=batch.category_id[b],
                            clone=True,
                        ).to(
                            device=self.device,
                        )
                        mesh_verts = transf3d_broadcast(
                            pts3d=mesh_verts,
                            transf4x4=pseudo_tform_obj_b,
                        )
                        scale = 1.0 / mesh_verts.abs().max()
                        pseudo_tform_obj_b[:3, :] *= scale

                        mesh_verts = batch.mesh.get_verts_with_mesh_id(
                            mesh_id=batch.category_id[b],
                            clone=True,
                        ).to(
                            device=self.device,
                        )
                        mesh_verts = transf3d_broadcast(
                            mesh_verts,
                            transf4x4=pseudo_tform_obj_b,
                        )
                        batch.mesh.set_verts_with_mesh_id(
                            value=mesh_verts,
                            mesh_id=batch.category_id[b],
                        )

                        if OD3D_FRAME_MODALITIES.PCL in batch.modalities:
                            batch.pcl[b] = transf3d_broadcast(
                                batch.pcl[b].to(self.device),
                                pseudo_tform_obj_b,
                            )
                    else:
                        logger.info("no mesh in batch to normalize object space.")

                    cam_tform4x4_obj[b] = tform4x4(
                        cam_tform4x4_obj[b],
                        inv_tform4x4(pseudo_tform_obj_b),
                    )
                    scale = (
                        cam_tform4x4_obj[b][:3, :3]
                        .norm(dim=-1, keepdim=True)
                        .mean(dim=-2, keepdim=True)
                    )
                    cam_tform4x4_obj[b][:3] = cam_tform4x4_obj[b][:3] / scale

        # if torch.rand(1)[0] < 1.1:
        #     print("blbal")

        # self.pseudo_tform4x4_obj = {}
        # cam_tform4x4_obj = []
        # for b in range(B):
        #     results_inf = self.inference_batch(batch[b])
        #     cam_tform4x4_obj.append(results_inf['cam_tform4x4_obj'])
        # cam_tform4x4_obj = torch.cat(cam_tform4x4_obj, dim=0)
        #

        # logger.info(f"batch.category_id {batch.category_id}")
        # logger.info(f"batch.size {batch.size}")

        # B x F+N x C
        # logger.info(f"batch.size {batch.size}")
        logger.info(f"categories before network {batch.category_id}")

        backbone_out, net_out = self.net(batch.rgb, return_backbone_output=True)
        feats2d_img = net_out.featmap
        logger.info(f"categories {batch.category_id}")

        if (
            self.config.train.get("epoch_inst_def_start", 0)
            <= self.scheduler.last_epoch
        ):
            instance_deform = self.meshes.get_instance_deform(
                backbone_out,
                img_feats_canonical=feats2d_img,
                objects_ids=batch.category_id,
            )
        else:
            instance_deform = None

        # logger.info(f"batch.size {batch.size}")
        loss_app_feats2d_img_mask = torch.ones(
            size=(feats2d_img.shape[0], 1, feats2d_img.shape[2], feats2d_img.shape[3]),
        ).to(device=self.device)

        if self.config.train.loss.appear.use_mask_rgb:
            loss_app_feats2d_img_mask *= resize(
                batch.rgb_mask,
                H_out=feats2d_img.shape[2],
                W_out=feats2d_img.shape[3],
            )

        if self.config.train.loss.appear.use_mask_object:
            loss_app_feats2d_img_mask *= resize(
                batch.mask,
                H_out=feats2d_img.shape[2],
                W_out=feats2d_img.shape[3],
            )

        add_clutter = True

        losses = []
        losses_names = []
        if self.net_pose is not None and self.config.train.loss.pose.weight is not None:
            cam_tform4x4_objs = self.pred_poses(
                backbone_out=backbone_out,
                cam_intr4x4=batch.cam_intr4x4,
                size=batch.size,
                cam_tform4x4_obj=batch.cam_tform4x4_obj,
            )
            from od3d.cv.geometry.transform import matrix_to_rotation_6d

            pred_cam_rot6d_objs = matrix_to_rotation_6d(cam_tform4x4_objs[..., :3, :3])
            gt_cam_rot6d_objs = matrix_to_rotation_6d(
                batch.cam_tform4x4_obj[..., :3, :3],
            )

            pred_cam_transl3d_objs = cam_tform4x4_objs[..., :3, 3]
            gt_cam_transl3d_objs = batch.cam_tform4x4_obj[..., :3, 3]

            pred_othant = pred_cam_rot6d_objs[
                ...,
                self.net_pose_rotations_othant,
            ].sign()
            gt_othant = gt_cam_rot6d_objs[..., self.net_pose_rotations_othant].sign()
            mask_othant_equal = (pred_othant == gt_othant[:, None]).all(dim=-1)
            pred_cam_rot6d_objs_sel = pred_cam_rot6d_objs[mask_othant_equal]
            pred_cam_transl3d_objs_sel = pred_cam_transl3d_objs[mask_othant_equal]
            loss_pose = ((pred_cam_rot6d_objs_sel - gt_cam_rot6d_objs) ** 2).mean() + (
                (pred_cam_transl3d_objs_sel - gt_cam_transl3d_objs) ** 2
            ).mean()

            loss_pose = loss_pose * self.config.train.loss.pose.weight
            results_batch["loss_pose"] = loss_pose[None,].detach()
            losses.append(loss_pose)
            losses_names.append("pose")

        if (
            self.config.train.loss.appear.weight > 0.0
            and self.config.train.get("epoch_app_start", 0) <= self.scheduler.last_epoch
        ):
            if "cross_entropy" in self.config.train.loss.appear.type:
                add_other_objects = self.config.train.get("inter_class_loss", True)
                (
                    labels,
                    labels_mask,
                    noise_pxl2d,
                    sim,
                    feats,
                ) = self.meshes.get_label_and_sim_feats2d_img_to_all(
                    feats2d_img=feats2d_img,
                    imgs_sizes=batch.size,
                    cams_tform4x4_obj=cam_tform4x4_obj,
                    cams_intr4x4=batch.cam_intr4x4,
                    objects_ids=batch.category_id,
                    broadcast_batch_and_cams=False,
                    feats2d_img_mask=loss_app_feats2d_img_mask,
                    down_sample_rate=self.down_sample_rate,
                    add_clutter=add_clutter,
                    add_other_objects=add_other_objects,
                    sample_clutter_count=self.config.num_noise,
                    dense=self.config.train.loss.appear.dense_loss,
                    smooth_labels="smooth" in self.config.train.loss.appear.type,
                    coarse_labels="coarse" in self.config.train.loss.appear.type,
                    sim_temp=self.config.train.T,
                    return_feats=True,
                    instance_deform=instance_deform,
                    detach_objects=self.config.train.loss.appear.dense_detach_geo,
                    detach_deform=self.config.train.loss.appear.dense_detach_geo,
                )

                if labels.isnan().any():
                    logger.error("labels contains nan")
                if labels_mask is not None and labels_mask.isnan().any():
                    logger.error("labels_mask contains nan")
                if noise_pxl2d is not None and noise_pxl2d.isnan().any():
                    logger.error("noise_pxl2d contains nan")
                if sim.isnan().any():
                    logger.error("sim contains nan")
                if feats.isnan().any():
                    logger.error("feats contains nan")

                if self.config.train.bank_feats_update == "moving_average":
                    assert (
                        not self.config.train.loss.appear.dense_loss
                        and not "smooth" in self.config.train.loss.appear.type
                        and not "coarse" in self.config.train.loss.appear.type
                    )
                    self.meshes.update_feats_moving_average(
                        labels=labels,
                        labels_mask=labels_mask,
                        feats=feats,
                        objects_ids=batch.category_id,
                        alpha=self.config.train.alpha,
                        add_clutter=add_clutter,
                        add_other_objects=add_other_objects,
                    )

                if self.config.train.bank_feats_update == "average":
                    assert (
                        not self.config.train.loss.appear.dense_loss
                        and not "smooth" in self.config.train.loss.appear.type
                        and not "coarse" in self.config.train.loss.appear.type
                    )
                    self.meshes.update_feats_total_average(
                        labels=labels,
                        labels_mask=labels_mask,
                        feats=feats,
                        objects_ids=batch.category_id,
                        add_clutter=add_clutter,
                        add_other_objects=add_other_objects,
                    )

                if labels.dim() == 2:
                    labels = labels[labels_mask]
                elif labels.dim() == 3:
                    labels = labels.permute(0, 2, 1)
                    if labels_mask is not None:
                        labels = labels[labels_mask]

                if sim.dim() == 3:
                    sim_batchwise = (sim.max(dim=1).values * labels_mask).flatten(
                        1,
                    ).sum(
                        dim=-1,
                    ) / (
                        labels_mask.flatten(1).sum(dim=-1) + 1e-6
                    ).detach()
                    sim = sim.permute(0, 2, 1)[labels_mask]
                else:
                    sim_batchwise = sim.max(dim=1).values.flatten(1).mean(dim=-1)

            elif (
                "sim_max" in self.config.train.loss.appear.type
                or "sim_neg_mse_max" in self.config.train.loss.appear.type
            ):
                sim_batchwise = self.meshes.get_sim_render(
                    feats2d_img=feats2d_img,
                    cams_tform4x4_obj=cam_tform4x4_obj,
                    cams_intr4x4=batch.cam_intr4x4,
                    objects_ids=batch.category_id,
                    broadcast_batch_and_cams=False,
                    down_sample_rate=self.down_sample_rate,
                    feats2d_img_mask=loss_app_feats2d_img_mask,
                    allow_clutter=False,
                    return_sim_pxl=False,
                    add_clutter=True,
                    add_other_objects=False,
                    temp=self.config.train.T,
                    instance_deform=instance_deform,
                    use_neg_mse="sim_neg_mse_max" in self.config.train.loss.appear.type,
                )
                sim = sim_batchwise.mean()
                noise_pxl2d = None
            else:
                raise ValueError(f"Unknown loss {self.config.train.loss.appear.type}")

            if "cross_entropy" in self.config.train.loss.appear.type:
                if sim.dim() == 4:
                    sim = (
                        self.loss_appear_dropout(sim.flatten(2).permute(0, 2, 1))
                        .permute(0, 2, 1)
                        .reshape(*sim.shape)
                    )
                elif sim.dim() == 2:
                    sim = self.loss_appear_dropout(sim)
                else:
                    raise NotImplementedError

                loss_app = self.criterion(sim, labels)
            elif (
                "sim_max" in self.config.train.loss.appear.type
                or "sim_neg_mse_max" in self.config.train.loss.appear.type
            ):
                loss_app = -sim
            else:
                raise ValueError(f"Unknown loss {self.config.train.loss.appear.type}")

            loss_app = loss_app * self.config.train.loss.appear.weight
            results_batch["loss_app"] = loss_app[None,].detach()
            losses.append(loss_app)
            losses_names.append("app")
        else:
            noise_pxl2d = None
            sim_batchwise = None
            loss_app = (
                torch.ones([]).to(device=self.device)
                * self.config.train.loss.appear.weight
                * 10.0
            )
            results_batch["loss_app"] = loss_app[None,].detach()
            losses.append(loss_app)
            losses_names.append("app")

        if self.config.train.get("replace_mask_with_rendered_mask", False):
            batch.mask = (
                batch.mesh.render(
                    modalities="mask",
                    cams_tform4x4_obj=batch.cam_tform4x4_obj,
                    cams_intr4x4=batch.cam_intr4x4,
                    imgs_sizes=batch.size,
                    broadcast_batch_and_cams=False,
                )
                > 0.5
            )

        # from od3d.cv.visual.show import show_scene
        # show_scene(meshes=batch.mesh[0:2])

        geo_mask_vals = self.get_geo_mask_metrics(
            batch=batch,
            cam_tform4x4_obj=cam_tform4x4_obj,
            instance_deform=instance_deform,
            metrics=["iou", "mse", "overlap_dt", "overlap_inv_dt"],
        )

        if self.config.train.loss.geo.mask.weight > 0.0:
            loss_geo_mask = -geo_mask_vals["iou"]
            loss_geo_mask = (
                loss_geo_mask.mean() * self.config.train.loss.geo.mask.weight
            )

            results_batch["loss_geo_mask"] = loss_geo_mask[None,].detach()
            losses.append(loss_geo_mask)
            losses_names.append("geo_mask")

        if self.config.train.loss.geo.mask_mse.weight > 0.0:
            loss_geo_mask_mse = geo_mask_vals["mse"]
            loss_geo_mask_mse = (
                loss_geo_mask_mse.mean() * self.config.train.loss.geo.mask_mse.weight
            )
            results_batch["loss_geo_mask_mse"] = loss_geo_mask_mse[None,].detach()
            losses.append(loss_geo_mask_mse)
            losses_names.append("geo_mask_mse")

        if (
            self.config.train.loss.geo.mask_dt.weight > 0.0
            and "overlap_dt" in geo_mask_vals
        ):
            loss_geo_mask_dt = -geo_mask_vals["overlap_dt"]
            loss_geo_mask_dt = (
                loss_geo_mask_dt.mean() * self.config.train.loss.geo.mask_dt.weight
            )
            results_batch["loss_geo_mask_dt"] = loss_geo_mask_dt[None,].detach()
            losses.append(loss_geo_mask_dt)
            losses_names.append("geo_mask_dt")

        if (
            self.config.train.loss.geo.mask_inv_dt.weight > 0.0
            and "overlap_inv_dt" in geo_mask_vals
        ):
            loss_geo_mask_inv_dt = -geo_mask_vals["overlap_inv_dt"]
            loss_geo_mask_inv_dt = (
                loss_geo_mask_inv_dt.mean()
                * self.config.train.loss.geo.mask_inv_dt.weight
            )
            results_batch["loss_geo_mask_inv_dt"] = loss_geo_mask_inv_dt[None,].detach()
            losses.append(loss_geo_mask_inv_dt)
            losses_names.append("geo_mask_inv_dt")

        if self.config.train.loss.geo.rec.weight > 0.0:
            loss_geo = self.get_geo_loss(
                batch=batch,
                instance_deform=instance_deform,
                rec_type=self.config.train.loss.geo.rec.type,
            )
            loss_geo = loss_geo.mean() * self.config.train.loss.geo.rec.weight
            results_batch["loss_geo"] = loss_geo[None,].detach()
            losses.append(loss_geo)
            losses_names.append("geo")

            """
            self.config.train.loss.appear.type
            self.config.train.loss.appear.weight
            self.config.train.loss.geo.rec.weight
            self.config.train.loss.geo.smooth.weight
            self.config.train.loss.geo.deform_reg.weight
            appear:
              type: cross_entropy # cross_entropy, nll_softmax, nll_clip, nll_affine_to_prob, l2, l2_squared
              dense_loss: False
              smooth_loss: False
              weight: 0.01
            geo:
              rec:
                weight: 1.
              smooth:
                weight: 1.
              deform_reg:
                weight: 0.1
            """
        from od3d.cv.geometry.objects3d.dmtet_x_gaussians import DMTet_x_Gaussians
        from od3d.cv.geometry.objects3d.dmtet import DMTet
        from od3d.cv.geometry.objects3d.flexicubes import Flexicubes

        if self.config.train.loss.geo.smooth.weight > 0.0 and (
            instance_deform is not None
            or not (
                isinstance(self.meshes, DMTet_x_Gaussians)
                or isinstance(self.meshes, DMTet)
                or isinstance(self.meshes, Flexicubes)
            )
        ):
            loss_geo_smooth = None
            if instance_deform is not None:
                loss_geo_smooth = self.meshes.get_geo_smooth_loss(
                    objects_ids=batch.category_id,
                    instance_deform=instance_deform,
                    detach_objects_verts=True,
                )
            if not (
                isinstance(self.meshes, DMTet_x_Gaussians)
                or isinstance(self.meshes, DMTet)
                or isinstance(self.meshes, Flexicubes)
            ):
                loss_geo_smooth_cat = self.meshes.get_geo_smooth_loss(
                    objects_ids=batch.category_id,
                    instance_deform=None,
                )
                if loss_geo_smooth is None:
                    loss_geo_smooth = loss_geo_smooth_cat
                else:
                    loss_geo_smooth += loss_geo_smooth_cat
                    loss_geo_smooth /= 2.0

            loss_geo_smooth = (
                loss_geo_smooth.mean() * self.config.train.loss.geo.smooth.weight
            )
            results_batch["loss_geo_smooth"] = loss_geo_smooth[None,].detach()
            losses.append(loss_geo_smooth)
            losses_names.append("geo_smooth")

        if self.config.train.loss.geo.sdf_reg.weight > 0.0 and (
            isinstance(self.meshes, DMTet_x_Gaussians)
            or isinstance(self.meshes, DMTet)
            or isinstance(self.meshes, Flexicubes)
        ):
            loss_geo_sdf_reg = self.meshes.get_geo_sdf_reg_loss(
                objects_ids=batch.category_id,
            )
            loss_geo_sdf_reg = (
                loss_geo_sdf_reg.mean() * self.config.train.loss.geo.sdf_reg.weight
            )
            results_batch["loss_geo_sdf_reg"] = loss_geo_sdf_reg[None,].detach()
            losses.append(loss_geo_sdf_reg)
            losses_names.append("geo_sdf_reg")

        if self.config.train.loss.geo.deform_reg.weight > 0.0:
            if instance_deform is not None:
                loss_deform_reg = instance_deform.verts_deform.norm(p=2, dim=-1).mean()
            else:
                loss_deform_reg = torch.zeros([]).to(device=self.device)

            loss_deform_reg = (
                loss_deform_reg * self.config.train.loss.geo.deform_reg.weight
            )
            results_batch["loss_deform_reg"] = loss_deform_reg[None,].detach()
            losses.append(loss_deform_reg)
            losses_names.append("geo_deform_reg")

        if (
            self.config.train.loss.geo.deform_latent_reg.weight > 0.0
            and instance_deform is not None
            and instance_deform.latent is not None
        ):
            if self.config.train.loss.geo.deform_latent_reg.type == "unit":
                loss_geo_deform_latent_reg = (
                    instance_deform.latent.norm(dim=-1) - 1.0
                ).abs() ** 2
                loss_geo_deform_latent_reg = (
                    loss_geo_deform_latent_reg.mean()
                    * self.config.train.loss.geo.deform_latent_reg.weight
                )

            elif self.config.train.loss.geo.deform_latent_reg.type == "kl":
                if (
                    instance_deform.latent_mu is None
                    or instance_deform.latent_logvar is None
                ):
                    logger.warning("latent distribution is missing to calculate kl")
                    loss_geo_deform_latent_reg = torch.zeros([]).to(device=self.device)
                else:
                    loss_geo_deform_latent_reg = (
                        -0.5
                        * (
                            1
                            + instance_deform.latent_logvar
                            - instance_deform.latent_mu.pow(2)
                            - instance_deform.latent_logvar.exp()
                        ).mean()
                        * self.config.train.loss.geo.deform_latent_reg.weight
                    )
                    # Compute the Kullback-Leibler divergence between the learned latent variable distribution and a standard Gaussian distribution
                    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            results_batch["loss_geo_deform_latent_reg"] = loss_geo_deform_latent_reg[
                None,
            ].detach()
            losses.append(loss_geo_deform_latent_reg)
            losses_names.append("geo_deform_latent_reg")

        if self.config.train.loss.geo.deform_smooth_reg.weight > 0.0:
            if instance_deform is not None:
                loss_deform_smooth_reg = self.meshes.get_geo_deform_smooth_loss(
                    objects_ids=batch.category_id,
                    instance_deform=instance_deform,
                )  # .mean()
            else:
                loss_deform_smooth_reg = torch.zeros([]).to(device=self.device)

            loss_deform_smooth_reg = (
                loss_deform_smooth_reg
                * self.config.train.loss.geo.deform_smooth_reg.weight
            )
            results_batch["loss_deform_smooth_reg"] = loss_deform_smooth_reg[
                None,
            ].detach()
            losses.append(loss_deform_smooth_reg)
            losses_names.append("geo_deform_smooth_reg")

        if len(losses) > 0:
            for l in range(len(losses)):
                if losses[l].isnan().any():
                    logger.warning(
                        f"Loss {losses_names[l]} contains NaNs. Setting it to zero.",
                    )
                    losses[l][:] = 0.0

            loss = torch.stack(losses).sum(dim=0)
            loss = loss / self.config.train.batch_accumulate_to_next_step
            if self.back_propagate and loss.requires_grad:
                loss.backward()
        else:
            loss = torch.zeros([]).to(device=self.device)

        logger.info(f"loss {loss.item()}")

        if sim_batchwise is not None:
            sim_batchwise = sim_batchwise.detach()
            results_batch["sim"] = sim_batchwise

        if noise_pxl2d is not None:
            noise_pxl2d = noise_pxl2d.detach()
        results_batch["noise2d"] = noise_pxl2d
        results_batch["loss"] = loss[None,].detach()
        results_batch["item_id"] = batch.item_id
        results_batch["name_unique"] = batch.name_unique
        results_batch["gt_cam_tform4x4_obj"] = cam_tform4x4_obj.detach()
        if self.config.train.bank_feats_update == "average":
            result_visual = OD3D_Results(logging_dir=self.logging_dir)
            bar_image = show_bar_chart(
                int(self.meshes.feats_objects.shape[0]),
                self.meshes.feats_total_count[: self.meshes.feats_objects.shape[0]],
                pts2d_colors=self.meshes.feats_rgb_object_id,
                return_visualization=True,
            )
            bar_image_wandb = image_as_wandb_image(
                bar_image,
                caption=f"number of vertices seen in epoch",
            )
            result_visual["vertices_count"] = bar_image_wandb
            result_visual.log_with_prefix(prefix=f"train/visual")

        return results_batch

    def get_geo_mask_metrics(
        self,
        batch,
        cam_tform4x4_obj=None,
        pred_masks=None,
        instance_deform=None,
        metrics=["iou"],
    ):
        B = len(batch)
        vals = {}
        if pred_masks is None:
            logger.info(f"size {batch.size}")
            pred_masks = self.meshes.render(
                cams_tform4x4_obj=cam_tform4x4_obj,
                cams_intr4x4=batch.cam_intr4x4,
                imgs_sizes=batch.size,
                objects_ids=batch.category_id,
                modalities=PROJECT_MODALITIES.MASK,
                instance_deform=instance_deform,
                down_sample_rate=self.down_sample_rate,
            )

        gt_masks = resize(
            (batch.mask > 0.5) * 1.0,
            H_out=pred_masks.shape[-2],
            W_out=pred_masks.shape[-1],
        )
        if OD3D_FRAME_MODALITIES.RGB_MASK in batch.modalities:
            rgb_mask_resized = resize(
                batch.rgb_mask,
                H_out=pred_masks.shape[-2],
                W_out=pred_masks.shape[-1],
                mode="nearest_v2",
            )
            gt_masks *= rgb_mask_resized
            pred_masks *= rgb_mask_resized

        for metric in metrics:
            if metric == "iou":
                iou = (gt_masks * pred_masks).flatten(1).sum(dim=-1) / (
                    (gt_masks * pred_masks)
                    + ((1.0 - gt_masks) * pred_masks)
                    + ((1.0 - pred_masks) * gt_masks)
                ).detach().flatten(1).sum(dim=-1) + 1e-10
                vals[metric] = iou

            elif metric == "iou_amodal":
                gt_masks_amod = batch.mesh.render(
                    cams_tform4x4_obj=batch.cam_tform4x4_obj,
                    cams_intr4x4=batch.cam_intr4x4,
                    imgs_sizes=batch.size,
                    modalities=PROJECT_MODALITIES.MASK,
                    down_sample_rate=self.down_sample_rate,
                )
                if OD3D_FRAME_MODALITIES.RGB_MASK in batch.modalities:
                    gt_masks_amod *= rgb_mask_resized
                iou = (gt_masks_amod * pred_masks).flatten(1).sum(dim=-1) / (
                    (gt_masks_amod * pred_masks)
                    + ((1.0 - gt_masks_amod) * pred_masks)
                    + ((1.0 - pred_masks) * gt_masks_amod)
                ).detach().flatten(1).sum(dim=-1) + 1e-10
                vals[metric] = iou

            elif metric == "mse":
                mse = ((gt_masks - pred_masks) ** 2).flatten(1).mean(dim=-1)
                vals[metric] = mse

            elif (
                metric == "overlap_dt"
                and OD3D_FRAME_MODALITIES.MASK_DT in batch.modalities
            ):
                gt_masks_dt = resize(
                    batch.mask_dt,
                    H_out=pred_masks.shape[-2],
                    W_out=pred_masks.shape[-1],
                )
                if OD3D_FRAME_MODALITIES.RGB_MASK in batch.modalities:
                    gt_masks_dt *= rgb_mask_resized
                iou = (gt_masks_dt * pred_masks).flatten(1).mean(dim=-1)
                vals[metric] = iou

            elif (
                metric == "overlap_inv_dt"
                and OD3D_FRAME_MODALITIES.MASK_INV_DT in batch.modalities
            ):
                gt_masks_inv_dt = resize(
                    batch.mask_inv_dt,
                    H_out=pred_masks.shape[-2],
                    W_out=pred_masks.shape[-1],
                )
                if OD3D_FRAME_MODALITIES.RGB_MASK in batch.modalities:
                    gt_masks_inv_dt *= rgb_mask_resized
                pred_masks_inv = 1.0 - pred_masks.clone()
                iou = (gt_masks_inv_dt * pred_masks_inv).flatten(1).mean(dim=-1)
                vals[metric] = iou

        if len(metrics) == 1:
            return vals[metrics[0]]
        else:
            return vals

    def get_geo_loss(
        self,
        batch,
        instance_deform=None,
        rec_type="cd_mesh",
        gt_tform_obj=None,
    ):
        # types: cd_mesh, cd_pcl, pf_mesh, pf_pcl
        if (
            rec_type == "cd_mesh"
            or rec_type == "pf_mesh"
            or rec_type == "pf_mesh_v2"
            or rec_type == "pf_mesh_v2_cd_mesh"
        ) and not OD3D_FRAME_MODALITIES.MESH in batch.modalities:
            loss_geo = torch.zeros(len(batch)).to(device=self.device)
        elif (
            rec_type == "cd_pcl"
            or rec_type == "pf_pcl"
            or rec_type == "pf_pcl_v2"
            or rec_type == "pf_pcl_v2_cd_pcl"
        ) and not OD3D_FRAME_MODALITIES.PCL in batch.modalities:
            loss_geo = torch.zeros(len(batch)).to(device=self.device)

        else:
            # BxVx3
            pred_meshes_verts = self.meshes.get_verts_stacked_with_mesh_ids(
                batch.category_id,
            ).to(device=self.device)

            if (
                rec_type == "pf_mesh"
                or rec_type == "pf_mesh_v2"
                or rec_type == "pf_mesh_v2_cd_mesh"
                or rec_type == "pf_pcl"
                or rec_type == "pf_pcl_v2"
                or rec_type == "pf_pcl_v2_cd_pcl"
            ):
                pred_meshes_faces = self.meshes.get_faces_stacked_with_mesh_ids(
                    batch.category_id,
                ).to(device=self.device)
                pred_meshes_faces_mask = (
                    self.meshes.get_faces_stacked_mask_with_mesh_ids(
                        batch.category_id,
                    ).to(device=self.device)
                )

            if instance_deform is not None:
                pred_meshes_verts += instance_deform.verts_deform

            if gt_tform_obj is not None:
                pred_meshes_verts = transf3d_broadcast(
                    pts3d=pred_meshes_verts,
                    transf4x4=gt_tform_obj[:, None],
                )

            pred_meshes_verts_mask = self.meshes.get_verts_stacked_mask_with_mesh_ids(
                batch.category_id,
            ).to(
                device=self.device,
            )

            if (
                rec_type == "cd_mesh"
                or rec_type == "pf_mesh"
                or rec_type == "pf_mesh_v2"
                or rec_type == "pf_mesh_v2_cd_mesh"
            ) and OD3D_FRAME_MODALITIES.MESH in batch.modalities:
                # pred_ranges = pred_meshes_verts.max(dim=1)[0] - pred_meshes_verts.min(dim=1)[0]
                gt_meshes_verts = batch.mesh.get_verts_stacked_with_mesh_ids().to(
                    device=self.device,
                )
                gt_meshes_verts_mask = (
                    batch.mesh.get_verts_stacked_mask_with_mesh_ids().to(
                        device=self.device,
                    )
                )

                if rec_type == "cd_mesh":
                    loss_geo = batch_chamfer_distance(
                        pred_meshes_verts,
                        gt_meshes_verts,
                        pred_meshes_verts_mask,
                        gt_meshes_verts_mask,
                    )
                elif rec_type == "pf_mesh":
                    loss_geo = batch_point_face_distance(
                        verts1=pred_meshes_verts,
                        faces1=pred_meshes_faces,
                        faces1_mask=pred_meshes_faces_mask,
                        pts2=gt_meshes_verts,
                        verts1_mask=pred_meshes_verts_mask,
                        pts2_mask=gt_meshes_verts_mask,
                    )
                elif rec_type == "pf_mesh_v2":
                    loss_geo = batch_point_face_distance_v2(
                        verts1=pred_meshes_verts,
                        faces1=pred_meshes_faces,
                        faces1_mask=pred_meshes_faces_mask,
                        pts2=gt_meshes_verts,
                        verts1_mask=pred_meshes_verts_mask,
                        pts2_mask=gt_meshes_verts_mask,
                    )
                elif rec_type == "pf_mesh_v2_cd_mesh":
                    loss_geo = batch_chamfer_distance(
                        gt_meshes_verts,
                        pred_meshes_verts,
                        gt_meshes_verts_mask,
                        pred_meshes_verts_mask,
                        only_pts2_nn=True,
                        uniform_weight_pts1=False,
                    )
                    loss_geo += batch_point_face_distance_v2(
                        verts1=pred_meshes_verts,
                        faces1=pred_meshes_faces,
                        faces1_mask=pred_meshes_faces_mask,
                        pts2=gt_meshes_verts,
                        verts1_mask=pred_meshes_verts_mask,
                        pts2_mask=gt_meshes_verts_mask,
                    )

            elif (
                rec_type == "cd_pcl"
                or rec_type == "pf_pcl"
                or rec_type == "pf_pcl_v2"
                or rec_type == "pf_pcl_v2_cd_pcl"
            ) and OD3D_FRAME_MODALITIES.PCL in batch.modalities:
                B = len(batch.pcl)
                N = max([batch.pcl[b].shape[0] for b in range(B)])
                gt_pts3d = torch.zeros((B, N, 3)).to(device=self.device)
                gt_pts3d_mask = torch.zeros((B, N), dtype=bool).to(device=self.device)
                for b in range(B):
                    N_b = batch.pcl[b].shape[0]
                    gt_pts3d[b, :N_b] = batch.pcl[b]
                    gt_pts3d_mask[b, :N_b] = True

                # from od3d.cv.visual.show import show_scene
                # show_scene(pts3d=[pred_meshes_verts[0][pred_meshes_verts_mask[0]],gt_pts3d[0][gt_pts3d_mask[0]]])
                if rec_type == "cd_pcl":
                    loss_geo = batch_chamfer_distance(
                        pred_meshes_verts,
                        gt_pts3d,
                        pred_meshes_verts_mask,
                        gt_pts3d_mask,
                    )
                    # from od3d.cv.visual.show import show_scene
                    # show_scene(pts3d=[pred_meshes_verts[0], gt_pts3d[0]])
                elif rec_type == "pf_pcl":
                    loss_geo = batch_point_face_distance(
                        verts1=pred_meshes_verts,
                        faces1=pred_meshes_faces,
                        faces1_mask=pred_meshes_faces_mask,
                        pts2=gt_pts3d,
                        verts1_mask=pred_meshes_verts_mask,
                        pts2_mask=gt_pts3d_mask,
                    )
                elif rec_type == "pf_pcl_v2":
                    loss_geo = batch_point_face_distance_v2(
                        verts1=pred_meshes_verts,
                        faces1=pred_meshes_faces,
                        faces1_mask=pred_meshes_faces_mask,
                        pts2=gt_pts3d,
                        verts1_mask=pred_meshes_verts_mask,
                        pts2_mask=gt_pts3d_mask,
                    )
                elif rec_type == "pf_pcl_v2_cd_pcl":
                    loss_geo = batch_chamfer_distance(
                        gt_pts3d,
                        pred_meshes_verts,
                        gt_pts3d_mask,
                        pred_meshes_verts_mask,
                        only_pts2_nn=True,
                        uniform_weight_pts1=False,
                    )
                    loss_geo += batch_point_face_distance_v2(
                        verts1=pred_meshes_verts,
                        faces1=pred_meshes_faces,
                        faces1_mask=pred_meshes_faces_mask,
                        pts2=gt_pts3d,
                        verts1_mask=pred_meshes_verts_mask,
                        pts2_mask=gt_pts3d_mask,
                    )
            else:
                logger.warning("should not end here, but set geo loss to zero.")
                loss_geo = torch.zeros(len(batch)).to(device=self.device)
        return loss_geo

    """
    def get_sim_cam_tform4x4_obj(self, batch, cam_intr4x4, cam_tform4x4_obj, broadcast_batch_and_cams=False):
        with torch.no_grad():
            net_out = self.net(batch.rgb)
            net_feats2d = net_out.featmap
            mesh_feats2d_rendered = self.meshes.render_feats(cams_tform4x4_obj=cam_tform4x4_obj,
                                                             cams_intr4x4=cam_intr4x4,
                                                             imgs_sizes=batch.size, meshes_ids=batch.category_id,
                                                             down_sample_rate=self.down_sample_rate,
                                                             broadcast_batch_and_cams=broadcast_batch_and_cams)

            sim = self.get_sim_feats2d_net_and_rendered(feats2d_net=net_feats2d, feats2d_rendered=mesh_feats2d_rendered)
        return sim
    """

    def inference_batch_corresp(self, batch):
        # batch.visualize()
        results = OD3D_Results(logging_dir=self.logging_dir)

        B = len(batch.rgbs)
        backbone_outs = []
        net_outs = []

        from od3d.cv.select import (
            batched_indexMD_select,
            batched_argminMD_select,
            batched_index_in_bounds,
        )

        batch_categories = list({cat for cat in batch.category})

        results[f"kpts2d_acc/pck01"] = []
        results[f"kpts2d_acc/pck005"] = []
        results[f"kpts2d_acc/pck001"] = []

        for cat in batch_categories:
            results[f"kpts2d_acc/{cat}_pck01"] = []
            results[f"kpts2d_acc/{cat}_pck005"] = []
            results[f"kpts2d_acc/{cat}_pck001"] = []

        rgbs = torch.stack(
            [
                torch.stack([batch.rgbs[b][f] for f in range(2)], dim=0)
                for b in range(B)
            ],
            dim=0,
        ).detach()
        with torch.no_grad():
            backbone_outs, net_outs = self.net(
                rgbs.to(self.device).reshape(B * 2, *rgbs.shape[2:]),
                return_backbone_output=True,
            )
            net_outs = net_outs.featmap.detach().reshape(
                B,
                2,
                *net_outs.featmap.shape[1:],
            )
            backbone_outs = resize(
                backbone_outs.featmaps[-1].detach(),
                H_out=net_outs.shape[-2],
                W_out=net_outs.shape[-1],
            )
            backbone_outs = backbone_outs.reshape(B, 2, *backbone_outs.shape[1:])

        net_outs = net_outs / (net_outs.norm(dim=-3, keepdim=True) + 1e-10)
        backbone_outs = backbone_outs / (
            backbone_outs.norm(dim=-3, keepdim=True) + 1e-10
        )
        net_target_feats_pxl2d_all = []

        for b in range(B):
            # backbone_outs.append([])
            # net_outs.append([])
            # for f in range(2):
            #     with torch.no_grad():
            #         backbone_out, net_out = self.net(batch.rgbs[b][f][None,].to(self.device), return_backbone_output=True)
            #         backbone_outs[b].append(backbone_out.featmaps[-1][0].detach())
            #         net_outs[b].append(net_out.featmap[0].detach())

            backbone_query_keypoints2d = (
                (batch.kpts2d_annots[b][0].clone() / self.down_sample_rate)
                .long()
                .to(self.device)
            )
            net_query_keypoints2d = (
                (batch.kpts2d_annots[b][0].clone() / self.down_sample_rate)
                .long()
                .to(self.device)
            )
            bH, bW = backbone_outs[b][0].shape[1:]
            nH, nW = net_outs[b][0].shape[1:]

            backbone_query_keypoints2d = batched_index_in_bounds(
                indexMD=backbone_query_keypoints2d,
                bounds=[bW - 1, bH - 1],
            )
            net_query_keypoints2d = batched_index_in_bounds(
                indexMD=net_query_keypoints2d,
                bounds=[nW - 1, nH - 1],
            )

            backbone_query_feats = batched_indexMD_select(
                inputMD=backbone_outs[b][0],
                indexMD=backbone_query_keypoints2d,
                dims=[2, 1],
            )
            net_query_feats = batched_indexMD_select(
                inputMD=net_outs[b][0],
                indexMD=net_query_keypoints2d,
                dims=[2, 1],
            )

            backbone_sim_feats = torch.einsum(
                "kf,fhw->khw",
                backbone_query_feats,
                backbone_outs[b][1],
            )
            net_sim_feats = torch.einsum("kf,fhw->khw", net_query_feats, net_outs[b][1])

            net_sim_feats = (
                (1.0 - self.config.inference.feats_corresp_alpha) * backbone_sim_feats
                + self.config.inference.feats_corresp_alpha * net_sim_feats
            )

            # from od3d.cv.visual.draw import draw_pixels
            # img_gt1 = draw_pixels(batch.rgbs[b][0], pxls=batch.kpts2d_annots[b][0][:1].to(self.device))
            # img_gt2 = draw_pixels(batch.rgbs[b][1], pxls=batch.kpts2d_annots[b][1][:1].to(self.device))
            # net_sim_feats_1 = 255 * (net_sim_feats[:1] - net_sim_feats[:1].min()) / ((net_sim_feats[:1].max() - net_sim_feats[:1].min()) + 1e-10)
            # img_sim = resize(net_sim_feats_1, H_out=img_gt1.shape[-2], W_out=img_gt1.shape[-1]).expand_as(img_gt1).to(device=img_gt1.device)
            # from od3d.cv.visual.show import show_imgs
            # show_imgs([img_gt1, img_gt2, img_sim])

            # backbone_target_feats_pxl2d = batched_argminMD_select(inputMD=-backbone_sim_feats, dims=[2, 1])
            # backbone_target_feats_pxl2d = backbone_target_feats_pxl2d * self.down_sample_rate + (self.down_sample_rate - 1.) * 0.5

            net_target_feats_pxl2d = batched_argminMD_select(
                inputMD=-net_sim_feats,
                dims=[2, 1],
            )
            net_target_feats_pxl2d = (
                net_target_feats_pxl2d * self.down_sample_rate
                + (self.down_sample_rate - 1.0) * 0.5
            )

            # backbone_err = torch.norm(batch.kpts2d_annots[b][1].to(self.device) - backbone_target_feats_pxl2d, dim=1)
            net_err = torch.norm(
                batch.kpts2d_annots[b][1].to(self.device) - net_target_feats_pxl2d,
                dim=1,
            )

            net_target_feats_pxl2d_all.append(net_target_feats_pxl2d.detach())

            # from od3d.cv.visual.draw import draw_pixels
            # img_gt = draw_pixels(batch.rgbs[b][1], pxls=batch.kpts2d_annots[b][1].to(self.device))
            # img_pred = draw_pixels(batch.rgbs[b][1], pxls=net_target_feats_pxl2d)
            # from od3d.cv.visual.show import show_imgs
            # show_imgs([img_gt, img_pred])

            target_bbox = batch.bboxs[b][1]  # batch.bboxs[b][1]
            threshold = max(
                target_bbox[3] - target_bbox[1],
                target_bbox[2] - target_bbox[0],
            )
            results[f"kpts2d_acc/{batch.category[b]}_pck01"].append(
                (net_err < 0.1 * threshold).to(torch.float),
            )
            results[f"kpts2d_acc/{batch.category[b]}_pck005"].append(
                (net_err < 0.05 * threshold).to(torch.float),
            )
            results[f"kpts2d_acc/{batch.category[b]}_pck001"].append(
                (net_err < 0.01 * threshold).to(torch.float),
            )

            results[f"kpts2d_acc/pck01"].append(
                (net_err < 0.1 * threshold).to(torch.float),
            )
            results[f"kpts2d_acc/pck005"].append(
                (net_err < 0.05 * threshold).to(torch.float),
            )
            results[f"kpts2d_acc/pck001"].append(
                (net_err < 0.01 * threshold).to(torch.float),
            )

        for cat in batch_categories:
            results[f"kpts2d_acc/{cat}_pck01"] = torch.cat(
                results[f"kpts2d_acc/{cat}_pck01"],
            )
            results[f"kpts2d_acc/{cat}_pck005"] = torch.cat(
                results[f"kpts2d_acc/{cat}_pck005"],
            )
            results[f"kpts2d_acc/{cat}_pck001"] = torch.cat(
                results[f"kpts2d_acc/{cat}_pck001"],
            )

        # net_target_feats_pxl2d_all = torch.stack(net_target_feats_pxl2d_all, dim=0)
        results[f"kpts2d_annots"] = net_target_feats_pxl2d_all
        results[f"kpts2d_acc/pck01"] = torch.cat(results[f"kpts2d_acc/pck01"])
        results[f"kpts2d_acc/pck005"] = torch.cat(results[f"kpts2d_acc/pck005"])
        results[f"kpts2d_acc/pck001"] = torch.cat(results[f"kpts2d_acc/pck001"])

        results["item_id"] = batch.item_id

        return results

        # query_index = batch.kpts2d_annots[0] / self.net.downsample_rate
        # query_index = query_index.to(int)  # W*H
        # # output from head
        # queried_feats = feats2d_rendered_1[0][:, query_index[0, :, 1], query_index[0, :, 0]]  # h *w

        # net_out
        # .shape
        # batch.rgbs
        # backbone_out, net_out = self.net(batch.rgb, return_backbone_output=True)
        # feats2d_net = net_out.featmap
        # kpts2d_gt = pad_sequence(batch.kpts2d_annot, batch_first=True).to(dtype=cam_tform4x4_obj.dtype,
        #                                                                   device=cam_tform4x4_obj.device)
        # kpts2d_mask = pad_sequence(batch.kpts2d_annot_vsbl, batch_first=True).to(dtype=torch.bool,
        #                                                                          device=cam_tform4x4_obj.device)
        # kpts2d_dist = torch.norm(kpts2d[:, :kpts2d_gt.shape[1]] - kpts2d_gt, dim=-1)
        # results["kpts2d_acc"] = (kpts2d_dist.detach() < 0.1 * max_width_height[:, None, ].expand(*kpts2d_dist.shape))[
        #                             kpts2d_mask] * 1.
        #
        # for cat_id, cat in enumerate(self.config.categories):
        #     results[f"kpts2d_acc/{cat}"] = \
        #     (kpts2d_dist.detach() < 0.1 * max_width_height[:, None, ].expand(*kpts2d_dist.shape))[
        #         batch.category_id == cat_id][kpts2d_mask[batch.category_id == cat_id]] * 1.

    def inference_batch(
        self,
        batch,
        return_samples_with_sim=True,
        multiview=False,
        val=False,
    ):
        results = OD3D_Results(logging_dir=self.logging_dir)
        B = len(batch)

        """
        # these parameters are used in prev. version
        batch.cam_tform4x4_obj[:, 2, 3] = 5. * 6. # 5. * 6.
        batch.cam_tform4x4_obj[:, 0, 3] = 0.
        batch.cam_tform4x4_obj[:, 1, 3] = 0.
        batch.cam_intr4x4[:, 0, 0] = 3000.
        batch.cam_intr4x4[:, 1, 1] = 3000.
        batch.cam_intr4x4[:, 0, 2] = batch.size[1] / 2.
        batch.cam_intr4x4[:, 1, 2] = batch.size[0] / 2.
        """

        time_loaded = time.time()
        with torch.no_grad():
            backbone_out, net_out = self.net(batch.rgb, return_backbone_output=True)
            feats2d_net = net_out.featmap

            if (
                self.config.train.get("epoch_inst_def_start", 0)
                <= self.scheduler.last_epoch
            ):
                instance_deform = self.meshes.get_instance_deform(
                    backbone_out,
                    img_feats_canonical=feats2d_net,
                    objects_ids=batch.category_id,
                )
            else:
                instance_deform = None

            if self.net_pose is not None:
                pred_cam_tform4x4_objs = self.pred_poses(
                    backbone_out=backbone_out,
                    cam_intr4x4=batch.cam_intr4x4,
                    size=batch.size,
                    cam_tform4x4_obj=batch.cam_tform4x4_obj,
                )
            else:
                pred_cam_tform4x4_objs = None

            # if multiview:
            #     if instance_deform is not None:
            #         instance_deform = instance_deform.get_first_item_repeated()
            #     else:
            #         instance_deform = None

            feats2d_net_mask = resize(
                batch.rgb_mask,
                H_out=feats2d_net.shape[2],
                W_out=feats2d_net.shape[3],
            )
            if self.config.inference.use_mask_object:
                feats2d_net_mask = (
                    feats2d_net_mask
                    * 1.0
                    * resize(
                        batch.mask,
                        H_out=feats2d_net.shape[2],
                        W_out=feats2d_net.shape[3],
                    )
                )

            time_pred_net_feats2d = time.time()
            # logger.info(
            #    f"predicted net feats2d, took {(time_pred_net_feats2d - time_loaded):.3f}")
            results["time_feats2d"] = (
                torch.Tensor([time_pred_net_feats2d - time_loaded]) / B
            )

            meshes_scores = []
            for mesh_id in range(len(self.meshes)):
                if self.config.inference.get("render_classify", False):
                    sim = self.meshes.get_sim_render(
                        feats2d_img=feats2d_net,
                        cams_tform4x4_obj=batch.cam_tform4x4_obj,
                        cams_intr4x4=batch.cam_intr4x4,
                        objects_ids=torch.LongTensor([mesh_id] * B).to(
                            device=batch.cam_tform4x4_obj.device,
                        ),
                        broadcast_batch_and_cams=False,
                        down_sample_rate=self.down_sample_rate,
                        feats2d_img_mask=feats2d_net_mask,
                        allow_clutter=self.config.inference.allow_clutter,
                        return_sim_pxl=False,
                        add_clutter=True,
                        temp=self.config.T,
                        instance_deform=instance_deform,
                        normalize_surface=self.config.inference.normalize_surface,
                        object_mask=batch.mask
                        if self.config.inference.add_mask_object_to_sim
                        else None,
                    )
                    sim = sim.squeeze(1)
                else:
                    # logger.info(f'calc score for mesh {self.config.categories[mesh_id]}')
                    sim_feats2d = self.meshes.get_sim_feats2d_img_to_all(
                        feats2d_img=feats2d_net,
                        imgs_sizes=batch.size,
                        cams_tform4x4_obj=None,
                        cams_intr4x4=None,
                        objects_ids=mesh_id,
                        broadcast_batch_and_cams=False,
                        down_sample_rate=self.down_sample_rate,
                        add_clutter=True,
                        add_other_objects=False,
                        dense=True,
                        sim_temp=self.config.train.T,
                        instance_deform=instance_deform,
                    )
                    sim = sim_feats2d.max(dim=1).values.flatten(1).mean(dim=-1)

                meshes_scores.append(sim)
            meshes_scores = torch.stack(meshes_scores, dim=-1)

            if multiview:
                meshes_scores = meshes_scores.mean(dim=0, keepdim=True).expand(
                    *meshes_scores.shape,
                )

            pred_class_scores, pred_class_ids = meshes_scores.max(dim=1)

            # logger.info(f'pred class ids {pred_class_ids}')
            time_pred_class = time.time()
            # logger.info(f"predicted class: {self.config.categories[int(pred_class_ids[0])]}, took {(time_pred_class - time_pred_net_feats2d):.3f}")

            results["time_class"] = (
                torch.Tensor([time_pred_class - time_pred_net_feats2d]) / B
            )

            b_cams_multiview_tform4x4_obj, b_cams_multiview_intr4x4 = self.get_samples(
                config_sample=self.config.inference.sample,
                cam_intr4x4=batch.cam_intr4x4,
                cam_tform4x4_obj=batch.cam_tform4x4_obj,
                feats2d_net=feats2d_net,
                categories_ids=batch.category_id,
                feats2d_net_mask=feats2d_net_mask,
                multiview=multiview,
                instance_deform=instance_deform,
                pred_cam_tform4x4_objs=pred_cam_tform4x4_objs,
            )

            if "affine" in self.config.inference.sample.method:
                from od3d.cv.geometry.fit.phase_corr import get_affine_from_imgs

                imgs_sizes = (
                    torch.LongTensor(list(feats2d_net.shape[-2:]))
                    * self.down_sample_rate
                )
                feats2d_rendered_gt = self.meshes.render(
                    modalities=PROJECT_MODALITIES.FEATS,
                    cams_tform4x4_obj=batch.cam_tform4x4_obj,
                    cams_intr4x4=batch.cam_intr4x4,
                    imgs_sizes=imgs_sizes,
                    objects_ids=batch.category_id,
                    broadcast_batch_and_cams=False,
                    down_sample_rate=self.down_sample_rate,
                    add_clutter=True,
                    add_other_objects=False,
                    instance_deform=instance_deform,
                )
                feats2d_rendered = self.meshes.render(
                    modalities=PROJECT_MODALITIES.FEATS,
                    cams_tform4x4_obj=b_cams_multiview_tform4x4_obj,
                    cams_intr4x4=b_cams_multiview_intr4x4,
                    imgs_sizes=imgs_sizes,
                    objects_ids=batch.category_id,
                    broadcast_batch_and_cams=True,
                    down_sample_rate=self.down_sample_rate,
                    add_clutter=True,
                    add_other_objects=False,
                    instance_deform=instance_deform,
                )
                C = b_cams_multiview_tform4x4_obj.shape[1]
                # perhaps good with no theta ?
                # perhaps good with checkpoint with normalized features?
                # perhaps angle / scale selection is not ideal?
                M = (
                    torch.eye(4)
                    .expand(B, C, 4, 4)
                    .to(device=feats2d_net.device)
                    .clone()
                )
                for b in range(B):
                    for c in range(C):
                        M[b, c, :3, :3] = get_affine_from_imgs(
                            feats2d_rendered[b, c],
                            feats2d_rendered_gt[b],
                        )  # feats2d_net[b] / feats2d_net[b].norm(dim=0, keepdim=True)
                        from kornia.geometry.transform import invert_affine_transform

                        M[b : b + 1, c, :2, :3] = invert_affine_transform(
                            M[b : b + 1, c, :2, :3],
                        )
                        M[b, c, :2, 2] *= self.down_sample_rate

                # b_cams_multiview_intr4x4 = tform4x4(M, b_cams_multiview_intr4x4)
                b_cams_multiview_tform4x4_obj = tform4x4(
                    tform4x4(
                        tform4x4(torch.inverse(b_cams_multiview_intr4x4), M),
                        b_cams_multiview_intr4x4,
                    ),
                    b_cams_multiview_tform4x4_obj,
                )

            logger.info(b_cams_multiview_tform4x4_obj.shape)
            logger.info(b_cams_multiview_intr4x4.shape)

            logger.info(batch.category_id)
            #  OPTION A: Use 2d gradient of rendered features
            sim = self.meshes.get_sim_render(
                feats2d_img=feats2d_net,
                cams_tform4x4_obj=b_cams_multiview_tform4x4_obj,
                cams_intr4x4=b_cams_multiview_intr4x4,
                objects_ids=batch.category_id,
                broadcast_batch_and_cams=True,
                down_sample_rate=self.down_sample_rate,
                feats2d_img_mask=feats2d_net_mask,
                allow_clutter=self.config.inference.allow_clutter,
                return_sim_pxl=False,
                add_clutter=True,
                temp=self.config.train.T,
                instance_deform=instance_deform
                if self.config.inference.sample.get("use_instance_deform", True)
                else None,
                normalize_surface=self.config.inference.normalize_surface,
                object_mask=batch.mask
                if self.config.inference.add_mask_object_to_sim
                else None,
                use_neg_mse="sim_neg_mse_max" in self.config.train.loss.appear.type,
            )

            if multiview:
                sim = sim.mean(dim=0, keepdim=True).expand(*sim.shape)

            if return_samples_with_sim:
                results[
                    "samples_cam_tform4x4_obj"
                ] = b_cams_multiview_tform4x4_obj.detach()
                results["samples_cam_intr4x4"] = b_cams_multiview_intr4x4.detach()
                results["samples_sim"] = sim.detach()

            mesh_multiple_cams_loss = -sim

            mesh_cam_loss_min_val, mesh_cam_loss_min_id = mesh_multiple_cams_loss.min(
                dim=1,
            )

            cam_tform4x4_obj = (
                b_cams_multiview_tform4x4_obj[:, mesh_cam_loss_min_id]
                .permute(2, 3, 0, 1)
                .diagonal(
                    dim1=-2,
                    dim2=-1,
                )
                .permute(2, 0, 1)
            )

            sim = batched_index_select(
                input=sim.detach(),
                index=mesh_cam_loss_min_id[:, None],
            )[:, 0]

            cam_intr4x4 = (
                b_cams_multiview_intr4x4[:, mesh_cam_loss_min_id]
                .permute(2, 3, 0, 1)
                .diagonal(
                    dim1=-2,
                    dim2=-1,
                )
                .permute(2, 0, 1)
            )

            if multiview:
                objs_multiview_tform4x4_cuboid_front = tform4x4_broadcast(
                    inv_tform4x4(batch.cam_tform4x4_obj[:1])[:, None],
                    b_cams_multiview_tform4x4_obj,
                )
                obj_tform4x4_cuboid_front = objs_multiview_tform4x4_cuboid_front[
                    0,
                    mesh_cam_loss_min_id[0],
                ]

        if self.config.inference.refine.enabled:
            if (
                not val
                or self.config.inference.refine.get("dims_detached_val", None) is None
            ):
                dims_detached = self.config.inference.refine.dims_detached
            else:
                dims_detached = self.config.inference.refine.dims_detached_val

            if instance_deform is not None and instance_deform.latent is not None:
                if not multiview:
                    instance_deform_latent_tmp_param = torch.nn.Parameter(
                        instance_deform.latent.detach()
                        .clone()
                        .to(device=cam_tform4x4_obj.device),
                        requires_grad=True,
                    )
                    instance_deform_latent_tmp = instance_deform_latent_tmp_param
                else:
                    instance_deform_latent_tmp_param = torch.nn.Parameter(
                        instance_deform.latent.detach()
                        .clone()[:1]
                        .to(device=cam_tform4x4_obj.device),
                        requires_grad=True,
                    )
                    instance_deform_latent_tmp = (
                        instance_deform_latent_tmp_param.repeat(B, 1).clone()
                    )

            else:
                instance_deform_latent_tmp_param = None
                instance_deform_latent_tmp = None

            if not multiview:
                obj_tform6_tmp = torch.nn.Parameter(
                    torch.zeros(size=(B, 6)).to(device=cam_tform4x4_obj.device),
                    requires_grad=True,
                )
            else:
                init_vals = torch.zeros(size=(1, 7)).to(device=cam_tform4x4_obj.device)
                init_vals[:, -1] = 1.0
                obj_tform6_tmp = torch.nn.Parameter(
                    init_vals,
                    requires_grad=True,
                )

            # transl: 0, 1, 2 rot: 3, 4, 5
            if (
                not self.config.inference.refine.latent_detached
                and instance_deform_latent_tmp_param is not None
            ):
                params = [obj_tform6_tmp, instance_deform_latent_tmp_param]
            else:
                params = [obj_tform6_tmp]
            optim_inference = torch.optim.Adam(
                params=params,  # instance_deform_latent_tmp
                lr=self.config.inference.optimizer.lr,
                betas=(
                    self.config.inference.optimizer.beta0,
                    self.config.inference.optimizer.beta1,
                ),
            )

            time_before_pose_iterative = time.time()
            cam_tform4x4_obj = tform4x4_broadcast(
                cam_tform4x4_obj.detach(),
                se3_exp_map(obj_tform6_tmp[:, :6]),
            )

            if multiview:
                obj_scale3x3 = torch.eye(3).to(cam_tform4x4_obj.device)[None,]
                obj_scale3x3 = obj_scale3x3 * obj_tform6_tmp[:, -1]
                from od3d.cv.geometry.transform import transf4x4_from_rot3x3

                obj_scale4x4 = transf4x4_from_rot3x3(obj_scale3x3)
                cam_tform4x4_obj = tform4x4_broadcast(
                    cam_tform4x4_obj.detach(),
                    obj_scale4x4,
                )

            if not multiview:
                refine_update_max = self.refine_update_max[batch.category_id].clone()
            else:
                refine_update_max = self.refine_update_max[
                    batch.category_id[:1]
                ].clone()
                refine_update_max = torch.cat(
                    [refine_update_max, 2 * torch.ones_like(refine_update_max[:1, :1])],
                    dim=-1,
                )

            # if not self.config.inference.refine.latent_detached and instance_deform_latent_tmp is not None:
            #    self.meshes.instance_deform_net.train()

            for epoch in range(self.config.inference.optimizer.epochs):
                if multiview and instance_deform_latent_tmp is not None:
                    instance_deform_latent_tmp = (
                        instance_deform_latent_tmp_param.repeat(B, 1).clone()
                    )

                backbone_out.latent = instance_deform_latent_tmp  # [:1].repeat(len(instance_deform_latent_tmp), 1).clone()

                if (
                    self.config.train.get("epoch_inst_def_start", 0)
                    <= self.scheduler.last_epoch
                ):
                    instance_deform = self.meshes.get_instance_deform(
                        backbone_out,
                        img_feats_canonical=feats2d_net,
                        objects_ids=batch.category_id,
                    )
                else:
                    instance_deform = None

                cam_tform4x4_obj = tform4x4_broadcast(
                    cam_tform4x4_obj.detach(),
                    se3_exp_map(obj_tform6_tmp[:, :6].detach()),
                )
                if multiview:
                    obj_scale3x3 = torch.eye(3).to(cam_tform4x4_obj.device)[None,]
                    obj_scale3x3 = obj_scale3x3 * obj_tform6_tmp[:, -1]
                    from od3d.cv.geometry.transform import transf4x4_from_rot3x3

                    obj_scale4x4 = transf4x4_from_rot3x3(obj_scale3x3)
                    cam_tform4x4_obj = tform4x4_broadcast(
                        cam_tform4x4_obj.detach(),
                        obj_scale4x4.detach(),
                    )

                obj_tform6_tmp.data[:, :6] = 0.0
                if multiview:
                    obj_tform6_tmp.data[:, -1] = 1.0

                cam_tform4x4_obj = tform4x4_broadcast(
                    cam_tform4x4_obj.detach(),
                    se3_exp_map(obj_tform6_tmp[:, :6]),
                )
                if multiview:
                    obj_scale3x3 = torch.eye(3).to(cam_tform4x4_obj.device)[None,]
                    obj_scale3x3 = obj_scale3x3 * obj_tform6_tmp[:, -1]
                    from od3d.cv.geometry.transform import transf4x4_from_rot3x3

                    obj_scale4x4 = transf4x4_from_rot3x3(obj_scale3x3)
                    cam_tform4x4_obj = tform4x4_broadcast(
                        cam_tform4x4_obj,
                        obj_scale4x4,
                    )

                sim, sim_pxl = self.meshes.get_sim_render(
                    feats2d_img=feats2d_net,
                    cams_tform4x4_obj=cam_tform4x4_obj,
                    cams_intr4x4=cam_intr4x4,
                    objects_ids=batch.category_id,
                    broadcast_batch_and_cams=False,
                    down_sample_rate=self.down_sample_rate,
                    feats2d_img_mask=feats2d_net_mask,
                    allow_clutter=self.config.inference.allow_clutter,
                    return_sim_pxl=True,
                    add_clutter=True,
                    temp=self.config.train.T,
                    normalize_surface=self.config.inference.normalize_surface,
                    object_mask=batch.mask
                    if self.config.inference.add_mask_object_to_sim
                    else None,
                    instance_deform=instance_deform,
                    use_neg_mse="sim_neg_mse_max" in self.config.train.loss.appear.type,
                )

                # sim, sim_pxl = self.get_sim_feats2d_net_with_cams(
                #     feats2d_net=feats2d_net,
                #     feats2d_net_mask=feats2d_net_mask,
                #     cam_tform4x4_obj=cam_tform4x4_obj,
                #     cam_intr4x4=cam_intr4x4,
                #     categories_ids=batch.category_id,
                #     return_sim_pxl=True,
                #     broadcast_batch_and_cams=False,
                #     pre_rendered=False,
                #     only_use_rendered_inliers=self.config.inference.only_use_rendered_inliers,
                #     allow_clutter=True,
                #     use_sigmoid=self.config.inference.use_sigmoid,
                # )
                mesh_cam_loss = -sim

                if self.config.inference.live:
                    img_indices = [0]
                    # img_indices = [0, 3, 7, ]
                    # img_indices = [0, 1, 2, 3, 4, 5, 6, 7, ]
                    from od3d.cv.visual.show import show_imgs

                    show_imgs(
                        torch.stack(
                            [
                                blend_rgb(
                                    batch.rgb[img_index],
                                    (
                                        self.meshes.render(
                                            cams_tform4x4_obj=cam_tform4x4_obj[
                                                img_index : img_index + 1
                                            ],
                                            cams_intr4x4=cam_intr4x4[
                                                img_index : img_index + 1
                                            ],
                                            imgs_sizes=batch.size,
                                            objects_ids=batch.category_id[
                                                img_index : img_index + 1
                                            ],
                                            modalities=PROJECT_MODALITIES.PT3D_NCDS,
                                            instance_deform=instance_deform[
                                                img_index : img_index + 1
                                            ],
                                        )[0]
                                    ).to(dtype=batch.rgb.dtype),
                                )
                                for img_index in img_indices
                            ],
                            dim=0,
                        ),
                        duration=1,
                    )

                loss = mesh_cam_loss.sum()
                loss.backward()
                optim_inference.step()
                optim_inference.zero_grad()

                # detach update
                obj_tform6_tmp.data[:, dims_detached] = 0.0
                if multiview and 6 in dims_detached:
                    obj_tform6_tmp.data[:, 6] = 1.0

                # clip update
                refine_update_mask_lmax = obj_tform6_tmp.data.abs() > refine_update_max
                if refine_update_mask_lmax.sum() > 0:
                    logger.warning(
                        f"refinement contains too large values {obj_tform6_tmp.data.abs()}",
                    )
                    obj_tform6_tmp.data[refine_update_mask_lmax] = (
                        obj_tform6_tmp.data[refine_update_mask_lmax].sign()
                        * refine_update_max[refine_update_mask_lmax]
                    )

                refine_update_mask_nans = obj_tform6_tmp.data.isnan()
                if refine_update_mask_nans.sum() > 0:
                    logger.warning("refinement contains NaNs")
                    obj_tform6_tmp.data[refine_update_mask_nans] = 0.0

                refine_update_mask_infs = obj_tform6_tmp.data.isinf()
                if refine_update_mask_infs.sum() > 0:
                    logger.warning("refinement contains Infs")
                    obj_tform6_tmp.data[refine_update_mask_infs] = 0.0

            cam_tform4x4_obj = tform4x4_broadcast(
                cam_tform4x4_obj.detach(),
                se3_exp_map(obj_tform6_tmp[:, :6].detach()),
            )
            if multiview:
                obj_scale3x3 = torch.eye(3).to(cam_tform4x4_obj.device)[None,]
                obj_scale3x3 = obj_scale3x3 * obj_tform6_tmp[:, -1]
                from od3d.cv.geometry.transform import transf4x4_from_rot3x3

                obj_scale4x4 = transf4x4_from_rot3x3(obj_scale3x3)
                cam_tform4x4_obj = tform4x4_broadcast(
                    cam_tform4x4_obj.detach(),
                    obj_scale4x4.detach(),
                )

            results["time_pose_iterative"] = (
                torch.Tensor([time.time() - time_before_pose_iterative]) / B
            )

        cam_tform4x4_obj = cam_tform4x4_obj.clone().detach()

        if (
            instance_deform is not None
            and instance_deform is not None
            and instance_deform.latent is not None
        ):
            results["latent"] = instance_deform.latent.detach().clone()

        if self.config.train.loss.geo.rec.weight > 0.0:
            results["geo"] = self.get_geo_loss(
                batch=batch,
                instance_deform=instance_deform,
                rec_type=self.config.train.loss.geo.rec.type,
            ).detach()

        if OD3D_FRAME_MODALITIES.MASK in batch.modalities:
            results["iou_pose_gt"] = self.get_geo_mask_metrics(
                batch=batch,
                cam_tform4x4_obj=batch.cam_tform4x4_obj,
                instance_deform=instance_deform,
            ).detach()
            results["iou_pose_pred"] = self.get_geo_mask_metrics(
                batch=batch,
                cam_tform4x4_obj=cam_tform4x4_obj,
                instance_deform=instance_deform,
            ).detach()
        if OD3D_FRAME_MODALITIES.MESH in batch.modalities:
            if OD3D_FRAME_MODALITIES.MASK in batch.modalities:
                results["iou_amodal_pose_gt"] = self.get_geo_mask_metrics(
                    batch=batch,
                    cam_tform4x4_obj=batch.cam_tform4x4_obj,
                    instance_deform=instance_deform,
                    metrics=["iou_amodal"],
                ).detach()
                results["iou_amodal_pose_pred"] = self.get_geo_mask_metrics(
                    batch=batch,
                    cam_tform4x4_obj=cam_tform4x4_obj,
                    instance_deform=instance_deform,
                    metrics=["iou_amodal"],
                ).detach()

        # if self.config.train.loss.geo.mask.weight > 0.:
        #    results["mask"] = self.get_mask_loss(batch=batch, instance_deform=instance_deform,
        #                                         rec_type=self.config.train.loss.geo.rec.type).detach()

        # a) average per category,
        # b) use pascal3d nearest neighbor average
        if batch.kpts3d is not None and batch.kpts2d_annot is not None:
            if self.kpts3d_id is not None:
                if batch.bbox is not None:
                    max_width_height = (
                        (batch.bbox[..., 2:4] - batch.bbox[..., 0:2]).max(dim=-1).values
                    )

                else:
                    max_width_height = batch.size.max(dim=-1).values[None,]

                kpts3d_id = self.get_kpts3d_ids(category_id=batch.category_id).to(
                    device=cam_tform4x4_obj.device,
                )
                # B x K
                pts3d = self.meshes.sample(
                    objects_ids=batch.category_id,
                    modalities=PROJECT_MODALITIES.PT3D,
                    add_clutter=False,
                    add_other_objects=False,
                    instance_deform=instance_deform,
                )
                # B x V x 3
                kpts3d = batched_index_select(input=pts3d, index=kpts3d_id, dim=1)
                # kpts3d = pad_sequence(batch.kpts3d, batch_first=True).to(dtype=cam_tform4x4_obj.dtype, device=cam_tform4x4_obj.device)

                kpts2d = proj3d2d_broadcast(
                    proj4x4=tform4x4_broadcast(cam_intr4x4, cam_tform4x4_obj)[:, None],
                    pts3d=kpts3d,
                )
                # kpts2d = proj3d2d_broadcast(proj4x4=tform4x4_broadcast(batch.cam_intr4x4, batch.cam_tform4x4_obj)[:, None], pts3d=kpts3d)

                kpts2d_gt = pad_sequence(batch.kpts2d_annot, batch_first=True).to(
                    dtype=cam_tform4x4_obj.dtype,
                    device=cam_tform4x4_obj.device,
                )
                kpts2d_mask = pad_sequence(
                    batch.kpts2d_annot_vsbl,
                    batch_first=True,
                ).to(dtype=torch.bool, device=cam_tform4x4_obj.device)
                kpts2d_dist = torch.norm(
                    kpts2d[:, : kpts2d_gt.shape[1]] - kpts2d_gt,
                    dim=-1,
                )
                results["kpts2d_acc"] = (
                    kpts2d_dist.detach()
                    < 0.1 * max_width_height[:, None].expand(*kpts2d_dist.shape)
                )[kpts2d_mask] * 1.0

                for cat_id, cat in enumerate(self.config.categories):
                    results[f"kpts2d_acc/{cat}"] = (
                        kpts2d_dist.detach()
                        < 0.1 * max_width_height[:, None].expand(*kpts2d_dist.shape)
                    )[batch.category_id == cat_id][
                        kpts2d_mask[batch.category_id == cat_id]
                    ] * 1.0

                # if 'kpts3d' in batch.
                """
                - 'kpts2d_annot'
                - 'kpts2d_annot_vsbl'
                - 'kpts3d'
                """
            else:
                logger.info("set kpts3d_id to evaluate kpts acc")

        results["time_pose"] = torch.Tensor([time.time() - time_pred_class]) / B

        gt_cam_tform4x4_obj = batch.cam_tform4x4_obj.clone()
        pred_cam_tform4x4_obj = cam_tform4x4_obj.clone()
        gt_cam_tform4x4_obj_real_scale = batch.cam_tform4x4_obj.clone()
        pred_cam_tform4x4_obj_real_scale = cam_tform4x4_obj.clone()
        if batch.obj_tform4x4_objs is not None:
            batch_objs_scale = (
                torch.cat(batch.obj_tform4x4_objs, dim=0)[:, :3, :3]
                .norm(dim=1)
                .mean(dim=1)
            )
            gt_cam_tform4x4_obj_real_scale[:, :3] = (
                gt_cam_tform4x4_obj_real_scale[:, :3] * batch_objs_scale[:, None, None]
            )
            pred_cam_tform4x4_obj_real_scale[:, :3] = (
                pred_cam_tform4x4_obj_real_scale[:, :3]
                * batch_objs_scale[:, None, None]
            )

            # 1) batch_objs_scale dim != gt_cam_tform4x4_obj_real_scale dim
            # 2) float instead of bool
        if batch.mesh is not None:
            # BxVx3
            # from od3d.cv.geometry.objects3d.meshes.meshes import VERT_MODALITIES
            # pred_meshes_verts, _ = self.meshes.get_vert_mod_from_objs(mod=VERT_MODALITIES.PT3D, objs_ids=batch.category_id, padded=True, instance_deform=instance_deform, clone=True)
            # gt_meshes_verts, _ = batch.mesh.get_vert_mod_from_objs(mod=VERT_MODALITIES.PT3D, padded=True, clone=True)

            # gt_bboxs3d = torch.stack([gt_meshes_verts.min(dim=1).values, gt_meshes_verts.max(dim=1).values], dim=1)
            # pred_bboxs3d = torch.stack([pred_meshes_verts.min(dim=1).values, pred_meshes_verts.max(dim=1).values], dim=1)
            # TODO get 3D BOUNDING BOXES
            #
            gt_meshes = batch.mesh
            pred_meshes = self.meshes.get_meshes_with_ids(
                meshes_ids=batch.category_id,
                instance_deform=instance_deform,
            )

            gt_bboxs3d = gt_meshes.to_bboxs3d()
            pred_bboxs3d = pred_meshes.to_bboxs3d()

            gt_cam_bboxs3d = gt_bboxs3d.get_meshes_with_ids(clone=True)
            gt_cam_bboxs3d.transf3d(objs_new_tform4x4_objs=gt_cam_tform4x4_obj)
            pred_cam_bboxs3d = pred_bboxs3d.get_meshes_with_ids(clone=True)
            pred_cam_bboxs3d.transf3d(objs_new_tform4x4_objs=pred_cam_tform4x4_obj)

            gt_cam_bboxs3d_real_scale = gt_bboxs3d.get_meshes_with_ids(clone=True)
            gt_cam_bboxs3d_real_scale.transf3d(
                objs_new_tform4x4_objs=gt_cam_tform4x4_obj_real_scale,
            )
            pred_cam_bboxs3d_real_scale = pred_bboxs3d.get_meshes_with_ids(clone=True)
            pred_cam_bboxs3d_real_scale.transf3d(
                objs_new_tform4x4_objs=pred_cam_tform4x4_obj_real_scale,
            )

            ious_3d = []
            for b in range(len(gt_bboxs3d)):
                from trimesh.boolean import intersection, union

                a = gt_cam_bboxs3d.get_meshes_with_ids(meshes_ids=[b]).to_trimesh()
                b = pred_cam_bboxs3d.get_meshes_with_ids(meshes_ids=[b]).to_trimesh()
                iou_3d = (
                    intersection([a, b], check_volume=True, engine="manifold").volume
                    / union([a, b], check_volume=False, engine="manifold").volume
                )
                ious_3d.append(iou_3d)
            ious_3d = torch.Tensor(ious_3d).to(self.device)
            results["bbox3d_iou"] = ious_3d
            results["bbox3d_acc_@25"] = 1.0 * (results["bbox3d_iou"] > 0.25)
            results["bbox3d_acc_@50"] = 1.0 * (results["bbox3d_iou"] > 0.50)
            results["bbox3d_acc_@75"] = 1.0 * (results["bbox3d_iou"] > 0.75)

            ious_3d = []
            for b in range(len(gt_bboxs3d)):
                from trimesh.boolean import intersection, union

                a = gt_cam_bboxs3d_real_scale.get_meshes_with_ids(
                    meshes_ids=[b],
                ).to_trimesh()
                b = pred_cam_bboxs3d_real_scale.get_meshes_with_ids(
                    meshes_ids=[b],
                ).to_trimesh()
                iou_3d = (
                    intersection([a, b], check_volume=True, engine="manifold").volume
                    / union([a, b], check_volume=False, engine="manifold").volume
                )
                ious_3d.append(iou_3d)
            ious_3d = torch.Tensor(ious_3d).to(self.device)
            results["bbox3d_real_scale_iou"] = ious_3d
            results["bbox3d_real_scale_acc_@25"] = 1.0 * (results["bbox3d_iou"] > 0.25)
            results["bbox3d_real_scale_acc_@50"] = 1.0 * (results["bbox3d_iou"] > 0.50)
            results["bbox3d_real_scale_acc_@75"] = 1.0 * (results["bbox3d_iou"] > 0.75)

        batch_rot_diff_rad = get_pose_diff_in_rad(
            pred_tform4x4=cam_tform4x4_obj,
            gt_tform4x4=batch.cam_tform4x4_obj,
        )

        batch_transl_diff = (
            pred_cam_tform4x4_obj[:, :3, 3] - gt_cam_tform4x4_obj[:, :3, 3]
        ).norm(dim=-1)
        results["transl_diff"] = batch_transl_diff.detach()
        results["transl_acc_@5%"] = 1.0 * (results["transl_diff"] < 0.1)
        results["transl_acc_@10%"] = 1.0 * (results["transl_diff"] < 0.2)

        batch_transl_diff_real_scale = (
            pred_cam_tform4x4_obj_real_scale[:, :3, 3]
            - gt_cam_tform4x4_obj_real_scale[:, :3, 3]
        ).norm(dim=-1)
        results["transl_diff_real_scale"] = batch_transl_diff_real_scale.detach()
        results["transl_acc_real_scale_@2cm"] = 1.0 * (
            results["transl_diff_real_scale"] < 0.02
        )
        results["transl_acc_real_scale_@5cm"] = 1.0 * (
            results["transl_diff_real_scale"] < 0.05
        )
        results["transl_acc_real_scale_@10cm"] = 1.0 * (
            results["transl_diff_real_scale"] < 0.1
        )

        results["rot_diff_rad"] = batch_rot_diff_rad.detach()
        results["rot_acc_@5deg"] = 1.0 * (
            results["rot_diff_rad"] < ((5.0 / 360.0) * torch.pi * 2.0)
        )
        results["rot_acc_@10deg"] = 1.0 * (
            results["rot_diff_rad"] < ((10.0 / 360.0) * torch.pi * 2.0)
        )
        results["rot_acc_@30deg"] = 1.0 * (
            results["rot_diff_rad"] < ((30.0 / 360.0) * torch.pi * 2.0)
        )

        results["transl_rot_acc_@10deg_5cm"] = (
            results["rot_acc_@10deg"] * results["transl_acc_real_scale_@5cm"]
        )
        results["transl_rot_acc_@30deg_10cm"] = (
            results["rot_acc_@30deg"] * results["transl_acc_real_scale_@10cm"]
        )

        results["transl_rot_acc_@10deg_5%"] = (
            results["rot_acc_@10deg"] * results["transl_acc_@5%"]
        )
        results["transl_rot_acc_@30deg_10%"] = (
            results["rot_acc_@30deg"] * results["transl_acc_@10%"]
        )

        for cat_id, cat in enumerate(self.config.categories):
            results[f"{cat}_rot_diff_rad"] = batch_rot_diff_rad[
                batch.category_id == cat_id
            ]
        results["label_gt"] = batch.category_id
        results["label_pred"] = pred_class_ids.detach()
        results["label_names"] = self.config.categories
        results["sim"] = sim.detach()
        results["cam_tform4x4_obj"] = cam_tform4x4_obj.detach()
        results["cam_tform4x4_obj_gt"] = batch.cam_tform4x4_obj.detach()
        results["item_id"] = batch.item_id
        results["name_unique"] = batch.name_unique

        return results

    def set_kpts(self, dataset: OD3D_Dataset, subset_fraction=1.0):
        """
        Args:
            dataset (OD3D_Dataset)
        """

        from od3d.datasets.dataset import OD3D_DATASET_SPLITS

        dataset_sub, _ = dataset.get_split(
            fraction1=subset_fraction,
            fraction2=1.0 - subset_fraction,
            split=OD3D_DATASET_SPLITS.RANDOM,
        )

        dataset_sub.transform = copy.deepcopy(self.transform_test)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset_sub,
            batch_size=self.config.test.dataloader.batch_size,
            shuffle=False,
            collate_fn=dataset_sub.collate_fn,
            num_workers=self.config.test.dataloader.num_workers,
            pin_memory=self.config.test.dataloader.pin_memory,
        )
        logger.info("setting kpts3d")
        logger.info(f"Dataset contains {len(dataset_sub)} frames.")
        ### METHOD 1: find 3D keypoints among mesh vertices
        if (
            self.config.test.kpts3d == "vertex_sim_max"
            or self.config.test.kpts3d == "vertex_dist2d_min"
        ):
            sim_feats_total = []
            cats_total = []
            K = 0
            V = 0
            for i, batch in tqdm(enumerate(iter(dataloader))):
                batch.to(device=self.device)
                with torch.no_grad():
                    # results_batch = self.inference_batch(batch=batch, multiview=False)
                    net_out = self.net(batch.rgb)
                    feats2d_net = net_out.featmap
                    obj3d_feats1d_sampled = self.meshes.sample(
                        modalities=PROJECT_MODALITIES.FEATS,
                        cams_tform4x4_obj=None,
                        cams_intr4x4=None,
                        imgs_sizes=None,
                        objects_ids=batch.category_id,
                        broadcast_batch_and_cams=False,
                        down_sample_rate=1.0,
                        add_clutter=False,
                        add_other_objects=False,
                        instance_deform=None,
                    )  # B x V x F

                    from od3d.cv.visual.sample import sample_pxl2d_pts
                    from torch.nn.utils.rnn import pad_sequence

                    kpts2d_vsbl_gt = pad_sequence(
                        batch.kpts2d_annot_vsbl,
                        batch_first=True,
                    ).to(feats2d_net.device)
                    kpts2d_gt = pad_sequence(batch.kpts2d_annot, batch_first=True).to(
                        feats2d_net.device,
                    )
                    img_feats1d_sampled = sample_pxl2d_pts(
                        x=feats2d_net,
                        pxl2d=kpts2d_gt / self.down_sample_rate,
                    )  # B x K x F
                    sim_feats1d = (
                        self.meshes.get_sim_feats1d_img_and_feats1d_obj(
                            img_feats1d_sampled,
                            obj3d_feats1d_sampled,
                            add_clutter=False,
                            temp=self.config.train.T,
                        )
                        * kpts2d_vsbl_gt[:, None, :]
                    )  # B x V x K
                    if sim_feats1d.shape[-1] > K:
                        K = sim_feats1d.shape[-1]
                    if sim_feats1d.shape[-2] > V:
                        V = sim_feats1d.shape[-2]

                    if len(sim_feats_total) == 0:
                        sim_feats_total = sim_feats1d
                    else:
                        _sim_feats_total = torch.zeros(
                            (sim_feats_total.shape[0] + sim_feats1d.shape[0], V, K),
                        ).to(device=sim_feats1d.device)
                        _sim_feats_total[
                            : sim_feats_total.shape[0],
                            : sim_feats_total.shape[1],
                            : sim_feats_total.shape[2],
                        ] = sim_feats_total
                        _sim_feats_total[
                            sim_feats_total.shape[0] :,
                            : sim_feats1d.shape[1],
                            : sim_feats1d.shape[2],
                        ] = sim_feats1d
                        sim_feats_total = _sim_feats_total
                    cats_total.append(batch.category_id)

            kpts3d = []
            kpts3d_id = []
            cats_total = torch.cat(cats_total)  # B
            for cat_id in range(len(dataset.categories)):
                cat_sim_feats_total = sim_feats_total[cats_total == cat_id]
                cat_vert_id = cat_sim_feats_total.mean(dim=0).argmax(dim=0)
                cat_kpts3d = self.meshes.get_verts_with_mesh_id(mesh_id=cat_id)[
                    cat_vert_id
                ]
                kpts3d.append(cat_kpts3d)
                kpts3d_id.append(cat_vert_id)

        elif self.config.test.kpts3d == "avg_kpts3d":
            ### METHOD 2: simply average 3D keypoints
            kpts3d = []
            kpts3d_id = []
            for cat_id in range(len(dataset.categories)):
                kpts3d.append([])

            for i, batch in tqdm(enumerate(iter(dataloader))):
                batch.to(device=self.device)
                for b in range(len(batch.category_id)):
                    cat_id = batch.category_id[b].item()
                    kpts3d[cat_id].append(batch.kpts3d[b])

            for cat_id in range(len(dataset.categories)):
                # kpts3d[cat_id] = torch.stack(kpts3d[cat_id]).unique().mean(dim=0)
                if len(kpts3d[cat_id]) > 0:
                    kpts3d[cat_id] = torch.stack(kpts3d[cat_id])  # .mean(dim=0)
                    kpts3d_unique = kpts3d[cat_id].unique(dim=0)
                    kpts3d_unique_noninf = ~kpts3d_unique.isinf()
                    kpts3d_unique[~kpts3d_unique_noninf] = 0
                    kpts3d_unique_mean = kpts3d_unique.sum(
                        dim=0,
                    ) / kpts3d_unique_noninf.sum(dim=0)
                    kpts3d[cat_id] = kpts3d_unique_mean.to(
                        device=self.device,
                    )  #  kpts3d[cat_id].unique(dim=0).mean(dim=0)
                    cat_verts = self.meshes.get_verts_with_mesh_id(mesh_id=cat_id).to(
                        device=self.device,
                    )  # Vx3
                    cat_vert_id = (
                        (kpts3d[cat_id][:, None] - cat_verts[None, :])
                        .norm(dim=-1)
                        .argmin(dim=-1)
                    )
                    kpts3d_id.append(cat_vert_id)
                else:
                    logger.warning(
                        f"no kpts set for category {dataset.categories[cat_id]}",
                    )
                    kpts3d[cat_id] = torch.zeros((1, 3)).to(device=self.device)
                    kpts3d_id.append(
                        torch.zeros((1,)).to(dtype=torch.long, device=self.device),
                    )

        from torch.nn.utils.rnn import pad_sequence

        self.kpts3d = pad_sequence(kpts3d, batch_first=True)  # CxKx3
        self.kpts3d_id = pad_sequence(kpts3d_id, batch_first=True).long()  # CxKx3
        del kpts3d
        del kpts3d_id

    def get_kpts3d_ids(self, category_id):
        """
        Args:
            category_id (torch.LongTensor): B,
        Returns
            kpts3d_id (torch.Tensor): BxK
        """
        if self.kpts3d_id is None:
            raise ValueError("kpts3d is not set")

        else:
            return self.kpts3d_id.to(category_id.device)[category_id].clone()

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

            if "gt_cam_tform4x4_obj" in results_batch.keys():
                batch.cam_tform4x4_obj = results_batch["gt_cam_tform4x4_obj"].to(
                    device=self.device,
                )[batch_result_ids]
            if (
                "noise2d" in results_batch.keys()
                and results_batch["noise2d"] is not None
            ):
                batch.noise2d = results_batch["noise2d"].to(device=self.device)[
                    batch_result_ids
                ]

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

            backbone_out, net_out = self.net(batch.rgb, return_backbone_output=True)
            if "latent" in results_batch.keys():
                backbone_out.latent = torch.stack(
                    [results_batch["latent"][b] for b in batch_result_ids],
                    dim=0,
                ).to(device=self.device)
            feats2d_net = net_out.featmap

            if (
                self.config.train.get("epoch_inst_def_start", 0)
                <= self.scheduler.last_epoch
            ):
                instance_deform = self.meshes.get_instance_deform(
                    backbone_out,
                    img_feats_canonical=feats2d_net,
                    objects_ids=batch.category_id,
                )
            else:
                instance_deform = None

            feats2d_net_mask = 1.0 * resize(
                batch.rgb_mask,
                H_out=feats2d_net.shape[2],
                W_out=feats2d_net.shape[3],
            )
            sample1d_mods = self.meshes.sample_with_img2d(
                img2d=feats2d_net,
                img2d_mask=feats2d_net_mask,
                cams_intr4x4=batch.cam_intr4x4,
                cams_tform4x4_obj=batch.cam_tform4x4_obj,
                imgs_sizes=batch.size,
                objects_ids=batch.category_id,
                down_sample_rate=self.down_sample_rate,
                modalities=[
                    PROJECT_MODALITIES.PXL2D,
                    PROJECT_MODALITIES.MASK,
                    PROJECT_MODALITIES.IMG,
                ],
                instance_deform=instance_deform,
            )
            vts2d, vts2d_mask = (
                sample1d_mods[PROJECT_MODALITIES.PXL2D],
                sample1d_mods[PROJECT_MODALITIES.MASK],
            )
            net_feats = sample1d_mods[PROJECT_MODALITIES.IMG]
            N = vts2d.shape[1]
            C = net_feats.shape[2]

            #
            # vts2d_feats2d_net_mask = sample_pxl2d_pts(
            #     feats2d_net_mask,
            #     pxl2d=torch.cat([vts2d], dim=1),
            # )
            # vts2d_mask = vts2d_mask * (vts2d_feats2d_net_mask[:, :, 0] > 0.5)
            # net_feats = sample_pxl2d_pts(
            #     feats2d_net,
            #     pxl2d=torch.cat([vts2d, noise2d], dim=1),
            # )
            #

            if (
                VISUAL_MODALITIES.LATENT_INTERP in modalities
                and instance_deform is not None
                and instance_deform.latent is not None
            ):
                L = 4  # int(latent_scale * 2)
                latent_interp = torch.stack(
                    [instance_deform.latent, instance_deform.latent.roll(1, dims=(0,))],
                    dim=-1,
                )
                latent_interp = torch.nn.functional.interpolate(
                    latent_interp,
                    scale_factor=L / 2.0,
                    mode="linear",
                )

                latent_interp = latent_interp.permute(0, 2, 1).reshape(B * L, -1)
                objects_ids_interp = batch.category_id.repeat_interleave(L)
                feats2d_net_interp = feats2d_net.repeat_interleave(L, dim=0)
                backbone_out.latent = latent_interp
                instance_deform_interp = self.meshes.get_instance_deform(
                    backbone_out,
                    img_feats_canonical=feats2d_net_interp,
                    objects_ids=objects_ids_interp,
                )
                from od3d.cv.visual.show import (
                    get_default_camera_intrinsics_from_img_size,
                )
                from od3d.cv.geometry.transform import (
                    get_cam_tform4x4_obj_for_viewpoints_count,
                )

                V = 3
                cam_tform4x4_obj_interp = get_cam_tform4x4_obj_for_viewpoints_count(
                    viewpoints_count=V,
                    dist=5.0,
                )
                cam_intr4x4_interp = get_default_camera_intrinsics_from_img_size(
                    H=batch.size[0].item(),
                    W=batch.size[0].item(),
                )  # batch.cam_intr4x4[:1].repeat(V, 1, 1)

                ncds = self.meshes.render(
                    cams_tform4x4_obj=cam_tform4x4_obj_interp.to(self.device),
                    cams_intr4x4=cam_intr4x4_interp.to(self.device),
                    imgs_sizes=batch.size,
                    objects_ids=objects_ids_interp,
                    down_sample_rate=down_sample_rate,
                    broadcast_batch_and_cams=True,
                    modalities=PROJECT_MODALITIES.PT3D_NCDS,
                    instance_deform=instance_deform_interp,
                )
                ncds = (
                    ncds.reshape(B, L, V, *ncds.shape[-3:])
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(
                        B * V,
                        L,
                        *ncds.shape[-3:],
                    )
                )

                ncds_rgb_interp_1 = resize(
                    (batch.rgb - batch.rgb.min())
                    / ((batch.rgb.max() - batch.rgb.min()) + 1e-10),
                    H_out=ncds.shape[-2],
                    W_out=ncds.shape[-1],
                )
                ncds_rgb_interp_2 = ncds_rgb_interp_1.roll(1, dims=(0,)).clone()

                ncds_rgb_interp_1 = ncds_rgb_interp_1[:, None].repeat_interleave(
                    V,
                    dim=0,
                )
                ncds_rgb_interp_2 = ncds_rgb_interp_2[:, None].repeat_interleave(
                    V,
                    dim=0,
                )

                ncds = torch.cat([ncds_rgb_interp_1, ncds, ncds_rgb_interp_2], dim=1)
                # show_imgs(ncds)

                logger.info("upload latent interp...")
                for b in range(B):
                    results_batch_visual[
                        f"visual/{VISUAL_MODALITIES.LATENT_INTERP}/{batch_sel_names[b]}"
                    ] = image_as_wandb_image(
                        imgs_to_img(ncds[b * V : (b + 1) * V], pad_value=1.0),
                        caption=f"{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}",
                    )

            if VISUAL_MODALITIES.MESH in modalities:
                logger.info("upload meshes...")

                pred_o3d_meshes = self.meshes.get_rgb_meshes_as_list_of_o3d(
                    category_id=batch.category_id,
                    instance_deform=instance_deform,
                    device=self.device,
                )

                pred_o3d_meshes_cat = self.meshes.get_rgb_meshes_as_list_of_o3d(
                    category_id=batch.category_id,
                    instance_deform=None,
                    device=self.device,
                )

                gt_o3d_meshes = batch.mesh.get_rgb_meshes_as_list_of_o3d(
                    device=self.device,
                )

                for b in range(B):
                    offset_pred_cat = (
                        np.array(pred_o3d_meshes[b].vertices)[:, 2].max()
                        - np.array(pred_o3d_meshes_cat[b].vertices)[:, 2].min()
                    ) * 1.4
                    offset_gt = (
                        offset_pred_cat
                        + (
                            np.array(pred_o3d_meshes_cat[b].vertices)[:, 2].max()
                            - np.array(gt_o3d_meshes[b].vertices)[:, 2].min()
                        )
                        * 1.4
                    )

                    pred_cat_o3d_meshes_vertices = np.array(
                        pred_o3d_meshes_cat[b].vertices,
                    )
                    pred_cat_o3d_meshes_vertices[:, 2] += offset_pred_cat
                    pred_o3d_meshes_cat[b].vertices = open3d.utility.Vector3dVector(
                        pred_cat_o3d_meshes_vertices,
                    )
                    pred_vertices_count = pred_cat_o3d_meshes_vertices.shape[0]

                    gt_o3d_meshes_vertices = np.array(gt_o3d_meshes[b].vertices)
                    gt_o3d_meshes_vertices[:, 2] += offset_gt
                    gt_o3d_meshes[b].vertices = open3d.utility.Vector3dVector(
                        gt_o3d_meshes_vertices,
                    )
                    gt_vertices_count = gt_o3d_meshes_vertices.shape[0]

                    pred_o3d_meshes[b] += gt_o3d_meshes[b]
                    pred_o3d_meshes[b] += pred_o3d_meshes_cat[b]

                    from od3d.cv.geometry.transform import (
                        get_ico_cam_tform4x4_obj_for_viewpoints_count,
                    )
                    from od3d.cv.visual.show import (
                        get_default_camera_intrinsics_from_img_size,
                        show_imgs,
                    )

                    _mesh = Meshes.from_o3d(pred_o3d_meshes[b], device=self.device)
                    _mesh.verts = (
                        _mesh.verts.clone()
                        - _mesh.get_limits().max(dim=0).values.sum(dim=0) / 2
                    )
                    obj_scale = _mesh.get_range1d()

                    imgs_sizes = batch.size / down_sample_rate * 2
                    H = imgs_sizes[0]
                    W = imgs_sizes[1]
                    cams_tform4x4_obj = get_ico_cam_tform4x4_obj_for_viewpoints_count(
                        viewpoints_count=8,
                        radius=4.0 * obj_scale.cpu(),
                        theta_count=1,
                        viewpoints_uniform=True,
                        theta_uniform=True,
                    ).to(device=_mesh.device)
                    cam_intr4x4 = get_default_camera_intrinsics_from_img_size(
                        W=W,
                        H=H,
                    ).to(_mesh.device)

                    imgs = _mesh.render(
                        cams_tform4x4_obj=cams_tform4x4_obj,
                        cams_intr4x4=cam_intr4x4,
                        imgs_sizes=imgs_sizes,
                        modalities=PROJECT_MODALITIES.RGB,
                        broadcast_batch_and_cams=True,
                    )

                    feats = self.meshes.render(
                        cams_tform4x4_obj=cams_tform4x4_obj,
                        cams_intr4x4=cam_intr4x4,
                        imgs_sizes=imgs_sizes,
                        modalities=PROJECT_MODALITIES.FEATS,
                        broadcast_batch_and_cams=True,
                    )

                    from od3d.cv.cluster.embed import pca

                    feats_rgb = pca(feats.permute(0, 1, 3, 4, 2), C=3).permute(
                        0,
                        1,
                        4,
                        2,
                        3,
                    )
                    # show_imgs(imgs)

                    results_batch_visual[
                        f"visual/{VISUAL_MODALITIES.MESH}/{batch_sel_names[b]}"
                    ] = image_as_wandb_image(
                        img=imgs_to_img(imgs[0]),
                        caption=f"vertices count: pred={pred_vertices_count}, gt={gt_vertices_count}",
                    )

                    results_batch_visual[
                        f"visual/{VISUAL_MODALITIES.MESH_FEATS}/{batch_sel_names[b]}"
                    ] = image_as_wandb_image(
                        img=imgs_to_img(feats_rgb[0]),
                        caption=f"vertices count: pred={pred_vertices_count}, gt={gt_vertices_count}",
                    )

                    # import plotly.graph_objects as go
                    # fig = go.Figure(layout= {"title": f"vertices count: pred={pred_vertices_count}, gt={gt_vertices_count}"}, data=[
                    #     go.Mesh3d(
                    #         name=batch_sel_names[b],
                    #         x=np.asarray(pred_o3d_meshes[b].vertices)[:, 0],
                    #         y=np.asarray(pred_o3d_meshes[b].vertices)[:, 1],
                    #         z=np.asarray(pred_o3d_meshes[b].vertices)[:, 2],
                    #         i=np.asarray(pred_o3d_meshes[b].triangles)[:, 0],
                    #         j=np.asarray(pred_o3d_meshes[b].triangles)[:, 1],
                    #         k=np.asarray(pred_o3d_meshes[b].triangles)[:, 2],
                    #         vertexcolor=np.asarray(pred_o3d_meshes[b].vertex_colors),
                    #         showscale=True)])
                    #
                    # # fig.show(renderer='browser')
                    # results_batch_visual[
                    #     f"visual/{VISUAL_MODALITIES.MESH}/{batch_sel_names[b]}"
                    # ] = wandb.Plotly(fig)

                    # # Create a temporary file
                    # with tempfile.NamedTemporaryFile(delete=False,
                    #                                  suffix=".obj") as temp_file_pred:  # obj,gltf glb babylon stl,
                    #     temp_filename_pred = temp_file_pred.name
                    #     open3d.io.write_triangle_mesh(temp_filename_pred, pred_o3d_meshes[b])
                    #     # open3d.visualization.draw_geometries([pred_o3d_meshes[b]])
                    #     results_batch_visual[
                    #         f"visual/{VISUAL_MODALITIES.MESH}/{batch_sel_names[b]}"
                    #     ] = [wandb.Object3D(open(temp_filename_pred))]
                    #
                    #     results_batch_visual[
                    #         f"visual/{VISUAL_MODALITIES.MESH}/{batch_sel_names[b]}_text"
                    #     ] = image_as_wandb_image(img=torch.zeros(3, 10, 10), caption=f"vertices count: pred={pred_vertices_count}, gt={gt_vertices_count}")

            if VISUAL_MODALITIES.NET_FEATS_NEAREST_VERTS in modalities:
                logger.info("create net_feats_nearest_verts ...")
                nearest_pt3d_ncds = self.meshes.sample_nearest_to_feats2d_img(
                    feats2d_img=feats2d_net,
                    objects_ids=batch.category_id,
                    modalities=PROJECT_MODALITIES.PT3D_NCDS,
                    add_clutter=True,
                    instance_deform=instance_deform,
                )

                # verts3d = self.get_nearest_verts3d_to_feats2d_net(
                #    feats2d_net=feats2d_net,
                #    categories_ids=batch.category_id,
                #    zero_if_sim_clutter_larger=True,
                # )
                nearest_pt3d_ncds = resize(
                    nearest_pt3d_ncds,
                    scale_factor=self.down_sample_rate / down_sample_rate,
                )
                for b in range(len(batch)):
                    img = blend_rgb(
                        resize(batch.rgb[b], scale_factor=1.0 / down_sample_rate),
                        nearest_pt3d_ncds[b],
                    )
                    results_batch_visual[
                        f"visual/{VISUAL_MODALITIES.NET_FEATS_NEAREST_VERTS}/{batch_sel_names[b]}"
                    ] = image_as_wandb_image(
                        img,
                        caption=f"{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}",
                    )
                    if live:
                        show_img(img)

            if VISUAL_MODALITIES.SAMPLES in modalities:
                logger.info("create samples ...")
                s_cam_tform4x4_obj = results_batch["samples_cam_tform4x4_obj"].to(
                    device=self.device,
                )[batch_result_ids]
                s_cam_intr4x4 = results_batch["samples_cam_intr4x4"].to(
                    device=self.device,
                )[batch_result_ids]
                sim = results_batch["samples_sim"].to(device=self.device)[
                    batch_result_ids
                ]

                """
                s_cam_tform4x4_obj, s_cam_intr4x4 = self.get_samples(config_sample=self.config.inference.sample,
                                                                                           cam_intr4x4=batch.cam_intr4x4,
                                                                                           cam_tform4x4_obj=batch.cam_tform4x4_obj,
                                                                                           feats2d_net=feats2d_net,
                                                                                           categories_ids=batch.category_id)
                sim = self.get_sim_feats2d_net_with_cams(feats2d_net=feats2d_net,
                                                         cam_tform4x4_obj=s_cam_tform4x4_obj,
                                                         cam_intr4x4=s_cam_intr4x4,
                                                         categories_ids=batch.category_id,
                                                         broadcast_batch_and_cams=True)
                """

                ncds = self.meshes.render(
                    cams_tform4x4_obj=s_cam_tform4x4_obj,
                    cams_intr4x4=s_cam_intr4x4,
                    imgs_sizes=batch.size,
                    objects_ids=batch.category_id,
                    down_sample_rate=down_sample_rate,
                    broadcast_batch_and_cams=True,
                    modalities=PROJECT_MODALITIES.PT3D_NCDS,
                    instance_deform=instance_deform,
                )

                for b in range(len(batch)):
                    imgs = ncds[b]
                    imgs_sim = sim[b][:].expand(
                        *sim[b].shape,
                    )  # , *mesh_feats2d_rendered.shape[-2:]
                    if "uniform" in self.config.inference.sample.method:
                        imgs = imgs.reshape(
                            self.config.inference.sample.uniform.azim.steps,
                            self.config.inference.sample.uniform.elev.steps,
                            self.config.inference.sample.uniform.theta.steps,
                            *imgs.shape[-3:],
                        )[:, :, :]
                        imgs_sim = imgs_sim.reshape(
                            self.config.inference.sample.uniform.azim.steps,
                            self.config.inference.sample.uniform.elev.steps,
                            self.config.inference.sample.uniform.theta.steps,
                        )[:, :, :]
                    imgs = blend_rgb(
                        resize(batch.rgb[b], scale_factor=1.0 / down_sample_rate),
                        imgs,
                    )

                    if samples_sorted:
                        imgs_sim = imgs_sim.flatten(0)
                        imgs = imgs.reshape(-1, *imgs.shape[-3:])
                        imgs_sim_ids = imgs_sim.sort(descending=True)[1]
                        imgs_sim_ids = imgs_sim_ids[:49]
                        imgs = imgs[imgs_sim_ids]
                        imgs_sim = imgs_sim[imgs_sim_ids]

                    if samples_scores:
                        logger.info("create plot samples scores...")

                        plt.ioff()
                        fig, ax = plt.subplots()
                        ax.plot(
                            imgs_sim.detach().cpu().numpy(),
                            label="sim",
                        )  # density=False would make counts
                        # ax.set_ylim(0., 1.)
                        # ax.ylabel('sim')
                        # ax.xlabel('samples')

                        img = get_img_from_plot(ax=ax, fig=fig)
                        plt.close(fig)
                        img = resize(img, H_out=imgs.shape[-2], W_out=imgs.shape[-1])
                        img = draw_text_in_rgb(
                            img,
                            fontScale=0.4,
                            lineThickness=2,
                            fontColor=(0, 0, 0),
                            text=f"{batch_sel_scores[b]}\nmin={imgs_sim.min().item():.3f}\nmax={imgs_sim.max().item():.3f}",
                        )
                        imgs = torch.cat(
                            [imgs, img[None,].to(device=imgs.device)],
                            dim=0,
                        )
                        # resize(img, )
                        """
                        samples_score_size = imgs.shape[-1] // 5
                        imgs[..., -samples_score_size:, -samples_score_size:] = (
                                255 * imgs_sim.reshape(*imgs_sim.shape, 1, 1, 1).expand(*imgs.shape[:-2],
                                                                                        samples_score_size,
                                                                                        samples_score_size)).to(
                            torch.uint8)
                        """

                    img = imgs_to_img(imgs)
                    results_batch_visual[
                        f"visual/{VISUAL_MODALITIES.SAMPLES}/{batch_sel_names[b]}"
                    ] = image_as_wandb_image(
                        img,
                        caption=f"{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}, min={imgs_sim.min().item():.3f}, max={imgs_sim.max().item():.3f}",
                    )
                    if live:
                        show_img(img)

            if VISUAL_MODALITIES.SIM_PXL in modalities:
                logger.info("create sim pxl...")
                batch_pred_label = results_batch["label_pred"].to(device=self.device)[
                    batch_result_ids
                ]
                batch_pred_cam_tform4x4 = results_batch["cam_tform4x4_obj"].to(
                    device=self.device,
                )[batch_result_ids]

                sim, sim_pxl = self.meshes.get_sim_render(
                    feats2d_img=feats2d_net,
                    cams_intr4x4=batch.cam_intr4x4,
                    cams_tform4x4_obj=batch_pred_cam_tform4x4,
                    objects_ids=batch.category_id,
                    return_sim_pxl=True,
                    broadcast_batch_and_cams=False,
                    allow_clutter=self.config.inference.allow_clutter,
                    add_clutter=True,
                    instance_deform=instance_deform,
                    normalize_surface=self.config.inference.normalize_surface,
                    object_mask=batch.mask
                    if self.config.inference.add_mask_object_to_sim
                    else None,
                )

                sim_pxl = resize(
                    sim_pxl,
                    scale_factor=self.down_sample_rate / down_sample_rate,
                )
                for b in range(len(batch)):
                    img = blend_rgb(
                        resize(batch.rgb[b], scale_factor=1.0 / down_sample_rate),
                        sim_pxl[b],
                    )
                    results_batch_visual[
                        f"visual/{VISUAL_MODALITIES.SIM_PXL}/{batch_sel_names[b]}"
                    ] = image_as_wandb_image(
                        img,
                        caption=f"{batch_sel_names[b]}, {batch_names[b]}, mean sim={sim[b].item()}",
                    )
                    if live:
                        show_img(img)

            if (
                VISUAL_MODALITIES.PRED_VERTS_NCDS_IN_RGB in modalities
                or VISUAL_MODALITIES.PRED_VS_GT_VERTS_NCDS_IN_RGB in modalities
            ):
                logger.info("create pred verts ncds...")
                batch_pred_label = results_batch["label_pred"].to(device=self.device)[
                    batch_result_ids
                ]
                batch_pred_cam_tform4x4 = results_batch["cam_tform4x4_obj"].to(
                    device=self.device,
                )[batch_result_ids]

                pred_verts_ncds = self.meshes.render(
                    cams_tform4x4_obj=batch_pred_cam_tform4x4,
                    cams_intr4x4=batch.cam_intr4x4,
                    imgs_sizes=batch.size,
                    objects_ids=batch.category_id,
                    down_sample_rate=down_sample_rate,
                    broadcast_batch_and_cams=False,
                    modalities=PROJECT_MODALITIES.PT3D_NCDS,
                    instance_deform=instance_deform,
                )

                if VISUAL_MODALITIES.PRED_VERTS_NCDS_IN_RGB in modalities:
                    for b in range(len(batch)):
                        img = blend_rgb(
                            resize(batch.rgb[b], scale_factor=1.0 / down_sample_rate),
                            pred_verts_ncds[b],
                        )
                        results_batch_visual[
                            f"visual/{VISUAL_MODALITIES.PRED_VERTS_NCDS_IN_RGB}/{batch_sel_names[b]}"
                        ] = image_as_wandb_image(
                            img,
                            caption=f"{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}",
                        )
                        if live:
                            show_img(img)

            if (
                VISUAL_MODALITIES.GT_VERTS_NCDS_IN_RGB in modalities
                or VISUAL_MODALITIES.PRED_VS_GT_VERTS_NCDS_IN_RGB in modalities
            ):
                logger.info("create gt verts ncds...")

                gt_verts_ncds = self.meshes.render(
                    cams_tform4x4_obj=batch.cam_tform4x4_obj,
                    cams_intr4x4=batch.cam_intr4x4,
                    imgs_sizes=batch.size,
                    objects_ids=batch.category_id,
                    down_sample_rate=down_sample_rate,
                    broadcast_batch_and_cams=False,
                    modalities=PROJECT_MODALITIES.PT3D_NCDS,
                    instance_deform=instance_deform,
                )

                if VISUAL_MODALITIES.GT_VERTS_NCDS_IN_RGB in modalities:
                    for b in range(len(batch)):
                        img = blend_rgb(
                            resize(batch.rgb[b], scale_factor=1.0 / down_sample_rate),
                            gt_verts_ncds[b],
                        )
                        if (
                            "noise2d" in results_batch.keys()
                            and results_batch["noise2d"] is not None
                        ):
                            from od3d.cv.visual.draw import draw_pixels

                            img = draw_pixels(
                                img,
                                batch.noise2d[b]
                                * (self.down_sample_rate / down_sample_rate),
                            )

                        results_batch_visual[
                            f"visual/{VISUAL_MODALITIES.GT_VERTS_NCDS_IN_RGB}/{batch_sel_names[b]}"
                        ] = image_as_wandb_image(
                            img,
                            caption=f"{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}",
                        )
                        if live:
                            show_img(img)
            if VISUAL_MODALITIES.PRED_VS_GT_VERTS_NCDS_IN_RGB in modalities:
                logger.info("create pred vs gt verts ncds...")
                for b in range(len(batch)):
                    # if 'noise2d' in results
                    img1 = blend_rgb(
                        resize(batch.rgb[b], scale_factor=1.0 / down_sample_rate),
                        pred_verts_ncds[b],
                    )
                    img2 = blend_rgb(
                        resize(batch.rgb[b], scale_factor=1.0 / down_sample_rate),
                        gt_verts_ncds[b],
                    )
                    img = imgs_to_img(torch.stack([img1, img2], dim=0)[None,])

                    results_batch_visual[
                        f"visual/{VISUAL_MODALITIES.PRED_VS_GT_VERTS_NCDS_IN_RGB}/{batch_sel_names[b]}"
                    ] = image_as_wandb_image(
                        img,
                        caption=f"{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}",
                    )
                    if live:
                        show_img(img)
            if VISUAL_MODALITIES.TSNE_PER_IMAGE in modalities:
                logger.info("create tsne plots for the mesh and image features...")
                from od3d.cv.cluster.embed import tsne

                # normalize feats2d_net

                # fg_feats = torch.masked_select(feats2d_net, feats2d_net_mask >= 0.5).view(feats2d_net.shape[0],feats2d_net.shape[1],-1).permute(0,2,1)
                color_ = plt.get_cmap("tab20", len(self.meshes))

                # bg_feats = torch.masked_select(feats2d_net, feats2d_net_mask < 0.5).view(feats2d_net.shape[0],feats2d_net.shape[1],-1).permute(0,2,1)

                for b in range(len(batch)):
                    mesh_and_image_feats_colors = self.meshes.feats_rgb_object_id.copy()
                    mesh_and_image_feats_length = [self.meshes.feats_objects.shape[0]]
                    fg_feats = net_feats[b, :N][vts2d_mask[b]]
                    bg_feats = net_feats[b, N:].reshape(-1, C)
                    print(fg_feats.shape, bg_feats.shape)
                    mesh_and_image_feats_colors.extend(
                        [color_(batch.category_id[b].cpu().numpy())]
                        * fg_feats.shape[0],
                    )
                    mesh_and_image_feats_colors.extend(
                        [(0, 0, 0, 1)] * bg_feats.shape[0],
                    )
                    mesh_and_image_feats_length.extend(
                        [fg_feats.shape[0], bg_feats.shape[0]],
                    )
                    feats_tsne_all = tsne(
                        torch.cat(
                            [self.meshes.feats_objects, fg_feats, bg_feats],
                            dim=0,
                        ),
                        C=2,
                    )
                    print(len(mesh_and_image_feats_colors))
                    print(mesh_and_image_feats_length)

                    img = show_scene2d(
                        [feats_tsne_all],
                        pts2d_colors=[mesh_and_image_feats_colors],
                        pts2d_lengths=mesh_and_image_feats_length,
                        return_visualization=True,
                    )

                    results_batch_visual[
                        f"visual/{VISUAL_MODALITIES.TSNE_PER_IMAGE}/{batch_sel_names[b]}"
                    ] = image_as_wandb_image(
                        img,
                        caption=f"{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}",
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
        sel_names = (
            epoch_best_names + epoch_worst_names + epoch_rand_names + epoch_sel_names
        )
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

        if VISUAL_MODALITIES.TSNE in modalities:
            logger.info("create tsne plots for the mesh...")
            from od3d.cv.cluster.embed import tsne

            feats_tsne = tsne(self.meshes.feats_objects, C=2)

            img = show_scene2d(
                [feats_tsne],
                pts2d_colors=[self.meshes.feats_rgb_object_id],
                return_visualization=True,
            )

            results[f"visual/{VISUAL_MODALITIES.TSNE}"] = image_as_wandb_image(
                img,
                caption=f"tsne of mesh feats",
            )

        if VISUAL_MODALITIES.PCA in modalities:
            logger.info("create pca plots for the mesh...")
            from od3d.cv.cluster.embed import pca

            feats_pca = pca(self.meshes.feats_objects, C=2)

            img = show_scene2d(
                [feats_pca],
                pts2d_colors=[self.meshes.feats_rgb_object_id],
                return_visualization=True,
            )

            results[f"visual/{VISUAL_MODALITIES.PCA}"] = image_as_wandb_image(
                img,
                caption=f"PCA of mesh feats",
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

    def pred_poses(
        self,
        backbone_out,
        cam_intr4x4: torch.Tensor,
        size: torch.Tensor,
        cam_tform4x4_obj: torch.Tensor = None,
    ):
        from torch.nn import Softplus

        softplus = Softplus(beta=2 * np.log(2))

        from od3d.cv.geometry.transform import (
            cam_intr4x4_2_rays3d,
            cam_intr4x4_downsample,
            inv_tform4x4,
        )

        backbone_out = copy.deepcopy(backbone_out)
        cam_intr4x4_down, img_size_down = cam_intr4x4_downsample(
            cams_intr4x4=cam_intr4x4,
            imgs_sizes=size,
            down_sample_rate=self.net.backbone.downsample_rate,
        )
        rays3d = cam_intr4x4_2_rays3d(cam_intr4x4_down, img_size_down)

        # PxNx2x3
        # lines = torch.stack([rays3d[0].flatten(1).permute(1, 0) * 0, 3 * rays3d[0].flatten(1).permute(1, 0)], dim=-2)
        # lines = transf3d_broadcast(pts3d=lines, transf4x4=inv_tform4x4(cam_tform4x4_obj[0]))
        # from od3d.cv.visual.show import show_scene
        # show_scene(meshes=self.meshes, lines3d=[lines.detach().cpu()], cams_tform4x4_world=[cam_tform4x4_obj.detach().cpu()[0]], cams_intr4x4=[cam_intr4x4.detach().cpu()[0]])
        # show_scene(show_coordinate_frame=True, cams_tform4x4_world=poses_tform4x4[0].detach().cpu(), cams_intr4x4=cam_intr4x4[:1].repeat(poses_tform4x4[0].shape[0], 1, 1).detach().cpu())

        backbone_out.featmaps[-1] = torch.cat(
            [backbone_out.featmaps[-1], rays3d],
            dim=-3,
        )
        # feats2d_net = backbone_out.feats_map
        poses = self.net_pose(backbone_out)
        poses = poses.feat.reshape(
            poses.feat.shape[0],
            self.net_pose_translations,
            self.net_pose_rotations,
            self.net_pose_rotation_dim + 3,
        )
        # [B, T, R, 9])
        # rot 6d to matrix, [x, y, z], x=lookat, y=up
        # zero dim: [False, False, False, False, False, False]
        # add othant: [True, True, True, False, True, False]
        # 2 ** 6 # 64
        poses_rot6d = poses[..., -6:].clone()
        # poses_rot6d_othants = poses_rot6d[..., self.net_pose_rotations_othant]
        # poses_rot6d_othants = softplus(poses_rot6d_othants)

        poses_rot6d[..., self.net_pose_rotations_othant] = softplus(
            poses_rot6d[..., self.net_pose_rotations_othant],
        ).clone()

        # self.net_pose_rotations_othant_signs #R x 9
        poses_rot6d = poses_rot6d * self.net_pose_rotations_othant_signs[None, None].to(
            device=self.device,
        )
        # poses[..., -6:] = poses.clone()[..., -6:] * self.net_pose_rotations_othant_signs[None, None].to(device=self.device)

        poses = torch.cat([poses[..., :3], poses_rot6d], dim=-1)

        from od3d.cv.geometry.transform import (
            transf4x4_from_rot3x3_and_transl3,
            rotation_6d_to_matrix,
            cam_intr4x4_to_cam_intr_ncds4x4,
            inv_cam_intr4x4,
            inv_tform4x4,
        )

        # cam_intr_ncds4x4 = cam_intr4x4_to_cam_intr_ncds4x4(cam_intr4x4=cam_intr4x4, size=size)
        # cam_intr_ncds_inv4x4 = inv_cam_intr4x4(cam_intr_ncds4x4)
        # poses[..., :3] = transf3d_broadcast(transf4x4=cam_intr_ncds_inv4x4[:, None, None], pts3d=poses[..., :3])

        # poses[..., :3] = poses[..., :3] * 0
        # poses[..., 2] = 10.
        # poses[..., -3] = 0.
        # poses[..., -2] = 1.
        # poses[..., -1] = 0.

        poses_rot3x3 = rotation_6d_to_matrix(poses[..., -6:])
        poses_tform4x4 = transf4x4_from_rot3x3_and_transl3(
            rot3x3=poses_rot3x3,
            transl3=poses[..., :3],
        )

        poses_tform4x4 = poses_tform4x4.reshape(
            poses_tform4x4.shape[0],
            -1,
            *poses_tform4x4.shape[-2:],
        )

        # from od3d.cv.visual.show import show_scene
        # show_scene(cam_intr4x4=cam_intr4x4[:1], cam_)
        # show_scene(show_coordinate_frame=True, cams_tform4x4_world=poses_tform4x4[0].detach().cpu(), cams_intr4x4=cam_intr4x4[:1].repeat(poses_tform4x4[0].shape[0], 1, 1).detach().cpu())

        return poses_tform4x4

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
                pred_cam_tform4x4_objs=pred_cam_tform4x4_objs_first,
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
                tform4x4_broadcast(
                    inv_tform4x4(cam_tform4x4_obj[:1, None]),
                    b_cams_multiview_tform4x4_obj,
                ),
            )

            b_cams_multiview_intr4x4 = cam_intr4x4[:, None].expand(
                *b_cams_multiview_tform4x4_obj.shape,
            )

        else:
            B = len(feats2d_net)
            if config_sample.method == "pred":
                C = pred_cam_tform4x4_objs.shape[1]
                b_cams_multiview_tform4x4_obj = pred_cam_tform4x4_objs.clone()
                # assumption 1: distance translation to object is known
                # b_cams_multiview_tform4x4_obj[:, :, 2, 3] = cam_tform4x4_obj[:, None].repeat(1, C, 1, 1)[:, :, 2, 3]
                # logger.info(f'dist {batch.cam_tform4x4_obj[:, 2, 3]}')
                # assumption 2: translation to object is known
                # b_cams_multiview_tform4x4_obj[:, :, :4, 3] = cam_tform4x4_obj[
                #    :,
                #    None,
                # ].repeat(1, C, 1, 1)[:, :, :4, 3]

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

                if "pred" in config_sample.method:
                    b_cams_multiview_tform4x4_obj[:, :, :3, 3] = pred_cam_tform4x4_objs.clone()[:, :, :3, 3]

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
                    coarse_labels=coarse_labels,
                )

                H, W = feats2d_net.shape[-2:]
                # sim_feats[sim_feats == 0.] = -torch.inf
                sim_feats_top_vals, sim_feats_top_ids = sim_feats[:, :].sort(
                    dim=1,
                    descending=True,
                )
                sim_feats_selected = torch.ones_like(sim_feats)
                sim_feats_selected[:] = -torch.inf

                # add top
                pt_nn_top_k = config_sample.epnp3d2d.pt_nn_top_k
                sim_feats_selected = torch.scatter(
                    input=sim_feats_selected,
                    dim=1,
                    index=sim_feats_top_ids[:, :pt_nn_top_k],
                    src=sim_feats_top_vals[:, :pt_nn_top_k],
                )
                # add clutter
                sim_feats_selected[:, -1:] = sim_feats[:, -1:]

                modality_pt3d = (
                    PROJECT_MODALITIES.PT3D
                    if not coarse_labels
                    else PROJECT_MODALITIES.PT3D_COARSE
                )
                pts3d = self.meshes.sample(
                    objects_ids=categories_ids,
                    modalities=modality_pt3d,
                    add_clutter=False,
                    add_other_objects=False,
                    instance_deform=instance_deform,
                )
                pts3d_prob = torch.softmax(sim_feats_selected, dim=1)[:, :-1].flatten(2)

                # from od3d.cv.visual.show import show_imgs
                # show_imgs(feats2d_net_mask)

                # from od3d.cv.visual.show import show_imgs
                # show_imgs(pts3d_prob.sum(dim=1).reshape(B, 1, H, W))

                if feats2d_net_mask is not None:
                    pts3d_prob = pts3d_prob * feats2d_net_mask.flatten(2)

                # B x F x N
                pts3d_prob[pts3d_prob.sum(dim=-1).sum(dim=-1) == 0.0] = 1.0
                K = config_sample.epnp3d2d.count_cams
                N = config_sample.epnp3d2d.count_pts
                if N == -1:
                    pts3d_prob[(pts3d_prob.sum(dim=-1) > 0).sum(dim=-1) < 4] = 1.0
                else:
                    pts3d_prob[(pts3d_prob.sum(dim=-1) > 0).sum(dim=-1) < N] = 1.0

                # rgb = torch.zeros_like(feats2d_net)[:, :3]
                # from od3d.cv.select import batched_indexMD_fill, batched_index_select
                # pts3d_plus_zero = torch.cat([pts3d, torch.zeros_like(pts3d[:, :1])], dim=1)
                # rgb = batched_index_select(input=pts3d_plus_zero, index=sim_feats_top_ids[:, 0].flatten(1), dim=1).clone().permute(0, 2, 1).reshape(-1, 3, *feats2d_net.shape[-2:])
                # #rgb = batched_indexMD_fill(inputMD=rgb, indexMD=sim_feats_top_ids[:, :1].permute(0, 2, 1)[..., None, :, :].expand(4, 3, 200, 2).long(), value=pts3d_pairs, dims=[2, 3])
                # #rgb = rgb.reshape(4, 3, 64, 64)
                # rgb = (rgb / (rgb.norm(dim=1, keepdim=True) + 1e-10)).clamp(0, 1)
                # from od3d.cv.visual.show import show_imgs
                # show_imgs(rgb)

                pxl2d = (
                    get_pxl2d_like(sim_feats.permute(0, 2, 3, 1))
                    .permute(
                        0,
                        3,
                        1,
                        2,
                    )
                    .flatten(2)
                )

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

                # SAMPLING SEQUENTIAL: First 3D
                # sample 1st 3D
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

                # from od3d.cv.visual.show import show_imgs
                # show_imgs(255 * grid_pxl2d_prob[:3].reshape(3, 1, H, W))

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
                        # pts3d_prob.flatten(1),
                        pts3d_prob[:, None]
                        .expand(pts3d_prob.shape[0], K, *pts3d_prob.shape[1:])
                        .reshape(pts3d_prob.shape[0] * K, -1),  # BxVxH*W
                        num_samples=N,
                        replacement=False,
                    ).reshape(pts3d_prob.shape[0], K, N)

                    pts3d_ids = pts3d2d_ids // count_pxl2d  # BxKxN
                    pxl2d_ids = pts3d2d_ids % count_pxl2d  # BxKxN
                    device = pts3d.device
                    dtype = pts3d.dtype
                    pts3d_pairs = torch.zeros(B, 3, K, N).to(device=device, dtype=dtype)
                    pts3d_pairs_prob = torch.zeros(B, K, K, N).to(
                        device=device,
                        dtype=dtype,
                    )
                    pxl2d_pairs = torch.zeros(B, 2, K, N).to(device=device, dtype=dtype)
                    pts3d2d_pairs_mask = (
                        torch.eye(K)
                        .to(device=device, dtype=bool)[None, :, :, None]
                        .expand(B, K, K, N)
                    )  # (B, K, K*N)
                    pts3d2d_pairs_mask = pts3d2d_pairs_mask.reshape(B, K, K * N)
                    for b in range(B):
                        pts3d_pairs[b] = pts3d[b, pts3d_ids[b], :].permute(2, 0, 1)
                        pxl2d_pairs[b] = pxl2d[b, :, pxl2d_ids[b]]
                        pts3d_pairs_prob[b] = (
                            pts3d_prob[b]
                            .flatten()[pts3d2d_ids[b]][None,]
                            .expand(K, K, N)
                        )

                    pts3d_pairs_prob = pts3d_pairs_prob.reshape(B, K, K * N)
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
                    pts3d_pairs = batched_index_select(
                        input=pts3d,
                        index=pts3d_pairs_ids,
                        dim=1,
                    )
                    pts3d_pairs = pts3d_pairs.permute(0, 2, 1)
                    pts3d_pairs = pts3d_pairs.reshape(*pts3d_pairs.shape[:2], H, W)
                    pts3d_pairs_prob = pts3d_pairs_prob.reshape(
                        *pts3d_pairs_prob.shape[:1],
                        1,
                        H,
                        W,
                    )
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
                    logger.info(f"NaNs in cam_tform_obj")
                    b_nans = (
                        b_cams_multiview_tform4x4_obj.flatten(2).isnan().any(dim=-1)
                    )
                    b_cams_multiview_tform4x4_obj[b_nans] = torch.eye(4).to(
                        device=b_nans.device,
                    )[None]
                if b_cams_multiview_tform4x4_obj.abs().max() > 1e5:
                    logger.info(f"Too large values in cam_tform_obj")
                    b_too_largs = (
                        b_cams_multiview_tform4x4_obj.flatten(2).abs() > 1e5
                    ).any(dim=-1)
                    b_cams_multiview_tform4x4_obj[b_too_largs] = torch.eye(4).to(
                        device=b_too_largs.device,
                    )[None]

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

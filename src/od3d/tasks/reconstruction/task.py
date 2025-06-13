import logging

logger = logging.getLogger(__name__)
from omegaconf import DictConfig
from od3d.tasks.task import OD3D_Task, OD3D_Metrics, OD3D_Visuals
from od3d.datasets.frame import OD3D_FRAME_MODALITIES
from od3d.cv.visual.resize import resize
import torch


class Reconstruction(OD3D_Task):
    metrics_supported = [
        OD3D_Metrics.REC_RGB_MSE,
        OD3D_Metrics.REC_RGB_PSNR,
        OD3D_Metrics.REC_MASK_MSE,
        OD3D_Metrics.REC_MASK_DOT,
        OD3D_Metrics.REC_MASK_IOU,
        OD3D_Metrics.REC_MASK_DT_DOT,
        # add REC_MASK_BCE
        OD3D_Metrics.REC_MASK_INV_DT_DOT,
        OD3D_Metrics.REC_MASK_AMODAL_MSE,
        OD3D_Metrics.REC_MASK_AMODAL_DOT,
        OD3D_Metrics.REC_MASK_AMODAL_IOU,
        OD3D_Metrics.REC_MESH_VERTS_COUNT,
        OD3D_Metrics.REC_MESH_FACES_COUNT,
    ]

    visuals_supported = [
        OD3D_Visuals.PRED_VS_GT_MASK,
        OD3D_Visuals.PRED_VS_GT_MASK_AMODAL,
        OD3D_Visuals.PRED_VS_GT_RGB,
        OD3D_Visuals.PRED_VS_GT_MESH,
    ]

    def __init__(
        self,
        metrics: dict[OD3D_Metrics, float],
        visuals: list[OD3D_Visuals],
        apply_mask_rgb_gt=False,
        apply_mask_rgb_pred=False,
    ):
        super().__init__(metrics=metrics, visuals=visuals)
        self.apply_mask_rgb_pred = apply_mask_rgb_pred
        self.apply_mask_rgb_gt = apply_mask_rgb_gt

    def calc_mask_iou(self, pred_mask, gt_mask):
        return (gt_mask * pred_mask).flatten(1).sum(dim=-1) / (
            (gt_mask * pred_mask)
            + ((1.0 - gt_mask) * pred_mask)
            + ((1.0 - pred_mask) * gt_mask)
        ).detach().flatten(1).sum(dim=-1) + 1e-10

    def calc_mask_dot(self, pred_mask, gt_mask):
        return (gt_mask * pred_mask).flatten(1).mean(dim=-1)

    def calc_mask_mse(self, pred_mask, gt_mask):
        return ((gt_mask - pred_mask) ** 2).flatten(1).mean(dim=-1)

    def calc_rgb_mse(self, pred_rgb, gt_rgb):
        # from od3d.cv.visual.show import show_img
        # show_img((gt_rgb - pred_rgb)[0])
        return (
            (pred_rgb - gt_rgb) ** 2
        ).mean()  # sum(dim=1)).mean(dim=-1).mean(dim=-1)

    def calc_rgb_psnr(self, pred_rgb, gt_rgb):
        return 10 * torch.log10(1.0 / self.calc_rgb_mse(pred_rgb, gt_rgb))

    def eval(
        self,
        frames_pred,
        frames_gt=None,
    ):
        if OD3D_Metrics.REC_MESH_VERTS_COUNT in self.metrics:
            frames_pred.verts_count = frames_pred.meshes.verts_count
        if OD3D_Metrics.REC_MESH_FACES_COUNT in self.metrics:
            frames_pred.faces_count = frames_pred.meshes.faces_count

        if (
            OD3D_Metrics.REC_RGB_MSE in self.metrics
            or OD3D_Metrics.REC_RGB_PSNR in self.metrics
        ):
            W_out = frames_pred.rgb.shape[-1]
            H_out = frames_pred.rgb.shape[-2]
            gt_rgb = resize(frames_gt.rgb, H_out=H_out, W_out=W_out)
            pred_rgb = resize(frames_pred.rgb, H_out=H_out, W_out=W_out)

            if OD3D_FRAME_MODALITIES.RGB_MASK in frames_gt.modalities:
                rgb_mask_resized = (
                    resize(frames_gt.rgb_mask, H_out=H_out, W_out=W_out) > 0.9
                )  # , mode='nearest_v2')
                gt_rgb = gt_rgb * rgb_mask_resized
                pred_rgb = pred_rgb * rgb_mask_resized

            if self.apply_mask_rgb_pred or self.apply_mask_rgb_gt:
                if OD3D_FRAME_MODALITIES.MASK in frames_gt.modalities:
                    mask_resized = (
                        resize(frames_gt.mask, H_out=H_out, W_out=W_out) > 0.5
                    )  # , mode='nearest_v2')
                    if self.apply_mask_rgb_gt:
                        gt_rgb = gt_rgb * mask_resized
                    if self.apply_mask_rgb_pred:
                        pred_rgb = pred_rgb * mask_resized
                else:
                    logger.warning(
                        "cannot apply mask for reconstruction loss as mask is not available in gt batch",
                    )

            if OD3D_Metrics.REC_RGB_MSE in self.metrics:
                frames_pred.rec_rgb_mse = self.calc_rgb_mse(
                    pred_rgb=pred_rgb,
                    gt_rgb=gt_rgb,
                )

            if OD3D_Metrics.REC_RGB_PSNR in self.metrics:
                frames_pred.rec_rgb_psnr = self.calc_rgb_psnr(
                    pred_rgb=pred_rgb,
                    gt_rgb=gt_rgb,
                )

        if (
            OD3D_Metrics.REC_MASK_MSE in self.metrics
            or OD3D_Metrics.REC_MASK_IOU in self.metrics
            or OD3D_Metrics.REC_MASK_DOT in self.metrics
            or OD3D_Metrics.REC_MASK_DT_DOT in self.metrics
            or OD3D_Metrics.REC_MASK_INV_DT_DOT in self.metrics
        ):
            pred_mask = frames_pred.mask
            W_out = pred_mask.shape[-1]
            H_out = pred_mask.shape[-2]
            gt_mask = resize(frames_gt.mask * 1.0, H_out=H_out, W_out=W_out)

            if OD3D_FRAME_MODALITIES.RGB_MASK in frames_gt.modalities:
                rgb_mask_resized = (
                    resize(frames_gt.rgb_mask, H_out=H_out, W_out=W_out) > 0.9
                )  # , mode='nearest_v2')
                gt_mask = gt_mask * rgb_mask_resized
                pred_mask = pred_mask * rgb_mask_resized
            else:
                rgb_mask_resized = None

            if OD3D_Metrics.REC_MASK_MSE in self.metrics:
                frames_pred.rec_mask_mse = self.calc_mask_mse(
                    pred_mask=pred_mask,
                    gt_mask=gt_mask,
                )
            if OD3D_Metrics.REC_MASK_IOU in self.metrics:
                frames_pred.rec_mask_iou = self.calc_mask_iou(
                    pred_mask=pred_mask,
                    gt_mask=gt_mask,
                )
            if OD3D_Metrics.REC_MASK_DOT in self.metrics:
                frames_pred.rec_mask_dot = self.calc_mask_dot(
                    pred_mask=pred_mask,
                    gt_mask=gt_mask,
                )

            if OD3D_Metrics.REC_MASK_DT_DOT in self.metrics:
                gt_masks_dt = resize(frames_gt.mask_dt, H_out=H_out, W_out=W_out)
                if rgb_mask_resized is not None:
                    gt_masks_dt = gt_masks_dt * rgb_mask_resized
                frames_pred.rec_mask_dt_dot = self.calc_mask_dot(
                    pred_mask=pred_mask,
                    gt_mask=gt_masks_dt,
                )

            if OD3D_Metrics.REC_MASK_INV_DT_DOT in self.metrics:
                gt_masks_inv_dt = resize(
                    frames_gt.mask_inv_dt,
                    H_out=H_out,
                    W_out=W_out,
                )
                if rgb_mask_resized is not None:
                    rgb_mask_resized = resize(
                        frames_gt.rgb_mask,
                        H_out=H_out,
                        W_out=W_out,
                        mode="nearest_v2",
                    )
                    gt_masks_inv_dt = gt_masks_inv_dt * rgb_mask_resized
                pred_masks_inv = 1.0 - pred_mask.clone()
                frames_pred.rec_mask_inv_dt_dot = self.calc_mask_dot(
                    pred_mask=pred_masks_inv,
                    gt_mask=gt_masks_inv_dt,
                )

        if (
            OD3D_Metrics.REC_CD_PCL in self.metrics
            or OD3D_Metrics.REC_PF_PCL in self.metrics
            or OD3D_Metrics.REC_PF_PCL_V2 in self.metrics
            or OD3D_Metrics.REC_PF_PCL_V2_CD_PCL in self.metrics
        ):
            B = len(frames_gt.pcl)
            device = frames_gt.pcl.device
            N = max([frames_gt.pcl[b].shape[0] for b in range(B)])
            gt_pts3d = torch.zeros((B, N, 3)).to(device=device)
            gt_pts3d_mask = torch.zeros((B, N), dtype=bool).to(device=device)
            for b in range(B):
                N_b = frames_gt.pcl[b].shape[0]
                gt_pts3d[b, :N_b] = frames_gt.pcl[b]
                gt_pts3d_mask[b, :N_b] = True

        if (
            OD3D_Metrics.REC_CD_MESH in self.metrics
            or OD3D_Metrics.REC_PF_MESH in self.metrics
            or OD3D_Metrics.REC_PF_MESH_V2 in self.metrics
            or OD3D_Metrics.REC_PF_MESH_V2_CD_MESH in self.metrics
        ):
            from od3d.cv.geometry.metrics.dist import (
                batch_chamfer_distance,
                batch_point_face_distance,
                batch_point_face_distance_v2,
            )

            device = frames_pred.mesh.device
            gt_meshes_verts = frames_gt.mesh.get_verts_stacked_with_mesh_ids().to(
                device=device,
            )
            gt_meshes_verts_mask = (
                frames_gt.mesh.get_verts_stacked_mask_with_mesh_ids().to(device=device)
            )

            # BxVx3
            pred_meshes_verts = frames_pred.mesh.get_verts_stacked_with_mesh_ids().to(
                device=device,
            )
            pred_meshes_verts_mask = (
                frames_pred.mesh.get_verts_stacked_mask_with_mesh_ids().to(
                    device=device,
                )
            )

            if (
                OD3D_Metrics.REC_PF_MESH in self.metrics
                or OD3D_Metrics.REC_PF_MESH_V2 in self.metrics
                or OD3D_Metrics.REC_PF_MESH_V2_CD_MESH in self.metrics
            ):
                pred_meshes_faces = (
                    frames_pred.mesh.get_faces_stacked_with_mesh_ids().to(device=device)
                )
                pred_meshes_faces_mask = (
                    frames_pred.mesh.get_faces_stacked_mask_with_mesh_ids().to(
                        device=device,
                    )
                )

            if OD3D_Metrics.REC_CD_MESH in self.metrics:
                frames_pred.rec_cd_mesh = batch_chamfer_distance(
                    pred_meshes_verts,
                    gt_meshes_verts,
                    pred_meshes_verts_mask,
                    gt_meshes_verts_mask,
                )

            if OD3D_Metrics.REC_PF_MESH in self.metrics:
                frames_pred.rec_pf_mesh = batch_point_face_distance(
                    verts1=pred_meshes_verts,
                    faces1=pred_meshes_faces,
                    faces1_mask=pred_meshes_faces_mask,
                    pts2=gt_meshes_verts,
                    verts1_mask=pred_meshes_verts_mask,
                    pts2_mask=gt_meshes_verts_mask,
                )

            if OD3D_Metrics.REC_PF_MESH_V2 in self.metrics:
                frames_pred.rec_pf_mesh_v2 = batch_point_face_distance_v2(
                    verts1=pred_meshes_verts,
                    faces1=pred_meshes_faces,
                    faces1_mask=pred_meshes_faces_mask,
                    pts2=gt_meshes_verts,
                    verts1_mask=pred_meshes_verts_mask,
                    pts2_mask=gt_meshes_verts_mask,
                )

            if OD3D_Metrics.REC_PF_MESH_V2_CD_MESH in self.metrics:
                frames_pred.rec_pf_mesh_v2_cd_mesh = batch_chamfer_distance(
                    gt_meshes_verts,
                    pred_meshes_verts,
                    gt_meshes_verts_mask,
                    pred_meshes_verts_mask,
                    only_pts2_nn=True,
                    uniform_weight_pts1=False,
                )
                frames_pred.rec_pf_mesh_v2_cd_mesh += batch_point_face_distance_v2(
                    verts1=pred_meshes_verts,
                    faces1=pred_meshes_faces,
                    faces1_mask=pred_meshes_faces_mask,
                    pts2=gt_meshes_verts,
                    verts1_mask=pred_meshes_verts_mask,
                    pts2_mask=gt_meshes_verts_mask,
                )

        return frames_pred

    def visualize(
        self,
        frames_pred,
        frames_gt=None,
    ):
        if OD3D_Visuals.PRED_VS_GT_RGB in self.visuals:
            pred_rgb = resize(
                frames_pred.rgb,
                H_out=self.visuals_res,
                W_out=self.visuals_res,
            ).to(device=frames_gt.rgb.device)
            gt_rgb = resize(
                frames_gt.rgb,
                H_out=self.visuals_res,
                W_out=self.visuals_res,
            )
            frames_pred.pred_vs_gt_rgb = torch.cat([pred_rgb, gt_rgb], dim=-1)

        if OD3D_Visuals.PRED_VS_GT_MASK in self.visuals:
            pred_mask = resize(
                frames_pred.mask,
                H_out=self.visuals_res,
                W_out=self.visuals_res,
            ).to(device=frames_gt.mask.device)
            gt_mask = resize(
                frames_gt.mask,
                H_out=self.visuals_res,
                W_out=self.visuals_res,
            )
            frames_pred.pred_vs_gt_mask = torch.cat([pred_mask, gt_mask], dim=-1)

        if OD3D_Visuals.PRED_VS_GT_MASK_AMODAL in self.visuals:
            pred_mask_amodal = resize(
                frames_pred.mask_amodal,
                H_out=self.visuals_res,
                W_out=self.visuals_res,
            ).to(device=frames_gt.mask_amodal.device)
            gt_mask_amodal = resize(
                frames_gt.mask_amodal,
                H_out=self.visuals_res,
                W_out=self.visuals_res,
            )
            frames_pred.pred_vs_gt_mask_amodal = torch.cat(
                [pred_mask_amodal, gt_mask_amodal],
                dim=-1,
            )

        if OD3D_Visuals.PRED_VS_GT_MESH in self.visuals:
            frames_pred.pred_vs_gt_mesh = frames_pred.mesh + frames_gt.mesh

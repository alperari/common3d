import logging

logger = logging.getLogger(__name__)
import abc
from omegaconf import DictConfig
import od3d.io
from pathlib import Path
import inspect
from od3d.data.ext_enum import StrEnum


class OD3D_Metrics(StrEnum):
    REC_RGB_MSE = "rec_rgb_mse"
    REC_RGB_PSNR = "rec_rgb_psnr"
    REC_MASK_MSE = "rec_mask_mse"
    REC_MASK_IOU = "rec_mask_iou"
    REC_MASK_DOT = "rec_mask_dot"
    REC_MASK_DT_DOT = "rec_mask_dt_dot"
    REC_MASK_INV_DT_DOT = "rec_mask_inv_dt_dot"
    REC_MASK_AMODAL_MSE = "rec_mask_amodal_mse"
    REC_MASK_AMODAL_IOU = "rec_mask_amodal_iou"
    REC_MASK_AMODAL_DOT = "rec_mask_amodal_dot"
    REC_CD_MESH = "rec_cd_mesh"
    REC_PF_MESH = "rec_pf_mesh"
    REC_PF_MESH_V2 = "rec_pf_mesh_v2"
    REC_PF_MESH_V2_CD_MESH = "rec_pf_mesh_v2_cd_mesh"
    REC_CD_PCL = "rec_cd_pcl"
    REC_PF_PCL = "rec_pf_pcl"
    REC_PF_PCL_V2 = "rec_pf_pcl_v2"
    REC_PF_PCL_V2_CD_PCL = "rec_pf_pcl_v2_cd_pcl"
    REC_MESH_VERTS_COUNT = "rec_mesh_verts_count"
    REC_MESH_FACES_COUNT = "rec_mesh_faces_count"


class OD3D_Visuals(StrEnum):
    PRED_VS_GT_MASK = "pred_vs_gt_mask"
    PRED_VS_GT_MASK_AMODAL = "pred_vs_gt_mask_amodal"
    PRED_VS_GT_RGB = "pred_vs_gt_rgb"
    PRED_VS_GT_MESH = "pred_vs_gt_mesh"

    PRED_VERTS_NCDS_IN_RGB = "pred_verts_ncds_in_rgb"
    GT_VERTS_NCDS_IN_RGB = "gt_verts_ncds_in_rgb"
    PRED_VS_GT_VERTS_NCDS_IN_RGB = "pred_vs_gt_verts_ncds_in_rgb"
    NET_FEATS_NEAREST_VERTS = "net_feats_nearest_verts"


# tasks should specify which input modalities they require to be calculated
# tasks should take as input which metrics with what scale to consider, then there is a final metric
class OD3D_Task(abc.ABC):
    subclasses = {}

    metrics_supported = []
    visuals_supported = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(
        self,
        metrics: dict[OD3D_Metrics:float],
        visuals: list[OD3D_Visuals],
        visuals_res: int = 128,
        **kwargs,
    ):
        self.metrics = {
            metric_key: metric_val
            for metric_key, metric_val in metrics.items()
            if metric_key in self.metrics_supported
        }
        self.visuals = [
            visual for visual in visuals if visual in self.visuals_supported
        ]
        self.visuals_res = visuals_res

    @abc.abstractmethod
    def eval(
        self,
        frames_pred,
        frames_gt=None,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def visualize(
        self,
        frames_pred,
        frames_gt=None,
    ):
        raise NotImplementedError

    @classmethod
    def create_by_name(cls, name: str):
        config = od3d.io.read_config_intern(
            rfpath=Path("methods/tasks").joinpath(f"{name}.yaml"),
        )
        return cls.subclasses[config.class_name].create_from_config(config)

    @classmethod
    def create_from_config(cls, config: DictConfig):
        keys = inspect.getfullargspec(cls.subclasses[config.class_name].__init__)[0][1:]
        return cls.subclasses[config.class_name](
            **{
                key: config.get(key)
                for key in keys
                if config.get(key, None) is not None
            },
        )


class OD3D_Tasks(OD3D_Task):
    def __init__(self, tasks: list[OD3D_Task]):
        super().__init__(metrics=[], visuals=[])
        self.tasks: list[OD3D_Task] = tasks

    def eval(
        self,
        frames_pred,
        frames_gt=None,
    ):
        for task in self.tasks:
            if task is not None:
                frames_pred = task.eval(frames_pred=frames_pred, frames_gt=frames_gt)
            else:
                logger.warning(f"Task is None. {self.tasks}")
        return frames_pred

    def visualize(
        self,
        frames_pred,
        frames_gt=None,
    ):
        for task in self.tasks:
            if task is not None:
                frames_pred = task.visualize(
                    frames_pred=frames_pred,
                    frames_gt=frames_gt,
                )
            else:
                logger.warning(f"Task is None. {self.tasks}")
        return frames_pred

    @classmethod
    def create_from_config(cls, config: DictConfig):
        tasks: list[OD3D_Task] = []
        for config_task in config.tasks:
            tasks.append(
                OD3D_Task.subclasses[config_task.class_name].create_from_config(
                    config=config_task,
                ),
            )
        return OD3D_Tasks(tasks)

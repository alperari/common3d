import logging

logger = logging.getLogger(__name__)

import torch
from od3d.cv.geometry.transform import tform4x4, inv_tform4x4
from od3d.cv.io import read_image, write_mask_image
import torchvision
from dataclasses import dataclass
from typing import List, Union
from enum import Enum
from od3d.datasets.object import (
    OD3D_Object,
    OD3D_CamTform4x4ObjTypeMixin,
    OD3D_MaskTypeMixin,
    OD3D_MeshTypeMixin,
    OD3D_CAM_TFORM_OBJ_TYPES,
    OD3D_FRAME_MASK_TYPES,
    OD3D_FrameModalitiesMixin,
    OD3D_TformObjMixin,
    OD3D_MeshFeatsTypeMixin,
    OD3D_FRAME_DEPTH_TYPES,
    OD3D_DepthTypeMixin,
    OD3D_FRAME_KPTS2D_ANNOT_TYPES,
    OD3D_FrameKpts2d3dTypeMixin,
    OD3D_SCALE_TYPES,
)

from od3d.datasets.frame_meta import OD3D_FrameMeta
from pathlib import Path
from od3d.cv.io import write_depth_image, read_depth_image
from od3d.cv.geometry.objects3d.meshes import Meshes


class OD3D_FRAME_MODALITIES(str, Enum):
    NAME = "name"
    CAM_INTR4X4 = "cam_intr4x4"
    CAM_INTR4X4S = "cam_intr4x4s"
    CAM_TFORM4X4_OBJ = "cam_tform4x4_obj"
    CAM_TFORM4X4_OBJS = "cam_tform4x4_objs"
    OBJ_TFORM4X4_OBJS = "obj_tform4x4_objs"
    CATEGORY = "category"
    CATEGORY_ID = "category_id"
    CATEGORIES = "categories"
    CATEGORIES_IDS = "categories_ids"
    PCL = "pcl"
    SIZE = "size"
    SIZES = "sizes"
    RGB = "rgb"
    RGBS = "rgbs"
    RGB_MASK = "rgb_mask"
    PXL_CAT_ID = "pxl_cat_id"
    MASK = "mask"
    MASK_DT = "mask_dt"
    MASK_INV_DT = "mask_inv_dt"
    MASKS = "masks"
    DEPTH = "depth"
    DEPTH_MASK = "depth_mask"
    MESH = "mesh"
    MESH_ID_IN_BATCH = "mesh_id_in_batch"
    MESHS = "meshs"
    KPTS2D_ANNOT = "kpts2d_annot"
    KPTS2D_ANNOTS = "kpts2d_annots"
    KPTS2D_ANNOTS_IDS = "kpts2d_annots_ids"
    KPTS2D_ANNOT_VSBL = "kpts2d_annot_vsbl"
    KPTS3D = "kpts3d"
    KPTS_NAMES = "kpts_names"
    BBOX = "bbox"
    BBOXS = "bboxs"
    SEQUENCE = "sequence"
    SEQUENCE_NAME_UNIQUE = "sequence_name_unique"
    FRAME = "frame"
    FRAME_NAME_UNIQUE = "frame_name_unique"
    RAYS_CENTER3D = "rays_center3d"
    FEATMAP = "featmap"
    FEATMAPS = "featmaps"
    FEAT = "feat"
    FEATS = "feats"


class OD3D_FRAME_MODALITIES_STACKABLE(str, Enum):
    CAM_INTR4X4 = "cam_intr4x4"
    CAM_INTR4X4S = "cam_intr4x4s"
    CAM_TFORM4X4_OBJ = "cam_tform4x4_obj"
    CAM_TFORM4X4_OBJS = "cam_tform4x4_objs"
    CATEGORY_ID = "category_id"
    SIZE = "size"
    SIZES = "sizes"
    RGB = "rgb"
    RGB_MASK = "rgb_mask"
    PXL_CAT_ID = "pxl_cat_id"
    MASK = "mask"
    MASK_DT = "mask_dt"
    MASK_INV_DT = "mask_inv_dt"
    MASKS = "masks"
    DEPTH = "depth"
    DEPTH_MASK = "depth_mask"
    BBOX = "bbox"
    BBOXS = "bboxs"
    RAYS_CENTER3D = "rays_center3d"
    FEATMAP = "featmap"
    FEAT = "feat"
    FEATS = "feats"


@dataclass
class OD3D_Frame(OD3D_FrameModalitiesMixin, OD3D_Object):
    meta_type = OD3D_FrameMeta

    def get_modality(self, modality: OD3D_FRAME_MODALITIES):
        if modality in self.modalities:
            if modality == OD3D_FRAME_MODALITIES.CAM_INTR4X4:
                return self.get_cam_intr4x4()
            elif modality == OD3D_FRAME_MODALITIES.CAM_INTR4X4S:
                return self.get_cam_intr4x4s()
            elif modality == OD3D_FRAME_MODALITIES.CAM_TFORM4X4_OBJ:
                return self.get_cam_tform4x4_obj()
            elif modality == OD3D_FRAME_MODALITIES.CAM_TFORM4X4_OBJS:
                return self.get_cam_tform4x4_objs()
            elif modality == OD3D_FRAME_MODALITIES.OBJ_TFORM4X4_OBJS:
                return self.get_obj_tform4x4_objs()
            elif modality == OD3D_FRAME_MODALITIES.CATEGORY:
                return self.category
            elif modality == OD3D_FRAME_MODALITIES.CATEGORY_ID:
                return self.category_id
            elif modality == OD3D_FRAME_MODALITIES.CATEGORIES:
                return self.categories
            elif modality == OD3D_FRAME_MODALITIES.CATEGORIES_IDS:
                return self.categories_ids
            elif modality == OD3D_FRAME_MODALITIES.PCL:
                pts3d, pts3d_colors, pts3d_normals = self.get_pcl(max_pts=2000)
                return pts3d
            elif modality == OD3D_FRAME_MODALITIES.SIZE:
                return self.size
            elif modality == OD3D_FRAME_MODALITIES.SIZES:
                return self.sizes
            elif modality == OD3D_FRAME_MODALITIES.RGB:
                return self.get_rgb()
            elif modality == OD3D_FRAME_MODALITIES.RGBS:
                return self.get_rgbs()
            elif modality == OD3D_FRAME_MODALITIES.FEAT:
                return self.get_feat()
            elif modality == OD3D_FRAME_MODALITIES.FEATS:
                return self.get_feats()
            elif modality == OD3D_FRAME_MODALITIES.FEATMAP:
                return self.get_featmap()
            elif modality == OD3D_FRAME_MODALITIES.FEATMAPS:
                return self.get_featmaps()
            elif modality == OD3D_FRAME_MODALITIES.RGB_MASK:
                return self.get_rgb_mask()
            elif modality == OD3D_FRAME_MODALITIES.PXL_CAT_ID:
                return self.get_pxl_cat_id()
            elif modality == OD3D_FRAME_MODALITIES.MASK:
                return self.get_mask()
            elif modality == OD3D_FRAME_MODALITIES.MASK_DT:
                return self.get_mask_dt()
            elif modality == OD3D_FRAME_MODALITIES.MASK_INV_DT:
                return self.get_mask_inv_dt()
            elif modality == OD3D_FRAME_MODALITIES.DEPTH:
                return self.get_depth()
            elif modality == OD3D_FRAME_MODALITIES.DEPTH_MASK:
                return self.get_depth_mask()
            elif modality == OD3D_FRAME_MODALITIES.MESH:
                return self.get_mesh()
            elif modality == OD3D_FRAME_MODALITIES.BBOX:
                return self.get_bbox()
            elif modality == OD3D_FRAME_MODALITIES.BBOXS:
                return self.get_bboxs()
            elif modality == OD3D_FRAME_MODALITIES.KPTS2D_ANNOT:
                return self.get_kpts2d_annot()
            elif modality == OD3D_FRAME_MODALITIES.KPTS2D_ANNOT_VSBL:
                return self.get_kpts2d_annot_vsbl()
            elif modality == OD3D_FRAME_MODALITIES.KPTS3D:
                return self.get_kpts3d()
            elif modality == OD3D_FRAME_MODALITIES.KPTS_NAMES:
                return self.kpts_names
            elif modality == OD3D_FRAME_MODALITIES.RAYS_CENTER3D:
                return self.rays_center3d
            elif modality == OD3D_FRAME_MODALITIES.SEQUENCE:
                return self.sequence
            elif modality == OD3D_FRAME_MODALITIES.SEQUENCE_NAME_UNIQUE:
                return self.sequence.name_unique
            elif modality == OD3D_FRAME_MODALITIES.FRAME:
                return self
            elif modality == OD3D_FRAME_MODALITIES.FRAME_NAME_UNIQUE:
                return self.name_unique
            elif modality == OD3D_FRAME_MODALITIES.KPTS2D_ANNOTS:
                return self.get_kpts2d_annots()
            elif modality == OD3D_FRAME_MODALITIES.KPTS2D_ANNOTS_IDS:
                return self.get_kpts2d_annot_ids()

        logger.warning(f"modality {modality} not supported")
        return None

    # @property
    # def meta(self):
    #    return OD3D_FrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=self.name_unique)

    # pass
    # def __init__(self, path_raw: Path, path_preprocess: Path, name_unique: Path, modalities: List[OD3D_FRAME_MODALITIES], categories: List[str]):
    #     self.path_raw: Path = path_raw
    #     self.path_preprocess: Path = path_preprocess
    #     self.all_categories = categories
    #     self.name_unique = name_unique
    # self.modalities = modalities
    # self.meta: OD3D_FrameMetaClasses = meta
    # self.path_meta: Path = path_meta
    # #self.category_id = categories.index(self.category)
    # self.item_id = None
    # self._rgb = None
    # self._depth = None
    # self._depth_mask = None
    # self._kpts2d_orient = None
    # self._mesh = None
    # self._kpts3d = None


@dataclass
class OD3D_FrameFeatMixin(OD3D_Object):
    feat = None  # F

    def get_feat(self):
        return self.feat


@dataclass
class OD3D_FrameFeatsMixin(OD3D_Object):
    feats = None  # NxF

    def get_feats(self):
        return self.feats


@dataclass
class OD3D_FrameFeatmapMixin(OD3D_Object):
    featmap = None  # FxHxW

    def get_featmap(self):
        return self.featmap


@dataclass
class OD3D_FrameFeatmapsMixin(OD3D_Object):
    featmaps = None  # List(FxHxW) or NxFxHxW

    def get_featmaps(self):
        return self.featmaps


@dataclass
class OD3D_FrameMaskMixin(OD3D_MaskTypeMixin):
    mask = None

    @property
    def fpath_mask(self):
        if self.mask_type == OD3D_FRAME_MASK_TYPES.META:
            return self.path_raw.joinpath(self.meta.rfpath_mask)
        elif self.mask_type == OD3D_FRAME_MASK_TYPES.MESH:
            return self.path_preprocess.joinpath(
                "mask",
                f"{self.mask_type}",
                self.mesh_type_unique,
                f"{self.name_unique}.png",
            )
        else:
            return self.path_preprocess.joinpath(
                "mask",
                f"{self.mask_type}",
                f"{self.name_unique}.png",
            )

    def write_mask(self, value: torch.Tensor):
        if self.fpath_mask.parent.exists() is False:
            self.fpath_mask.parent.mkdir(parents=True, exist_ok=True)
        if value.dtype == torch.bool:
            value_write = value.to(torch.uint8).detach().cpu() * 255
        elif value.dtype == torch.uint8:
            value_write = value.detach().cpu()
        else:
            value_write = (value * 255).to(torch.uint8).detach().cpu()
        torchvision.io.write_png(input=value_write, filename=str(self.fpath_mask))
        self.mask = value

    def read_mask(self):
        img = read_image(self.fpath_mask) / 255.0
        if img.shape[0] == 4:
            img = img[3:]
        return img

    def get_mask(self):
        if self.mask is None:
            self.mask = self.read_mask()
        return self.mask

    def get_mask_inv(self):
        return 1.0 - self.get_mask()

    def get_mask_bin(self):
        return self.get_mask() > 0.5

    def get_mask_inv_bin(self):
        return ~self.get_mask_bin()

    def get_mask_dt(self):
        import cv2
        import numpy as np

        mask_bin = self.get_mask_bin()
        mask_np = np.uint8(mask_bin.numpy()[0] * 255.0)
        mask_dt = torch.FloatTensor(
            cv2.distanceTransform(mask_np, cv2.DIST_L2, cv2.DIST_MASK_PRECISE),
        )[
            None,
        ]
        mask_size = max(mask_dt.shape[-1], mask_dt.shape[-2])
        mask_dt /= mask_size
        if (~mask_bin).sum() == 0:
            mask_dt[:] = 1.0

        # from od3d.cv.visual.show import show_img
        # show_img(mask_dt)
        # show_img(self.get_mask_bin())
        return mask_dt

    def get_mask_inv_dt(self):
        import cv2
        import numpy as np

        mask_inv_bin = self.get_mask_inv_bin()
        mask_inv_np = np.uint8(mask_inv_bin.numpy()[0] * 255.0)
        mask_inv_dt = torch.FloatTensor(
            cv2.distanceTransform(mask_inv_np, cv2.DIST_L2, cv2.DIST_MASK_PRECISE),
        )[
            None,
        ]
        mask_size = max(mask_inv_dt.shape[-1], mask_inv_dt.shape[-2])
        mask_inv_dt /= mask_size

        if (~mask_inv_bin).sum() == 0:
            mask_inv_dt[:] = 1.0

        # from od3d.cv.visual.show import show_img
        # show_img(mask_dt_inv)
        # show_img(self.get_mask_inv_bin())
        return mask_inv_dt


@dataclass
class OD3D_FrameScaleMixin(OD3D_Object):
    scale_type: OD3D_SCALE_TYPES

    @property
    def scale(self):
        if self.scale_type == OD3D_SCALE_TYPES.REAL:
            return self.get_real_scale()
        elif self.scale_type == OD3D_SCALE_TYPES.NORM:
            return self.get_norm_scale()
        else:
            raise NotImplementedError

    def get_real_scale(self):
        raise NotImplementedError

    def get_norm_scale(self):
        return 1.0


@dataclass
class OD3D_FrameSizeMixin(OD3D_Object):
    # H, W
    _size = None

    @property
    def size(self):
        # H, W
        if self._size is None:
            self._size = self.meta.size
            if (self._size < 1).any():
                self._size = torch.Tensor([1, 1])
                logger.warning(
                    f"OD3D Frame {self.name_unique} has a size smaller than 1 setting to 1 1",
                )
        return self._size

    @size.setter
    def size(self, value: torch.Tensor):
        self._size = value

    @property
    def H(self):
        return int(self.size[0].item())

    @property
    def W(self):
        return int(self.size[1].item())


@dataclass
class OD3D_FrameSizesMixin(OD3D_Object):
    _sizes = None

    @property
    def sizes(self):
        # H, W
        if self._sizes is None:
            self._sizes = self.meta.sizes
            if (self._sizes < 1).any():
                self._sizes = torch.Tensor([1, 1])
                logger.warning(
                    f"OD3D Frame {self.name_unique} has a size smaller than 1 setting to 1 1",
                )
        return self._sizes

    @sizes.setter
    def sizes(self, value: torch.Tensor):
        self._sizes = value

    @property
    def Hs(self):
        return self.size[0] if self.size.dim() == 1 else self.size[:, 0]
        # need to change W*H to H*W in preprocessing

    @property
    def Ws(self):
        return self.size[1] if self.size.dim() == 1 else self.size[:, 1]


@dataclass
class OD3D_FrameRGBMaskMixin(OD3D_FrameSizeMixin):
    rgb_mask = None

    def get_rgb_mask(self):
        if self.rgb_mask is None:
            self.rgb_mask = torch.ones(size=(1, self.H, self.W), dtype=torch.bool)
        return self.rgb_mask


@dataclass
class OD3D_FrameCamTform4x4ObjMixin(OD3D_CamTform4x4ObjTypeMixin):
    cam_tform4x4_obj = None

    def read_cam_tform4x4_obj(self, cam_tform4x4_obj_type=None):
        if cam_tform4x4_obj_type is None:
            cam_tform4x4_obj_type = self.cam_tform4x4_obj_type

        if cam_tform4x4_obj_type == OD3D_CAM_TFORM_OBJ_TYPES.META:
            cam_tform4x4_obj = self.meta.cam_tform4x4_obj
        elif cam_tform4x4_obj_type == OD3D_CAM_TFORM_OBJ_TYPES.SFM:
            cam_tform4x4_obj = self.sequence.get_sfm_cam_tform4x4_obj(
                f"{Path(self.name_unique).stem}",
            )
        else:
            raise ValueError(
                f"cam_tform4x4_obj_type {self.cam_tform4x4_obj_type} not supported",
            )

        if cam_tform4x4_obj_type == self.cam_tform4x4_obj_type:
            self.cam_tform4x4_obj = cam_tform4x4_obj
        return cam_tform4x4_obj

    def get_cam_tform4x4_obj(self, cam_tform4x4_obj_type=None):
        if self.cam_tform4x4_obj is not None and (
            cam_tform4x4_obj_type is None
            or self.cam_tform4x4_obj_type == cam_tform4x4_obj_type
        ):
            cam_tform4x4_obj = self.cam_tform4x4_obj
        else:
            cam_tform4x4_obj = self.read_cam_tform4x4_obj(
                cam_tform4x4_obj_type=cam_tform4x4_obj_type,
            )
        return cam_tform4x4_obj


@dataclass
class OD3D_FramePCLMixin:
    pcl: torch.Tensor = None

    @property
    def fpath_pcl(self):
        return self.get_fpath_pcl()

    def read_pcl(self, pcl_type=None, max_pts=None):
        return self.sequence.read_pcl(pcl_type, max_pts=max_pts)

    def get_pcl(self, max_pts=None):
        return self.sequence.get_pcl(max_pts=max_pts)


@dataclass
class OD3D_FrameMeshMixin(OD3D_MeshFeatsTypeMixin, OD3D_MeshTypeMixin):
    mesh: Meshes = None

    @property
    def fpath_mesh(self):
        return self.get_fpath_mesh()

    def read_mesh(self, mesh_type=None):
        return self.sequence.read_mesh()

    def get_fpath_mesh(self, mesh_type=None):
        return self.sequence.get_fpath_mesh(mesh_type=mesh_type)

    def get_mesh(self, mesh_type=None, clone=False, device="cpu", tform_obj_type=None):
        return self.sequence.get_mesh(
            mesh_type=mesh_type,
            clone=clone,
            device=device,
            tform_obj_type=tform_obj_type,
        )


@dataclass
class OD3D_FrameCamIntr4x4Mixin(OD3D_Frame):
    cam_intr4x4 = None

    def get_cam_intr4x4(self):
        if self.cam_intr4x4 is None:
            self.cam_intr4x4 = self.read_cam_intr4x4()
        return self.cam_intr4x4

    def read_cam_intr4x4(self):
        if self.cam_intr4x4 is None:
            self.cam_intr4x4 = self.meta.cam_intr4x4.clone()
        return self.cam_intr4x4


@dataclass
class OD3D_CamProj4x4ObjMixin(OD3D_FrameCamTform4x4ObjMixin, OD3D_FrameCamIntr4x4Mixin):
    @property
    def cam_proj4x4_obj(self):
        return tform4x4(self.get_cam_intr4x4(), self.get_cam_tform4x4_obj())


@dataclass
class OD3D_FrameCategoryMixin(OD3D_Object):
    all_categories: List[str]
    map_categories_to_od3d = None

    @property
    def category(self):
        return self.meta.category

    @property
    def category_id(self):
        return self.all_categories.index(self.category)


@dataclass
class OD3D_FrameCategoriesMixin(OD3D_Object):
    all_categories: List[str]

    @property
    def categories(self):
        return self.meta.categories

    @property
    def category(self):
        return self.categories[0]

    @property
    def categories_ids(self):
        return torch.LongTensor(
            [self.all_categories.index(cat) for cat in self.categories],
        )

    @property
    def category_id(self):
        return self.categories_ids[0]


@dataclass
class OD3D_FrameBBoxMixin(OD3D_Object):
    # # x0, y0, x1, y1
    bbox = None

    def read_bbox(self):
        return self.meta.bbox.clone()

    def get_bbox(self):
        if self.bbox is None:
            self.bbox = self.read_bbox()
        return self.bbox


@dataclass
class OD3D_FrameBBoxsMixin(OD3D_Object):
    bboxs = None

    def read_bboxs(self):
        return self.meta.bboxs.clone()

    def get_bboxs(self):
        if self.bboxs is None:
            self.bboxs = self.read_bboxs()
        return self.bboxs


@dataclass
class OD3D_FrameBBoxsVsblMixin(OD3D_Object):
    bboxs_vsbl = None

    def read_bboxs_vsbl(self):
        return self.meta.bboxs_vsbl.clone()

    def get_bboxs_vsbl(self):
        if self.bboxs_vsbl is None:
            self.bboxs_vsbl = self.read_bboxs_vsbl()
        return self.bboxs_vsbl


@dataclass
class OD3D_FrameKpts2d3dMixin(OD3D_FrameKpts2d3dTypeMixin):
    kpts3d = None
    kpts2d_annot = None
    kpts2d_annot_vsbl = None

    @property
    def kpts_names(self):
        return self.meta.kpts_names

    @property
    def fpath_kpts2d_annot(self):
        if self.kpts2d_annot_type == OD3D_FRAME_KPTS2D_ANNOT_TYPES.META:
            raise ValueError("Meta kpts2d_annot is not saved in a file")
        else:
            return self.path_preprocess.joinpath(
                "kpts2d_annot",
                self.kpts2d_annot_type,
                f"{self.name_unique}.pt",
            )

    def read_kpts2d_annot(self):
        if self.kpts2d_annot_type == OD3D_FRAME_KPTS2D_ANNOT_TYPES.META:
            return self.meta.kpts2d_annot.clone()
        else:
            return torch.load(self.fpath_kpts2d_annot)

    def get_kpts2d_annot(self):
        if self.kpts2d_annot is None:
            self.kpts2d_annot = self.read_kpts2d_annot()
        return self.kpts2d_annot

    @property
    def kpts2d_annot_labeled(self):
        if self.kpts2d_annot_type == OD3D_FRAME_KPTS2D_ANNOT_TYPES.META:
            return True
        else:
            return

    def read_kpts2d_annot_vsbl(self):
        return self.meta.kpts2d_annot_vsbl.clone()

    def get_kpts2d_annot_vsbl(self):
        if self.kpts2d_annot_vsbl is None:
            self.kpts2d_annot_vsbl = self.read_kpts2d_annot_vsbl()
        return self.kpts2d_annot_vsbl

    def read_kpts3d(self):
        return self.meta.kpts3d.clone()

    def get_kpts3d(self):
        if self.kpts3d is None:
            self.kpts3d = self.read_kpts3d()
        return self.kpts3d

    # @property
    # def kpts3d(self):
    #     if self._kpts3d is None:
    #         self._kpts3d = self.meta.kpts3d
    #     return self._kpts3d
    #
    # @kpts3d.setter
    # def kpts3d(self, value: torch.Tensor):
    #         self._kpts3d = value

    # @property
    # def fpath_kpts2d_orient(self):
    #     return self.path_preprocess.joinpath("labels", "kpts2d_orient", f"{self.name_unique}.pt")
    #
    # @property
    # def kpts2d_orient_labeled(self):
    #     return self.fpath_kpts2d_orient.exists()
    # @property
    # def kpts2d_orient(self):
    #     kpts2d_orient = torch.load(self.fpath_kpts2d_orient)
    #     return kpts2d_orient


# class OD3D_Kpts3dMixin(OD3D_Object):


@dataclass
class OD3D_FrameKpts2dIDMixin(OD3D_Object):
    kpts2d_annot_type: OD3D_FRAME_KPTS2D_ANNOT_TYPES
    kpts2d_annots_ids = None
    kpts2d_annots = None

    def read_kpts2d_annots(self):
        if self.kpts2d_annot_type == OD3D_FRAME_KPTS2D_ANNOT_TYPES.META:
            return self.meta.kpts2d_annots.clone()
        else:
            return torch.load(self.fpath_kpts2d_annot)

    def get_kpts2d_annots(self):
        if self.kpts2d_annots is None:
            self.kpts2d_annots = self.read_kpts2d_annots()
        return self.kpts2d_annots

    def read_kpts2d_annots_ids(self):
        return self.meta.kpts2d_annots_ids.clone()

    def get_kpts2d_annot_ids(self):
        if self.kpts2d_annots_ids is None:
            self.kpts2d_annots_ids = self.read_kpts2d_annots_ids()
        return self.kpts2d_annots_ids


@dataclass
class OD3D_FrameBBoxFromKpts2d3dMixin(OD3D_FrameKpts2d3dMixin, OD3D_FrameSizeMixin):
    # # x0, y0, x1, y1
    bbox = None

    def read_bbox(self):
        kpts2d = self.read_kpts2d_annot()
        H, W = self.size  # problem not original size
        x0 = kpts2d[:, 0].min().item()  # .clamp(0, W-1).item()
        x1 = kpts2d[:, 0].max().item()  # .clamp(0, W-1).item()
        y0 = kpts2d[:, 1].min().item()  # .clamp(0, H-1).item()
        y1 = kpts2d[:, 1].max().item()  # .clamp(0, H-1).item()
        return torch.Tensor([x0, y0, x1, y1]).to(device=kpts2d.device)

    def get_bbox(self):
        if self.bbox is None:
            self.bbox = self.read_bbox()
        return self.bbox


class OD3D_FrameRGBMixin(OD3D_Object):
    rgb = None

    @property
    def fpath_rgb(self):
        return self.path_raw.joinpath(self.meta.rfpath_rgb)

    def read_rgb(self):
        rgb = torchvision.io.read_image(
            str(self.fpath_rgb),
            mode=torchvision.io.ImageReadMode.RGB,
        )
        return rgb

    def get_rgb(self):
        if self.rgb is None:
            self.rgb = self.read_rgb()
        return self.rgb


class OD3D_FrameRGBSMixin(OD3D_Object):
    rgbs = None

    @property
    def fpath_rgbs(self):
        return [self.path_raw.joinpath(path) for path in self.meta.rfpath_rgb]

    def read_rgbs(self):
        rgbs = [
            torchvision.io.read_image(
                str(path),
                mode=torchvision.io.ImageReadMode.RGB,
            )
            for path in self.fpath_rgbs
        ]
        return rgbs

    def get_rgbs(self):
        if self.rgbs is None:
            self.rgbs = self.read_rgbs()
        return self.rgbs


class OD3D_FrameDepthMixin(OD3D_DepthTypeMixin):
    depth = None

    @property
    def fpath_depth(self):
        if self.depth_type == OD3D_FRAME_DEPTH_TYPES.META:
            return self.path_raw.joinpath(self.meta.rfpath_depth)
        elif self.depth_type == OD3D_FRAME_DEPTH_TYPES.MESH:
            return self.path_preprocess.joinpath(
                "depth",
                f"{self.depth_type}",
                self.mesh_type_unique,
                f"{self.name_unique}.png",
            )
        else:
            return self.path_preprocess.joinpath(
                "depth",
                f"{self.depth_type}",
                f"{self.name_unique}.png",
            )

    def write_depth(self, value: torch.Tensor):
        write_depth_image(value, path=self.fpath_depth)
        self.depth = value

    def read_depth(self):
        if self.depth_type == OD3D_FRAME_DEPTH_TYPES.META:
            depth = torchvision.io.read_image(
                str(self.fpath_depth),
                mode=torchvision.io.ImageReadMode.UNCHANGED,
            )
        elif self.depth_type == OD3D_FRAME_DEPTH_TYPES.MESH:
            depth = read_depth_image(self.fpath_depth)
        else:
            raise NotImplementedError
        return depth

    def get_depth(self):
        if self.depth is None:
            self.depth = self.read_depth()
        return self.depth


class OD3D_FrameDepthMaskMixin(OD3D_DepthTypeMixin):
    depth_mask = None

    @property
    def fpath_depth_mask(self):
        if self.depth_type == OD3D_FRAME_DEPTH_TYPES.META:
            return self.path_raw.joinpath(self.meta.rfpath_depth_mask)
        elif self.depth_type == OD3D_FRAME_DEPTH_TYPES.MESH:
            return self.path_preprocess.joinpath(
                "depth_mask",
                f"{self.depth_type}",
                self.mesh_type_unique,
                f"{self.name_unique}.png",
            )
        else:
            return self.path_preprocess.joinpath(
                "depth_mask",
                f"{self.depth_type}",
                f"{self.name_unique}.png",
            )

    def read_depth_mask(self):
        depth_mask = read_image(self.fpath_depth_mask)
        return depth_mask

    def get_depth_mask(self):
        if self.depth_mask is None:
            self.depth_mask = self.read_depth_mask()
        return self.depth_mask

    def write_depth_mask(self, value: torch.Tensor):
        write_mask_image(value, path=self.fpath_depth_mask)
        self.depth_mask = value


@dataclass
class OD3D_FrameSequenceMixin(OD3D_Object):
    sequence_type = None  #  OD3D_Sequence

    @property
    def sequence_name(self):
        return self.meta.sequence_name

    @property
    def sequence_name_unique(self):
        return self.meta.sequence_name_unique

    @property
    def sequence(self):
        from dataclasses import fields

        frame_fields = fields(self)
        sequence_fields_names = [field.name for field in fields(self.sequence_type)]
        all_attrs_except_name_unique = {
            field.name: getattr(self, field.name)
            for field in frame_fields
            if field.name != "name_unique" and field.name in sequence_fields_names
        }
        return self.sequence_type(
            name_unique=self.sequence_name_unique,
            **all_attrs_except_name_unique,
        )


@dataclass
class OD3D_FrameRaysCenter3dMixin(OD3D_FrameSequenceMixin):
    _rays_center3d = None

    @property
    def rays_center3d(self):
        if self._rays_center3d is None:
            self._rays_center3d = self.sequence.get_sfm_rays_center3d()
        return self._rays_center3d


@dataclass
class OD3D_FrameSubsetMixin(OD3D_Object):
    @property
    def subset(self):
        return self.meta.subset


from od3d.datasets.frame_meta import OD3D_FrameMeta


@dataclass
class OD3D_FrameCamIntr4x4Mixin(OD3D_Frame):
    cam_intr4x4 = None

    def read_cam_intr(self):
        return self.meta.cam_intr4x4.clone()

    def get_cam_intr(self):
        if self.cam_intr4x4 is None:
            self.cam_intr4x4 = self.read_cam_intr()
        return self.cam_intr4x4


@dataclass
class OD3D_FrameTformObjMixin(
    OD3D_TformObjMixin,
    OD3D_FrameCamTform4x4ObjMixin,
    OD3D_Frame,
):
    def read_cam_tform4x4_obj(self, cam_tform4x4_obj_type=None, tform_obj_type=None):
        cam_tform4x4_obj = super().read_cam_tform4x4_obj(
            cam_tform4x4_obj_type=cam_tform4x4_obj_type,
        )

        tform_obj = self.sequence.get_tform_obj(tform_obj_type=tform_obj_type)
        if tform_obj is not None:
            cam_tform4x4_obj = tform4x4(cam_tform4x4_obj, inv_tform4x4(tform_obj))

        # note: note alignment of droid slam may include scale, therefore remove this scale.
        # note: projection does not change as we scale the depth z to the object as well
        scale = (
            cam_tform4x4_obj[:3, :3]
            .norm(dim=-1, keepdim=True)
            .mean(dim=-2, keepdim=True)
        )
        cam_tform4x4_obj[:3] = cam_tform4x4_obj[:3] / scale

        if (
            cam_tform4x4_obj_type is None
            or cam_tform4x4_obj_type == self.cam_tform4x4_obj_type
        ) and (tform_obj_type == self.tform_obj_type or tform_obj_type is None):
            self.cam_tform4x4_obj = cam_tform4x4_obj
        return cam_tform4x4_obj

    def get_cam_tform4x4_obj(self, cam_tform4x4_obj_type=None, tform_obj_type=None):
        if (
            self.cam_tform4x4_obj is not None
            and (
                cam_tform4x4_obj_type is None
                or self.cam_tform4x4_obj_type == cam_tform4x4_obj_type
            )
            and (tform_obj_type == self.tform_obj_type or tform_obj_type is None)
        ):
            cam_tform4x4_obj = self.cam_tform4x4_obj
        else:
            cam_tform4x4_obj = self.read_cam_tform4x4_obj(
                cam_tform4x4_obj_type=cam_tform4x4_obj_type,
                tform_obj_type=tform_obj_type,
            )
        return cam_tform4x4_obj


@dataclass
class OD3D_FrameCamIntr4x4sMixin(OD3D_Frame):
    cam_intr4x4s = None

    def get_cam_intr4x4s(self):
        if self.cam_intr4x4s is None:
            self.cam_intr4x4s = self.read_cam_intr4x4s()
        return self.cam_intr4x4s

    def read_cam_intr4x4s(self):
        if self.cam_intr4x4s is None:
            self.cam_intr4x4s = self.meta.cam_intr4x4s.clone()
        return self.cam_intr4x4s


@dataclass
class OD3D_FrameObjTform4x4ObjsMixin(OD3D_Object):
    obj_tform4x4_objs = None

    def read_obj_tform4x4_objs(self):
        return self.meta.obj_tform4x4_objs.clone()

    def get_obj_tform4x4_objs(self):
        if self.obj_tform4x4_objs is None:
            self.obj_tform4x4_objs = self.read_obj_tform4x4_objs()
        return self.obj_tform4x4_objs


@dataclass
class OD3D_FrameCamTform4x4ObjsMixin(
    OD3D_FrameObjTform4x4ObjsMixin,
    OD3D_FrameCamTform4x4ObjMixin,
):
    cam_tform4x4_objs = None
    # obj_tform_objs = None

    def read_cam_tform4x4_objs(self, cam_tform4x4_obj_type=None):
        if cam_tform4x4_obj_type is None:
            cam_tform4x4_obj_type = self.cam_tform4x4_obj_type

        if cam_tform4x4_obj_type == OD3D_CAM_TFORM_OBJ_TYPES.META:
            cam_tform4x4_objs = tform4x4(
                self.read_cam_tform4x4_obj()[:, None],
                self.read_obj_tform4x4_objs(),
            )  # self.meta.cam_tform4x4_objs
        else:
            raise ValueError(
                f"cam_tform4x4_obj_type {self.cam_tform4x4_obj_type} not supported",
            )

        if cam_tform4x4_obj_type == self.cam_tform4x4_obj_type:
            self.cam_tform4x4_obj = cam_tform4x4_objs
        return cam_tform4x4_objs

    def get_cam_tform4x4_objs(self, cam_tform4x4_obj_type=None):
        if self.cam_tform4x4_objs is not None and (
            cam_tform4x4_obj_type is None
            or self.cam_tform4x4_obj_type == cam_tform4x4_obj_type
        ):
            cam_tform4x4_objs = tform4x4(
                self.get_cam_tform4x4_obj()[:, None],
                self.get_obj_tform4x4_objs(),
            )  # self.cam_tform4x4_objs
        else:
            cam_tform4x4_objs = self.read_cam_tform4x4_objs(
                cam_tform4x4_obj_type=cam_tform4x4_obj_type,
            )
        return cam_tform4x4_objs


@dataclass
class OD3D_CamProj4x4ObjsMixin(
    OD3D_FrameCamTform4x4ObjsMixin,
    OD3D_FrameCamIntr4x4Mixin,
):
    @property
    def cam_proj4x4_obj(self):
        return tform4x4(self.get_cam_intr(), self.get_cam_tform4x4_obj())

    @property
    def cam_proj4x4_objs(self):
        return tform4x4(self.get_cam_intr(), self.get_cam_tform4x4_objs())


@dataclass
class OD3D_FrameTformObjsMixin(
    OD3D_TformObjMixin,
    OD3D_FrameCamTform4x4ObjsMixin,
    OD3D_Frame,
):
    def read_cam_tform4x4_objs(self, cam_tform4x4_obj_type=None, tform_obj_type=None):
        cam_tform4x4_objs = super().read_cam_tform4x4_objs(
            cam_tform4x4_obj_type=cam_tform4x4_obj_type,
        )

        tform_obj = self.sequence.get_tform_obj(tform_obj_type=tform_obj_type)
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

    def get_cam_tform4x4_objs(self, cam_tform4x4_obj_type=None, tform_obj_type=None):
        if (
            self.cam_tform4x4_objs is not None
            and (
                cam_tform4x4_obj_type is None
                or self.cam_tform4x4_obj_type == cam_tform4x4_obj_type
            )
            and (tform_obj_type == self.tform_obj_type or tform_obj_type is None)
        ):
            cam_tform4x4_objs = self.cam_tform4x4_objs
        else:
            cam_tform4x4_objs = self.read_cam_tform4x4_objs(
                cam_tform4x4_obj_type=cam_tform4x4_obj_type,
                tform_obj_type=tform_obj_type,
            )
        return cam_tform4x4_objs


@dataclass
class OD3D_CamProj4x4ObjsMixin(
    OD3D_FrameCamTform4x4ObjsMixin,
    OD3D_FrameCamIntr4x4sMixin,
):
    @property
    def cam_proj4x4_objs(self):
        return tform4x4(self.get_cam_intr4x4s(), self.get_cam_tform4x4_objs())


OD3D_FrameClasses = Union[
    OD3D_Object,
    OD3D_FrameCategoryMixin,
    OD3D_FrameCategoriesMixin,
    OD3D_FrameCamTform4x4ObjMixin,
    OD3D_CamProj4x4ObjMixin,
    OD3D_FrameCamIntr4x4Mixin,
    OD3D_FrameMaskMixin,
    OD3D_FrameDepthMixin,
    OD3D_FrameDepthMaskMixin,
    OD3D_FrameSizeMixin,
    OD3D_FrameRGBMixin,
    OD3D_FrameMeshMixin,
    OD3D_FrameSequenceMixin,
    OD3D_FrameSubsetMixin,
    OD3D_FrameBBoxMixin,
    OD3D_FrameKpts2d3dMixin,
    OD3D_FrameRGBMaskMixin,
]

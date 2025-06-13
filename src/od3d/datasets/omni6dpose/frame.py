import logging

from od3d.cv.io import read_depth_image
from od3d.cv.io import read_image
from od3d.datasets.object import OD3D_MESH_TYPES
from od3d.datasets.object import OD3D_TFROM_OBJ_TYPES
from od3d.datasets.omni6dpose.enum import MAP_CATEGORIES_OMNI6DPOSE_TO_OD3D

logger = logging.getLogger(__name__)
from od3d.cv.geometry.objects3d.meshes import Meshes
import torch
from pathlib import Path
from od3d.cv.geometry.transform import transf3d_broadcast, inv_tform4x4, tform4x4

from od3d.datasets.frame_meta import (
    OD3D_FrameMeta,
    OD3D_FrameMetaRGBMixin,
    OD3D_FrameMetaDepthMixin,
    OD3D_FrameMetaCategoriesMixin,
    OD3D_FrameMetaPxlCatIdMixin,
    OD3D_FrameMetaMaskMixin,
    OD3D_FrameMetaSizeMixin,
    OD3D_FrameMetaSequenceMixin,
    OD3D_FrameMetaDepthMaskMixin,
    OD3D_FrameMetaCamIntr4x4Mixin,
    OD3D_FrameMetaCamTform4x4ObjMixin,
    OD3D_FrameMetaObjTform4x4ObjsMixin,
    OD3D_FrameMetaObjsValidMixin,
    OD3D_FrameMetaObjsNameMixin,
    OD3D_FrameMetaSubsetMixin,
    OD3D_FrameMetaMeshsMixin,
)

from od3d.datasets.frame import (
    OD3D_Frame,
    OD3D_FrameBBoxsMixin,
    OD3D_FrameSizeMixin,
    OD3D_FrameRGBMixin,
    OD3D_FrameMaskMixin,
    OD3D_FrameRGBMaskMixin,
    OD3D_FrameRaysCenter3dMixin,
    OD3D_FrameMeshMixin,
    OD3D_FramePCLMixin,
    OD3D_FrameTformObjMixin,
    OD3D_FrameSequenceMixin,
    OD3D_FrameCategoriesMixin,
    OD3D_CamProj4x4ObjMixin,
    OD3D_FrameCamTform4x4ObjsMixin,
    OD3D_FrameDepthMixin,
    OD3D_FrameDepthMaskMixin,
)
from dataclasses import dataclass
from od3d.datasets.object import (
    OD3D_PCLTypeMixin,
    OD3D_MeshTypeMixin,
    OD3D_SequenceSfMTypeMixin,
)


@dataclass
class Omni6DPose_FrameMeta(
    OD3D_FrameMetaCamIntr4x4Mixin,
    OD3D_FrameMetaCamTform4x4ObjMixin,
    OD3D_FrameMetaObjTform4x4ObjsMixin,
    OD3D_FrameMetaObjsValidMixin,
    OD3D_FrameMetaObjsNameMixin,
    OD3D_FrameMetaMeshsMixin,
    OD3D_FrameMetaDepthMaskMixin,
    OD3D_FrameMetaDepthMixin,
    OD3D_FrameMetaPxlCatIdMixin,
    OD3D_FrameMetaRGBMixin,
    OD3D_FrameMetaSizeMixin,
    OD3D_FrameMetaCategoriesMixin,
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

    pass


@dataclass
class Omni6DPose_Frame(
    OD3D_FramePCLMixin,
    OD3D_FrameBBoxsMixin,
    OD3D_FrameMeshMixin,
    OD3D_FrameRaysCenter3dMixin,
    OD3D_FrameTformObjMixin,
    OD3D_FrameCamTform4x4ObjsMixin,
    OD3D_CamProj4x4ObjMixin,
    OD3D_FrameRGBMaskMixin,
    OD3D_FrameMaskMixin,
    OD3D_FrameRGBMixin,
    OD3D_FrameDepthMixin,
    OD3D_FrameDepthMaskMixin,
    OD3D_FrameCategoriesMixin,
    OD3D_FrameSequenceMixin,
    OD3D_FrameSizeMixin,
    OD3D_PCLTypeMixin,
    OD3D_SequenceSfMTypeMixin,
    OD3D_Frame,
):
    meta_type = Omni6DPose_FrameMeta
    map_categories_to_od3d = MAP_CATEGORIES_OMNI6DPOSE_TO_OD3D
    pxl_cat_id = None

    def __post_init__(self):
        # hack: prevents circular import
        from od3d.datasets.omni6dpose.sequence import Omni6DPose_Sequence

        self.sequence_type = Omni6DPose_Sequence

    @property
    def fpath_bboxs(self):
        return Path(
            self.path_preprocess.joinpath("bboxs", self.name_unique, "bboxs.pt"),
        )

    @property
    def fpath_bboxs_vsbl(self):
        return Path(
            self.path_preprocess.joinpath("bboxs", self.name_unique, "bboxs_vsbl.pt"),
        )

    def write_bboxs(self, bboxs: torch.LongTensor):
        self.fpath_bboxs.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bboxs.detach().cpu(), self.fpath_bboxs)

    def write_bboxs_vsbl(self, bboxs: torch.LongTensor):
        self.fpath_bboxs_vsbl.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bboxs.detach().cpu(), self.fpath_bboxs_vsbl)

    def preprocess_bboxs(self):
        self.mesh = self.read_mesh()
        self.mesh.cuda()
        #
        from od3d.cv.geometry.objects3d.meshes import PROJECT_MODALITIES

        scene = self.mesh.render(
            cams_tform4x4_obj=self.meta.cam_tform4x4_obj.clone()[None,].cuda(),
            cams_intr4x4=self.meta.cam_intr4x4.clone()[None,].cuda(),
            imgs_sizes=self.meta.size.clone(),
            # objects_ids=objs_ids,
            modalities=PROJECT_MODALITIES.MASK,
            broadcast_batch_and_cams=True,
            obj_tform4x4_objs=self.meta.obj_tform4x4_objs.clone().cuda(),
        )
        # scene: [16, 1, 1, 512, 512]) get bounding box per object in 16
        # from od3d.cv.visual.show import show_img, show_imgs
        # show_imgs(scene[:, 0])
        from torchvision.ops import masks_to_boxes

        bboxs = masks_to_boxes(scene[:, 0, 0] > 0).long()
        self.write_bboxs(bboxs=bboxs)

    def preprocess_bboxs_vsbl(self):
        self.mesh = self.read_mesh()
        self.mesh.cuda()
        from od3d.cv.geometry.objects3d.meshes import PROJECT_MODALITIES

        scene = self.mesh.render(
            cams_tform4x4_obj=self.meta.cam_tform4x4_obj.clone()[None,].cuda(),
            cams_intr4x4=self.meta.cam_intr4x4.clone()[None,].cuda(),
            imgs_sizes=self.meta.size.clone(),
            objects_ids=torch.arange(len(self.mesh))[None,],
            modalities=PROJECT_MODALITIES.OBJ_IN_SCENE_ONEHOT,
            broadcast_batch_and_cams=False,
            obj_tform4x4_objs=self.meta.obj_tform4x4_objs.clone()[None,].cuda(),
        )
        # scene: [1, 16, 512, 512]) get bounding box per object in 16
        # from od3d.cv.visual.show import show_img, show_imgs
        # show_imgs(scene[0])
        from torchvision.ops import masks_to_boxes

        bboxs = masks_to_boxes(scene[0] > 0).long()
        self.write_bboxs_vsbl(bboxs=bboxs)

    def read_bboxs(self):
        if not self.fpath_bboxs.exists():
            self.preprocess_bboxs()
        return torch.load(self.fpath_bboxs).cpu()

    def read_bboxs_vsbl(self):
        if not self.fpath_bboxs_vsbl.exists():
            self.preprocess_bboxs_vsbl()
        return torch.load(self.fpath_bboxs_vsbl).cpu()

    @property
    def objcentric(self):
        return (
            self.meta.subset != "test_real"
            and self.meta.subset != "test_ikea"
            and self.meta.subset != "test_matterport3d"
            and self.meta.subset != "test_scannetpp"
            and self.meta.subset != "train_ikea"
            and self.meta.subset != "train_matterport3d"
            and self.meta.subset != "train_scannetpp"
        )

    @property
    def fpath_pxl_cat_id(self):
        if not self.objcentric:
            return self.path_raw.joinpath(self.meta.rfpath_pxl_cat_id)
        else:
            return self.path_preprocess.joinpath(self.meta.rfpath_pxl_cat_id)

    @property
    def fpath_rgb(self):
        if not self.objcentric:
            return self.path_raw.joinpath(self.meta.rfpath_rgb)
        else:
            return self.path_preprocess.joinpath(self.meta.rfpath_rgb)

    @property
    def fpath_depth(self):
        if not self.objcentric:
            return self.path_raw.joinpath(self.meta.rfpath_depth)
        else:
            return self.path_preprocess.joinpath(self.meta.rfpath_depth)

    @property
    def fpath_depth_mask(self):
        if not self.objcentric:
            return self.path_raw.joinpath(self.meta.rfpath_depth_mask)
        else:
            return self.path_preprocess.joinpath(self.meta.rfpath_depth_mask)

    def get_pxl_cat_id(self):
        if self.pxl_cat_id is None:
            self.pxl_cat_id = self.read_pxl_cat_id()
        return self.pxl_cat_id

    def read_pxl_cat_id(self):
        if not self.objcentric:
            """Load the mask image.

            :return: uint8 array of shape (Height, Width), whose values are related
                to the objects' mask ids (:attr:`.image_meta.ObjectPoseInfo.mask_id`).
            """
            import cv2
            import numpy as np
            import torch
            import os

            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
            img = cv2.imread(
                str(self.fpath_pxl_cat_id),
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
            )
            if len(img.shape) == 3:
                img = img[:, :, 2]
            img = np.array(img * 255, dtype=np.uint8)
            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"

            pxl_cat_id = torch.from_numpy(img)[None,]
            pxl_cat_id[pxl_cat_id == 255] = 0  # bug in test_real subset
        else:
            pxl_cat_id = read_image(path=self.fpath_pxl_cat_id)
        return pxl_cat_id  # 1xHxW

    def read_mask(self):
        return self.read_pxl_cat_id() > 0  # 1xHxW

    def read_depth(self):
        if not self.objcentric:
            """
            This function read the depth image, selecting the first channel if multiple
            channels are detected.

            :return: A 2D float array of shape (Height, Width). For Omni6DPose,
                the unit of pixel value is meter.
            """
            import cv2
            import torch
            import os

            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
            depth = cv2.imread(
                str(self.fpath_depth),
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
            )
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]

            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
            depth = torch.from_numpy(depth)
        else:
            depth = read_depth_image(path=self.fpath_depth)

        return depth  # unit: m

    def read_depth_mask(self):
        return (self.read_depth() > 0)[None,]  # 1xHxW

    def get_fpaths_meshs(self, mesh_type=None):
        # TODO: add for multiple mesh types
        if mesh_type is None:
            mesh_type = OD3D_MESH_TYPES.META

        if mesh_type == OD3D_MESH_TYPES.META:
            fpaths_meshs = [
                self.get_fpath_mesh_with_rfpath(rfpath)
                for rfpath in self.meta.rfpaths_meshs
            ]
        else:
            # logger.info(self.path_preprocess)
            # logger.info(mesh_type)
            # logger.info(self.meta.rfpaths_meshs)
            # logger.info([ rfpath.with_suffix('.ply') for rfpath in self.meta.rfpaths_meshs])
            fpaths_meshs = [
                self.path_preprocess.joinpath(
                    "mesh",
                    str(mesh_type),
                    rfpath.with_suffix(".ply"),
                )
                for rfpath in self.meta.rfpaths_meshs
            ]
        return fpaths_meshs

    def read_obj_tform4x4_objs(self):
        obj_tform4x4_objs = super(
            OD3D_FrameCamTform4x4ObjsMixin,
            self,
        ).read_obj_tform4x4_objs()
        fpaths_meshes = self.get_fpaths_meshs(mesh_type=self.mesh_type)
        mesh = Meshes.read_from_ply_files(fpaths_meshes=fpaths_meshes)
        objscentric_tform4x4_objs = mesh.get_objscentric_tform4x4_objs()
        objs_tform4x4_objscentric = inv_tform4x4(objscentric_tform4x4_objs)
        obj_tform4x4_objs = tform4x4(obj_tform4x4_objs, objs_tform4x4_objscentric)
        return obj_tform4x4_objs

    def get_obj_tform4x4_objs(self):
        if self.obj_tform4x4_objs is None:
            self.obj_tform4x4_objs = self.read_obj_tform4x4_objs()
        return self.obj_tform4x4_objs

    def read_mesh(self, mesh_type=None, device="cpu", tform_obj_type=None):
        # mesh = Meshes.read_from_ply_file(
        #    fpath=self.get_fpath_mesh(mesh_type=mesh_type),
        #    device=device,
        # )
        if mesh_type is None:
            mesh_type = self.mesh_type
        fpaths_meshes = self.get_fpaths_meshs(mesh_type=mesh_type)
        mesh = Meshes.read_from_ply_files(fpaths_meshes=fpaths_meshes)

        objscentric_tform4x4_objs = mesh.get_objscentric_tform4x4_objs()
        # objs_tform4x4_objscentric = inv_tform4x4(objscentric_tform4x4_objs)

        mesh.transf3d(objscentric_tform4x4_objs)
        # tform_obj = self.get_tform_objs(tform_obj_type=tform_obj_type)
        # if tform_obj is not None:
        #     tform_obj = tform_obj.to(device=device)
        #     mesh.verts = transf3d_broadcast(pts3d=mesh.verts, transf4x4=tform_obj)

        if (mesh_type is None or mesh_type == self.mesh_type) and (
            tform_obj_type is None or tform_obj_type == self.tform_obj_type
        ):
            self.mesh = mesh
        return mesh

    def get_mesh(self, mesh_type=None, clone=False, device="cpu", tform_obj_type=None):
        if (
            self.mesh is not None
            and (mesh_type is None or mesh_type == self.mesh_type)
            and (tform_obj_type is None or tform_obj_type == self.tform_obj_type)
        ):
            mesh = self.mesh
        else:
            mesh = self.read_mesh(
                mesh_type=mesh_type,
                device=device,
                tform_obj_type=tform_obj_type,
            )

        if not clone:
            return mesh
        else:
            return mesh.clone()

    # def read_mesh(self, mesh_type=None, device="cpu"):
    #     fpaths_meshes = self.get_fpaths_meshs() # self.meta.rfpaths_meshs
    #     from od3d.cv.geometry.objects3d.meshes import Meshes
    #     mesh = Meshes.read_from_ply_files(fpaths_meshes=fpaths_meshes)
    #
    #     # mesh = Meshes.read_from_ply_file(
    #     #     fpath=self.get_fpath_mesh(mesh_type=mesh_type),
    #     #     device=device,
    #     # )
    #
    #     if (mesh_type is None or mesh_type == self.mesh_type):
    #         self.mesh = mesh
    #     return mesh
    #
    #
    #
    #
    # def get_mesh(self, mesh_type=None, clone=False, device="cpu"):
    #     if self.mesh is not None and (mesh_type is None or mesh_type == self.mesh_type):
    #         mesh = self.mesh
    #     else:
    #         mesh = self.read_mesh(
    #             mesh_type=mesh_type,
    #             device=device,
    #         )
    #
    #     if not clone:
    #         return mesh
    #     else:
    #         return mesh.clone()

    # def get_mesh(self, mesh_type=None, clone=False, device="cpu"):
    #     if self.mesh is not None and (mesh_type is None or mesh_type == self.mesh_type):
    #         mesh = self.mesh
    #     else:
    #         mesh = self.read_mesh(
    #             mesh_type=mesh_type,
    #             device=device,
    #         )
    #
    #     if not clone:
    #         return mesh
    #     else:
    #         return mesh.clone()
    #
    # def read_mesh(self, mesh_type=None):
    #
    #
    #     # return self.sequence.read_mesh()
    #
    #
    # def get_fpath_mesh(self, mesh_type=None):
    #     raise NotImplementedError
    #     #return self.sequence.get_fpath_mesh(mesh_type=mesh_type)
    #
    # def get_mesh(self, mesh_type=None, clone=False, device="cpu", tform_obj_type=None):

    # return self.sequence.get_mesh(mesh_type=mesh_type, clone=clone, device=device, tform_obj_type=tform_obj_type)

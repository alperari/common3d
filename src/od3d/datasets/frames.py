import logging

from od3d.cv.visual.draw import get_colors

logger = logging.getLogger(__name__)

from pathlib import Path
import torch
from od3d.cv.geometry.transform import tform4x4
from od3d.datasets.frame import (
    OD3D_FRAME_MODALITIES,
    OD3D_FRAME_MODALITIES_STACKABLE,
    OD3D_Frame,
)
from od3d.datasets.sequence import OD3D_Sequence
from typing import List
from od3d.cv.geometry.objects3d.meshes import Meshes, MESH_RENDER_MODALITIES, PROJECT_MODALITIES
from dataclasses import dataclass


@dataclass
class OD3D_Frames:
    modalities: List[OD3D_FRAME_MODALITIES]
    length: int
    name: List[str]
    name_unique: List[str]
    item_id: torch.LongTensor
    path_co3d: Path = None
    size: torch.Tensor = None
    sizes: torch.Tensor = None
    cam_intr4x4: torch.Tensor = None
    cam_intr4x4s: torch.Tensor = None
    cam_tform4x4_obj: torch.Tensor = None
    obj_tform4x4_objs: torch.Tensor = None
    cam_tform4x4_objs: torch.Tensor = None
    object_id: torch.LongTensor = None
    objects_ids: torch.LongTensor = None
    category: List[str] = None
    category_id: torch.LongTensor = None
    categories: List[List[str]] = None
    categories_ids: torch.LongTensor = None
    dtype: torch.dtype = None
    device: torch.device = None
    sequence_name_unique: str = None
    sequence: OD3D_Sequence = None
    rgb: torch.Tensor = None
    rgbs: torch.Tensor = None
    rgb_mask: torch.Tensor = None
    depth: torch.Tensor = None
    mask: torch.Tensor = None
    pxl_cat_id: torch.Tensor = None
    mask_dt: torch.Tensor = None
    mask_inv_dt: torch.Tensor = None
    depth_mask: torch.Tensor = None
    kpts2d_annot: torch.Tensor = None
    kpts2d_annots: torch.Tensor = None
    kpts2d_annots_ids: torch.Tensor = None
    kpts2d_annot_vsbl: torch.Tensor = None
    kpts_names: torch.Tensor = None
    kpts3d: torch.Tensor = None
    rays_center3d: torch.Tensor = None
    bbox: torch.Tensor = None
    bboxs: torch.Tensor = None
    mesh_id_in_batch: List = None
    mesh: Meshes = None
    pcl: List[torch.Tensor] = None

    @staticmethod
    def get_frames_from_list(
        frames: List[OD3D_Frame],
        modalities: List[OD3D_FRAME_MODALITIES],
        dtype,
        device,
    ):
        frame0 = frames[0]
        length = len(frames)
        name = [frame.name for frame in frames]
        name_unique = [frame.name_unique for frame in frames]
        dtype = dtype
        device = device
        item_id = torch.cat(
            [torch.LongTensor([frame.item_id]) for frame in frames],
            dim=0,
        )
        path_co3d = frame0.path_raw
        # size = frame0.size # .to(device=device)

        modality_kwargs = {}
        for modality in modalities:
            modality_data = [frame.get_modality(modality) for frame in frames]
            if modality in list(OD3D_FRAME_MODALITIES_STACKABLE):
                if type(modality_data[0]) == int:
                    modality_data = torch.LongTensor(modality_data)
                elif type(modality_data[0]) == float:
                    modality_data = torch.FloatTensor(modality_data)
                else:
                    modality_data = torch.stack(modality_data, dim=0)
            modality_kwargs[modality] = modality_data

        if OD3D_FRAME_MODALITIES.SIZE in modality_kwargs.keys():
            modality_kwargs[OD3D_FRAME_MODALITIES.SIZE] = modality_kwargs[
                OD3D_FRAME_MODALITIES.SIZE
            ][0]

        if OD3D_FRAME_MODALITIES.MESH in modality_kwargs.keys():
            modality_kwargs[OD3D_FRAME_MODALITIES.MESH], modality_kwargs[OD3D_FRAME_MODALITIES.MESH_ID_IN_BATCH] = (
            Meshes.cat_meshes(
                modality_kwargs[OD3D_FRAME_MODALITIES.MESH],
            ))

        return OD3D_Frames(
            modalities=modalities,
            length=length,
            name=name,
            name_unique=name_unique,
            item_id=item_id,
            path_co3d=path_co3d,
            dtype=dtype,
            device=device,
            **modality_kwargs,
        )

    def __getitem__(cls, x):
        if isinstance(x, int):
            return cls.get_items(items =[x])
        else:
            return cls.get_items(items=x)

    def get_items(self, items):
        modality_kwargs = {}
        for modality in self.modalities:
            modality_data = getattr(self, modality)
            if modality in list(OD3D_FRAME_MODALITIES_STACKABLE):
                modality_data = (
                    modality_data[items] if modality_data is not None else None
                )
            else:
                if modality == OD3D_FRAME_MODALITIES.MESH:
                    modality_data = modality_data[items]
                elif modality == OD3D_FRAME_MODALITIES.SIZE:
                    modality_data = modality_data
                else:
                    modality_data = (
                        [modality_data[item] for item in items]
                        if modality_data is not None
                        else None
                    )
            modality_kwargs[modality] = modality_data
        return OD3D_Frames(
            modalities=self.modalities,
            length=len(items),
            name=[self.name[item] for item in items],
            name_unique=[self.name_unique[item] for item in items],
            dtype=self.dtype,
            device=self.device,
            item_id=self.item_id[items],
            path_co3d=self.path_co3d,
            **modality_kwargs,
        )

    @property
    def cam_proj4x4_obj(self):
        return tform4x4(self.cam_intr4x4, self.cam_tform4x4_obj)

    def __len__(self):
        return self.length

    def visualize(self, cuboids: Meshes = None):
        from od3d.cv.visual.show import show_img
        from od3d.cv.visual.blend import blend_rgb
        from od3d.cv.visual.draw import draw_pixels, draw_bbox, draw_bboxs
        from od3d.cv.geometry.transform import proj3d2d_broadcast

        # show_pcl
        # print(self.rfpath_pcl[0])
        # show_img(self.rgb[0])

        # verts, _ = load_ply(str(self.path_co3d.joinpath(self.rfpath_pcl[0])))
        # verts = verts.to(self.device)

        # how_pcl(verts, cam_tform4x4_obj=self.cam_tform4x4_obj[0], cam_intr4x4=self.cam_intr4x4[0], img_size=self.size)

        # verts, faces = load_ply(filename)

        if OD3D_FRAME_MODALITIES.MASK in self.modalities:
            if OD3D_FRAME_MODALITIES.RGB in self.modalities:
                img = blend_rgb(self.rgb[0], self.mask[0] * 255)
            elif OD3D_FRAME_MODALITIES.RGBS in self.modalities:
                img = blend_rgb(self.rgbs[0], self.mask[0] * 255)

        else:
            if OD3D_FRAME_MODALITIES.RGB in self.modalities:
                img = self.rgb[0]
            if OD3D_FRAME_MODALITIES.RGBS in self.modalities:
                # from od3d.cv.visual.show import imgs_to_img
                img = self.rgbs[0][0]
                img2 = self.rgbs[0][1]

        if OD3D_FRAME_MODALITIES.KPTS2D_ANNOTS in self.modalities:
            img = draw_pixels(
                pxls=self.kpts2d_annots[0][0],
                img=img,
                colors=[0.0, 0.0, 255.0],
            )

            if OD3D_FRAME_MODALITIES.RGBS in self.modalities:
                img2 = draw_pixels(
                    pxls=self.kpts2d_annots[0][1],
                    img=img2,
                    colors=[0.0, 0.0, 255.0],
                )

        if OD3D_FRAME_MODALITIES.PXL_CAT_ID in self.modalities:
            if OD3D_FRAME_MODALITIES.RGB in self.modalities:
                cat_count = int(self.pxl_cat_id.max()) +1
                pxl_cat_id_long = self.pxl_cat_id.long()
                device = pxl_cat_id_long.device
                pxl_cat_onehot = torch.nn.functional.one_hot(pxl_cat_id_long, num_classes=cat_count).to(device)
                cat_colors = get_colors(cat_count, device=device, first_white=True)
                pxl_cat_onehot_rgb = torch.einsum('bchwo,or->brhw', pxl_cat_onehot * 1., cat_colors)
                from od3d.cv.visual.draw import draw_text_in_rgb
                pxl_cat_ids_unique = pxl_cat_id_long.unique()
                fontColors = cat_colors[pxl_cat_ids_unique].clone() * 255
                pxl_cat_onehot_rgb[0] = draw_text_in_rgb(img=pxl_cat_onehot_rgb[0],
                                                         text="\n".join(f"cat {id}" for id in pxl_cat_ids_unique.tolist()),
                                                         fontColor=fontColors)

                img = blend_rgb(self.rgb[0], pxl_cat_onehot_rgb[0] * 255)

        if OD3D_FRAME_MODALITIES.KPTS2D_ANNOT in self.modalities:
            img = draw_pixels(
                pxls=self.kpts2d_annot[0][self.kpts2d_annot_vsbl[0]],
                img=img,
                colors=[0.0, 0.0, 255.0],
            )
            # img = draw_pixels(pxls=self.kpts2d_annot[0], img=img, colors=[0., 0., 255.])

            kpts3d_inf_mask = torch.isinf(self.kpts3d[0]).any(dim=-1)
            if kpts3d_inf_mask.sum() > 0:
                logger.warning(
                    f"There are {kpts3d_inf_mask.sum()} kpts with infinity for label {self.category[0]}",
                )
            kpts3d = self.kpts3d[0][~kpts3d_inf_mask].to(self.device)
            kpts3d = torch.cat([kpts3d, torch.zeros(size=(1, 3), device=kpts3d.device)])

            kpts3d2d = proj3d2d_broadcast(proj4x4=self.cam_proj4x4_obj[0], pts3d=kpts3d)
            img = draw_pixels(pxls=kpts3d2d, img=img, colors=[0.0, 255.0, 0.0])

        if OD3D_FRAME_MODALITIES.BBOX in self.modalities:
            img = draw_bbox(img=img, bbox=self.bbox[0])

        if OD3D_FRAME_MODALITIES.BBOXS in self.modalities:
            img = draw_bboxs(img=img, bboxs=self.bboxs[0])
            if OD3D_FRAME_MODALITIES.RGBS in self.modalities:
                img2 = draw_bboxs(img=img2, bboxs=self.bboxs[0]) #

        if OD3D_FRAME_MODALITIES.PCL in self.modalities:
            from od3d.datasets.co3d.enum import PCL_SOURCES
            from od3d.cv.visual.show import show_scene

            pcl = self.sequence[0].get_pcl(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN)
            pts3d_colors = self.sequence[0].get_pcl_colors(
                pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN,
            )

            show_scene(pts3d=[pcl], pts3d_colors=[pts3d_colors])

        if OD3D_FRAME_MODALITIES.DEPTH in self.modalities:
            img = blend_rgb(img, self.depth[0])
            # from od3d.cv.visual.show import show_scene
            # from od3d.cv.geometry.transform import depth2pts3d_grid, transf3d_broadcast, inv_tform4x4
            # pts3d_obj = depth2pts3d_grid(self.depth[0], cam_intr4x4=self.cam_intr4x4[0])[0].reshape(3, -1).permute(1, 0).to(torch.float)
            # pts3d_obj = transf3d_broadcast(pts3d=pts3d_obj, transf4x4=inv_tform4x4(self.cam_tform4x4_obj[0]))
            # show_scene(pts3d=[pts3d_obj],
            #            cams_tform4x4_world=self.cam_tform4x4_obj[:1], cams_intr4x4=self.cam_intr4x4[:1],
            #            cams_imgs=self.rgb[:1])

        if OD3D_FRAME_MODALITIES.MESH in self.modalities:
            # from od3d.cv.geometry.fit3d2d import fit_se3_to_corresp_3d_2d_and_masks
            cam_tform4x4_obj = self.cam_tform4x4_obj[:1]
            # cam_tform4x4_obj = fit_se3_to_corresp_3d_2d_and_masks(masks_in=self.kpts2d_annot_vsbl[0][None,] * (~torch.isinf(self.kpts3d[0]).any(dim=-1))[None,], #
            #                                                  pts1=self.kpts3d[0].T, pxl2=self.kpts2d_annot[0].T,
            #                                                  proj_mat=self.cam_intr4x4[0][:2, :3].to(device='cpu'))
            # cam_tform4x4_obj = cam_tform4x4_obj.to(device='cuda:0')
            # self.mesh.verts *= 5. # this is ionly for pascal3d required currentlay

            #cam_tform4x4_obj[:, :3, :4] = cam_tform4x4_obj[:, :3, :4] / cam_tform4x4_obj[0, :3, :3].norm(dim=-1, keepdim=True)
            #cam_tform4x4_obj[:, :3, :3] *= 5
            #from od3d.cv.visual.show import show_scene
            #self.mesh.transf3d(self.obj_tform4x4_objs[0])
            #self.mesh.verts = self.mesh.verts - self.mesh.verts.mean(dim=0, keepdim=True)
            #show_scene(cams_tform4x4_world=self.cam_tform4x4_obj, cams_intr4x4=self.cam_intr4x4, meshes=self.mesh)

            logger.info(self.mesh.get_ranges())

            if self.obj_tform4x4_objs is not None:
                logger.info(f"mesh count {len(self.mesh_id_in_batch)}, tforms count {len(torch.cat(self.obj_tform4x4_objs, dim=0))}")

            img = blend_rgb(
                img,
                (
                    self.mesh.render(
                        cams_tform4x4_obj=cam_tform4x4_obj,
                        cams_intr4x4=self.cam_intr4x4[:1],
                        imgs_sizes=self.size,
                        objects_ids=self.mesh_id_in_batch[0:1], # torch.LongTensor([0]),
                        obj_tform4x4_objs=torch.stack(self.obj_tform4x4_objs[0:1]) if self.obj_tform4x4_objs is not None else None,
                        modalities=PROJECT_MODALITIES.PT3D_NCDS,
                        broadcast_batch_and_cams=False
                    )[0]
                    * 255
                ).to(dtype=self.rgb.dtype, device=img.device),
            )

        # elif self.sequence is not None and self.sequence[0].cuboid_labeled:
        #     # if self.sequence_name
        #     img = blend_rgb(img, (self.sequence[0].cuboid.render_feats(
        #                             cams_tform4x4_obj=self.cam_tform4x4_obj[:1],
        #                             cams_intr4x4=self.cam_intr4x4[:1],
        #                             imgs_sizes=self.size, meshes_ids=torch.LongTensor([0]),
        #                             modality=PROJECT_MODALITIES.PT3D_NCDS)[0]).to(dtype=self.rgb.dtype, device=img.device))

        # mix_real_with_synthetic = draw_pixels(mix_real_with_synthetic,
        #                                      proj3d2d_broadcast(pts3d=torch.cat((pts3d, self.kpts3d[0, self.kpts3d_vsbl[0]])),
        #                                               proj4x4=self.cam_proj4x4_obj[0]), colors=(0, 255, 0))
        # mix_real_with_synthetic = draw_pixels(mix_real_with_synthetic, self.kpts2d_annot[0, self.kpts2d_annot_vsbl[0]],
        #                                     colors=(0, 0, 255), radius_in=2, radius_out=4)

        if OD3D_FRAME_MODALITIES.RGBS in self.modalities:
            from od3d.cv.visual.show import imgs_to_img
            img = imgs_to_img(rgbs=[img, img2])

        show_img(img)
        return img

    def to(self, device: torch.device):
        if self.device != device:
            for k, a in self.__dict__.items():
                if isinstance(a, torch.Tensor):
                    setattr(self, k, a.to(device))
                if isinstance(a, list):
                    if len(a) > 0 and isinstance(a[0], torch.Tensor):
                        setattr(self, k, [el.to(device) for el in a])
                if isinstance(a, Meshes):
                    a.to(device)
                    # self.__dict__[k] = a.to(device)
            self.device = device

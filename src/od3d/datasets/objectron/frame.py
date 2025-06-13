import logging

logger = logging.getLogger(__name__)
from typing import List
import torch
from pathlib import Path

from od3d.datasets.frame_meta import (
    OD3D_FrameMeta,
    OD3D_FrameMetaRGBMixin,
    OD3D_FrameMetaCategoryMixin,
    OD3D_FrameMetaSizeMixin,
    OD3D_FrameMetaSequenceMixin,
    OD3D_FrameMetaCamTform4x4ObjMixin,
    OD3D_FrameMetaCamIntr4x4Mixin,
    OD3D_FrameMetaKpts2D3DMixin,
)
from dataclasses import dataclass

from dataclasses import dataclass

from od3d.datasets.frame import (OD3D_CamProj4x4ObjMixin, OD3D_FrameRGBMixin, OD3D_FrameRGBMaskMixin,
                                 OD3D_FrameCategoryMixin, OD3D_FrameSequenceMixin, OD3D_FrameSizeMixin, OD3D_Frame)
from od3d.datasets.objectron.enum import MAP_CATEGORIES_OBJECTRON_TO_OD3D

@dataclass
class Objectron_FrameMeta(
    OD3D_FrameMetaCamIntr4x4Mixin,
    OD3D_FrameMetaCamTform4x4ObjMixin,
    OD3D_FrameMetaKpts2D3DMixin,
    OD3D_FrameMetaRGBMixin,
    OD3D_FrameMetaSizeMixin,
    OD3D_FrameMetaCategoryMixin,
    OD3D_FrameMetaSequenceMixin,
    OD3D_FrameMeta,
):
    @staticmethod
    def load_from_raw(
        name: str,
        category: str,
        sequence_name: str,
        rfpath_rgb: Path,
        l_size: List,
        annotation
    ):
        if len(annotation.objects) != 1:
            logger.warning(f"sequence {sequence_name} contains annotation of not exactly one object ({len(annotation.objects)})")

        if len(annotation.objects) < 1:
            return None
        # annotation.objects[object_id].rotation
        # annotation.objects[object_id].translation
        # annotation.objects[object_id].scale
        # annotation.objects[object_id].category
        # annotation.objects[object_id].type

        # annotation.frame_annotations[frame_id].camera.transform
        # annotation.frame_annotations[frame_id].camera.intrinsics
        # annotation.frame_annotations[frame_id].camera.projection_matrix
        # annotation.frame_annotations[frame_id].camera.view_matrix
        # annotation.frame_annotations[frame_id].annotations[object_id].keypoints[keypoint_id].point_2d.x
        # annotation.frame_annotations[frame_id].annotations[object_id].keypoints[keypoint_id].point_2d.y
        # annotation.frame_annotations[frame_id].annotations[object_id].keypoints[keypoint_id].point_2d.depth
        # annotation.frame_annotations[frame_id].annotations[object_id].keypoints[keypoint_id].point_3d.x
        # annotation.frame_annotations[frame_id].annotations[object_id].keypoints[keypoint_id].point_3d.y
        # annotation.frame_annotations[frame_id].annotations[object_id].keypoints[keypoint_id].point_3d.z
        import numpy as np
        frame_id = int(name)
        annotation_frame = annotation.frame_annotations[frame_id]
        cam_intr4x4 = np.eye(4)
        cam_intr4x4[:3, :3] = np.array(annotation_frame.camera.intrinsics).reshape(3, 3)
        cam_intr4x4 = torch.from_numpy(cam_intr4x4)


        # cam_tform4x4_obj = torch.from_numpy(np.array(annotation_frame.camera.transform).reshape(4, 4))
        cam_tform4x4_world = torch.from_numpy(np.array(annotation_frame.camera.view_matrix).reshape(4, 4))

        from od3d.cv.geometry.transform import transf4x4_from_rot3x3_and_transl3, tform4x4, transf3d_broadcast, proj3d2d_broadcast, inv_tform4x4
        rot3x3_obj = torch.from_numpy(np.array(annotation.objects[0].rotation).reshape(3, 3))
        transl3_obj = torch.from_numpy(np.array(annotation.objects[0].translation))

        world_tform4x4_obj = transf4x4_from_rot3x3_and_transl3(rot3x3=rot3x3_obj, transl3=transl3_obj) # without scale
        cam_tform4x4_obj = tform4x4(cam_tform4x4_world, world_tform4x4_obj)

        cam_tform4x4_cam_objectron = torch.eye(4, dtype=torch.double)
        # negative z
        cam_tform4x4_cam_objectron[2, 2] = -1

        # swap x and y
        cam_tform4x4_cam_objectron[0, 0] = 0
        cam_tform4x4_cam_objectron[1, 1] = 0
        cam_tform4x4_cam_objectron[0, 1] = 1
        cam_tform4x4_cam_objectron[1, 0] = 1

        # # swap x and y
        cam_intr4x4 = cam_intr4x4[[1, 0, 2, 3, ]].clone()
        cam_intr4x4[0, 0] = cam_intr4x4[0, 1]
        cam_intr4x4[0, 1] = 0.
        cam_intr4x4[1, 1] = cam_intr4x4[1, 0]
        cam_intr4x4[1, 0] = 0.

        cam_tform4x4_obj = tform4x4(cam_tform4x4_cam_objectron, cam_tform4x4_obj)

        # swap y and z
        obj_tform_obj_objectron = torch.eye(4).to(cam_tform4x4_obj.device, cam_tform4x4_obj.dtype)
        obj_tform_obj_objectron[1, 1] = 0.
        obj_tform_obj_objectron[2, 2] = 0.
        obj_tform_obj_objectron[1, 2] = 1.
        obj_tform_obj_objectron[2, 1] = -1.
        cam_tform4x4_obj = tform4x4(cam_tform4x4_obj, obj_tform_obj_objectron)

        cam_proj4x4_obj = tform4x4(cam_intr4x4, cam_tform4x4_obj)

        # center, one, two
        scale_obj_pts3d = torch.from_numpy(np.array(annotation.objects[0].scale))
        obj_bbox3d = torch.from_numpy(np.stack([
            np.array([
                annotation.objects[0].keypoints[i].x,
                annotation.objects[0].keypoints[i].y,
                annotation.objects[0].keypoints[i].z])
            for i in range(1, 9)])) * scale_obj_pts3d[None,]

        obj_bbox3d = transf3d_broadcast(pts3d=obj_bbox3d, transf4x4=inv_tform4x4(obj_tform_obj_objectron))
        # # 8x3
        # cam_bbox3d = torch.from_numpy(np.stack([
        #     np.array([
        #         annotation.frame_annotations[frame_id].annotations[0].keypoints[i].point_3d.x,
        #         annotation.frame_annotations[frame_id].annotations[0].keypoints[i].point_3d.y,
        #         annotation.frame_annotations[frame_id].annotations[0].keypoints[i].point_3d.z])
        #     for i in range(1, 9)]))
        #
        # cam_bbox2d = torch.from_numpy(np.stack([
        #     np.array([
        #         annotation.frame_annotations[frame_id].annotations[0].keypoints[i].point_2d.x,
        #         annotation.frame_annotations[frame_id].annotations[0].keypoints[i].point_2d.y])
        #     for i in range(1, 9)]))
        #
        # cam_bbox3d_pts3d = transf3d_broadcast(pts3d=obj_bbox3d, transf4x4=cam_tform4x4_obj)


        cam_bbox2d = proj3d2d_broadcast(pts3d=obj_bbox3d.clone(), proj4x4=cam_proj4x4_obj)

        # cam_bbox2d = torch.from_numpy(np.stack([
        #     np.array([
        #         annotation.frame_annotations[frame_id].annotations[0].keypoints[i].point_2d.x,
        #         annotation.frame_annotations[frame_id].annotations[0].keypoints[i].point_2d.y])
        #     for i in range(1, 9)]))
        # cam_bbox2d[:, 0] = cam_bbox2d[:, 0] * l_size[1] # x * W
        # cam_bbox2d[:, 1] = cam_bbox2d[:, 1] * l_size[0] # y * H


        l_cam_intr4x4 = cam_intr4x4.tolist()
        l_cam_tform4x4_obj = cam_tform4x4_obj.tolist()
        l_kpts3d = obj_bbox3d.tolist()
        l_kpts2d_annot = cam_bbox2d.tolist()
        l_kpts2d_annot_vsbl = list((True,) * len(l_kpts3d))
        kpts_names = ['left_back_bottom', 'left_front_bottom', 'left_back_top', 'left_front_top',
                      'right_back_bottom', 'right_front_bottom', 'right_back_top', 'right_front_top']

        return Objectron_FrameMeta(
            rfpath_rgb=rfpath_rgb,
            category=category,
            sequence_name=sequence_name,
            l_size=l_size,
            name=name,
            l_cam_intr4x4=l_cam_intr4x4,
            l_cam_tform4x4_obj=l_cam_tform4x4_obj,
            l_kpts3d=l_kpts3d,
            l_kpts2d_annot=l_kpts2d_annot,
            l_kpts2d_annot_vsbl=l_kpts2d_annot_vsbl,
            kpts_names=kpts_names
        )

    @staticmethod
    def get_rfpath_frame_meta_with_category_sequence_and_frame_name(
            category: str,
            sequence_name: str,
            name: str,
    ):
        return Objectron_FrameMeta.get_rfpath_metas().joinpath(
            category,
            sequence_name,
            name + ".yaml",
        )

    @staticmethod
    def get_fpath_frame_meta_with_category_sequence_and_frame_name(
            path_meta: Path,
            category: str,
            sequence_name: str,
            name: str,
    ):
        return path_meta.joinpath(
            Objectron_FrameMeta.get_rfpath_frame_meta_with_category_sequence_and_frame_name(
                category=category,
                sequence_name=sequence_name,
                name=name,
            ),
        )



## NOT USED
# meta_fpath = 'OBJECTRON/bike/batch_0_0/meta.pbdata'
#
# from od3d.datasets.objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol
# import struct
# import numpy as np
#
# sequence_geometry = []
# with open(meta_fpath, 'rb') as pb:
#     proto_buf = pb.read()
#
#     i = 0
#     frame_number = 0
#
#     points = []
#     while i < len(proto_buf) // 10:
#         # print(i)
#         # print(len(proto_buf))
#         # Read the first four Bytes in little endian '<' integers 'I' format
#         # indicating the length of the current message.
#         msg_len = struct.unpack('<I', proto_buf[i:i + 4])[0]
#         i += 4
#         message_buf = proto_buf[i:i + msg_len]
#         i += msg_len
#         # ARMeshGeometry
#         frame_data = ar_metadata_protocol.ARFrame()
#         frame_data.ParseFromString(message_buf)
#         # print(frame_data.raw_feature_points.point)
#         current_points = [np.array([v.x, v.y, v.z])
#                           for v in frame_data.raw_feature_points.point]
#         current_points = np.array(current_points)
#
#         points.append(current_points)
#
# points = np.concatenate(points, axis=0)

from od3d.datasets.object import OD3D_PCLTypeMixin, OD3D_SequenceSfMTypeMixin
from od3d.datasets.frame import (OD3D_FrameKpts2d3dMixin, OD3D_FrameMeshMixin, OD3D_FrameRaysCenter3dMixin,
                                 OD3D_FrameTformObjMixin, OD3D_CamProj4x4ObjMixin, OD3D_FrameRGBMaskMixin,
                                 OD3D_FrameMaskMixin, OD3D_FrameRGBMixin, OD3D_FrameCategoryMixin, OD3D_FrameSizeMixin,
                                 OD3D_MeshTypeMixin, OD3D_Frame, OD3D_FrameBBoxFromKpts2d3dMixin)

@dataclass
class Objectron_Frame(
    OD3D_FrameMeshMixin,
    OD3D_FrameRaysCenter3dMixin,
    OD3D_FrameTformObjMixin,
    OD3D_CamProj4x4ObjMixin,
    OD3D_FrameRGBMaskMixin,
    OD3D_FrameMaskMixin,
    OD3D_FrameRGBMixin,
    OD3D_FrameCategoryMixin,
    OD3D_FrameSequenceMixin,
    OD3D_MeshTypeMixin,
    OD3D_PCLTypeMixin,
    OD3D_SequenceSfMTypeMixin,
    OD3D_FrameBBoxFromKpts2d3dMixin,
    OD3D_FrameSizeMixin,
    OD3D_Frame,
):
    meta_type = Objectron_FrameMeta
    map_categories_to_od3d = MAP_CATEGORIES_OBJECTRON_TO_OD3D

    def __post_init__(self):
        # hack: prevents circular import
        from od3d.datasets.objectron.sequence import Objectron_Sequence

        self.sequence_type = Objectron_Sequence


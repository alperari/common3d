import torchvision
from od3d.cv.transforms.transform import OD3D_Transform
from od3d.cv.geometry.transform import scale1d_tform4x4_with_dist, get_scale1d_tform4x4

class CamTform4x4ObjTform4x4Objs(OD3D_Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, frame):
        if 'obj_tform4x4_objs' in frame.modalities:
            obj_tform4x4_objs = frame.get_obj_tform4x4_objs()[0]
            cam_tform4x4_obj = frame.get_cam_tform4x4_obj()
            cam_tform4x4_obj = cam_tform4x4_obj @ obj_tform4x4_objs
            frame.cam_tform4x4_obj = scale1d_tform4x4_with_dist(cam_tform4x4_obj)
        return frame

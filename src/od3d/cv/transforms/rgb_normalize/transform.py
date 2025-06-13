from od3d.cv.transforms.transform import OD3D_Transform
from torchvision.transforms.transforms import Normalize
from od3d.datasets.frame import OD3D_FRAME_MODALITIES


class RGB_Normalize(OD3D_Transform):
    def __init__(self, mean=None, std=None):
        super().__init__()
        self.normalize = Normalize(mean=mean, std=std)

    def __call__(self, frame):
        if OD3D_FRAME_MODALITIES.RGB in frame.modalities:
            frame.rgb = self.normalize(frame.get_rgb())
        if OD3D_FRAME_MODALITIES.RGBS in frame.modalities:
            frame.rgbs = [self.normalize(rgb) for rgb in frame.get_rgbs()]
        return frame

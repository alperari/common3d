from pathlib import Path

import torch
from od3d.cv.geometry.objects3d.meshes import Meshes
from od3d.cv.visual.show import show_scene
from pytorch3d.ops import cot_laplacian

bunny = Meshes.load_by_name("bunny")
bunny.cuda()
# bunny.verts = bunny.verts * 20
bunny.visualize_coarse_labels()

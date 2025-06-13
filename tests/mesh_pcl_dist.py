import logging
from logging import getLogger
from pathlib import Path

import torch
from od3d.cv.geometry.fit.phase_corr import get_affine_from_imgs
from od3d.cv.geometry.fit.phase_corr import get_corr_amax
from od3d.cv.geometry.fit.phase_corr import logpolar_filter
from od3d.cv.geometry.fit.phase_corr import polar_transformer
from od3d.cv.io import read_image
from od3d.cv.visual.crop import crop
from od3d.cv.visual.show import show_img
from od3d.cv.visual.show import show_imgs

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

import torch
from pytorch3d.ops import cot_laplacian
from od3d.cv.geometry.objects3d.meshes import Meshes
from od3d.cv.visual.show import show_scene
from pathlib import Path

bunny = Meshes.load_by_name("bunny", faces_count=1000)
bunny.verts = bunny.verts * 20
bunny.verts = bunny.verts - bunny.verts.mean(dim=0, keepdim=True)

# show_scene(meshes=bunny, fpath=Path('bunny.webm'), viewpoints_count=10, fps=1, return_visualization=False)

bunnies = bunny.get_meshes_with_ids([0, 0], clone=True)

faces = bunnies.get_faces_with_mesh_id(mesh_id=0)
verts = bunnies.get_verts_with_mesh_id(mesh_id=0)
faces_normals = torch.cross(
    verts[faces[:, 1]] - verts[faces[:, 0]],
    verts[faces[:, 2]] - verts[faces[:, 0]],
    dim=-1,
)
faces_normals = faces_normals / (faces_normals.norm(dim=-1, keepdim=True) + 1e-15)
verts_dev = torch.randn(verts.shape)
pts3d = verts.clone() + verts_dev


def get_pts_face_dist(pts3d, verts, faces):
    #
    pts3d_signed_dist_to_faces = torch.einsum(
        "fd,fd->f",
        faces_normals,
        verts[faces[:, 0]],
    )[
        None,
    ] - torch.einsum(
        "fd,pd->pf",
        faces_normals,
        pts3d,
    )

    pts3d_on_faces = (
        pts3d[:, None] + faces_normals[None,] * pts3d_signed_dist_to_faces[:, :, None]
    )

    verts0_pts_faces = verts[faces[:, 0]][None,].repeat(pts3d_on_faces.shape[0], 1, 1)
    verts1_pts_faces = verts[faces[:, 1]][None,].repeat(pts3d_on_faces.shape[0], 1, 1)
    verts2_pts_faces = verts[faces[:, 2]][None,].repeat(pts3d_on_faces.shape[0], 1, 1)

    edge01_pts_faces = verts1_pts_faces - verts0_pts_faces
    edge02_pts_faces = verts2_pts_faces - verts0_pts_faces
    edge12_pts_faces = verts2_pts_faces - verts1_pts_faces

    pts3d_signed_dist_to_edge01 = torch.einsum(
        "pfd,pfd->pf",
        edge01_pts_faces,
        pts3d_on_faces - verts0_pts_faces,
    )
    pts3d_signed_dist_to_edge02 = torch.einsum(
        "pfd,pfd->pf",
        edge02_pts_faces,
        pts3d_on_faces - verts0_pts_faces,
    )
    pts3d_signed_dist_to_edge12 = torch.einsum(
        "pfd,pfd->pf",
        edge12_pts_faces,
        pts3d_on_faces - verts1_pts_faces,
    )

    pts3d_on_edge01 = (
        verts0_pts_faces + edge01_pts_faces * pts3d_signed_dist_to_edge01[:, :, None]
    )
    pts3d_on_edge02 = (
        verts0_pts_faces + edge02_pts_faces * pts3d_signed_dist_to_edge02[:, :, None]
    )
    pts3d_on_edge12 = (
        verts1_pts_faces + edge12_pts_faces * pts3d_signed_dist_to_edge12[:, :, None]
    )

    pts3d_on_edge01[
        (pts3d_signed_dist_to_edge01 > 1.0) + (pts3d_signed_dist_to_edge01 < 0.0)
    ] = -torch.inf
    pts3d_on_edge02[
        (pts3d_signed_dist_to_edge02 > 1.0) + (pts3d_signed_dist_to_edge02 < 0.0)
    ] = -torch.inf
    pts3d_on_edge12[
        (pts3d_signed_dist_to_edge12 > 1.0) + (pts3d_signed_dist_to_edge12 < 0.0)
    ] = -torch.inf

    A = torch.stack([verts0_pts_faces, verts1_pts_faces, verts2_pts_faces], dim=-1)
    B = pts3d_on_faces
    X = torch.linalg.solve(A, B)  # alpha beta gamma

    pts3d_on_faces_closest = pts3d_on_faces.clone() * 0
    pts3d_on_faces_closest_v0 = (
        (X[:, :, 0] > 0.0) * (X[:, :, 1] < 0.0) * (X[:, :, 2] < 0.0)
    )
    pts3d_on_faces_closest_v1 = (
        (X[:, :, 0] < 0.0) * (X[:, :, 1] > 0.0) * (X[:, :, 2] < 0.0)
    )
    pts3d_on_faces_closest_v2 = (
        (X[:, :, 0] < 0.0) * (X[:, :, 1] < 0.0) * (X[:, :, 2] > 0.0)
    )

    pts3d_on_faces_closest_e01 = (
        (X[:, :, 0] > 0.0) * (X[:, :, 1] > 0.0) * (X[:, :, 2] < 0.0)
    )
    pts3d_on_faces_closest_e02 = (
        (X[:, :, 0] > 0.0) * (X[:, :, 1] < 0.0) * (X[:, :, 2] > 0.0)
    )
    pts3d_on_faces_closest_e12 = (
        (X[:, :, 0] < 0.0) * (X[:, :, 1] > 0.0) * (X[:, :, 2] > 0.0)
    )

    pts3d_on_faces_closest_inside = (
        (X[:, :, 0] > 0.0) * (X[:, :, 1] > 0.0) * (X[:, :, 2] > 0.0)
    )
    pts3d_on_faces_closest_inside += pts3d_on_faces_closest_v0
    pts3d_on_faces_closest_inside += pts3d_on_faces_closest_v1
    pts3d_on_faces_closest_inside += pts3d_on_faces_closest_v2
    pts3d_on_faces_closest_inside += pts3d_on_faces_closest_e01
    pts3d_on_faces_closest_inside += pts3d_on_faces_closest_e02
    pts3d_on_faces_closest_inside += pts3d_on_faces_closest_e12

    # pts3d_on_faces_closest[~pts3d_on_faces_closest_inside] = 999999.

    pts3d_on_faces_closest[pts3d_on_faces_closest_v0] = verts0_pts_faces[
        pts3d_on_faces_closest_v0
    ]
    pts3d_on_faces_closest[pts3d_on_faces_closest_v1] = verts1_pts_faces[
        pts3d_on_faces_closest_v1
    ]
    pts3d_on_faces_closest[pts3d_on_faces_closest_v2] = verts2_pts_faces[
        pts3d_on_faces_closest_v2
    ]

    pts3d_on_faces_closest[pts3d_on_faces_closest_e01] = pts3d_on_edge01[
        pts3d_on_faces_closest_e01
    ]
    pts3d_on_faces_closest[pts3d_on_faces_closest_e02] = pts3d_on_edge02[
        pts3d_on_faces_closest_e02
    ]
    pts3d_on_faces_closest[pts3d_on_faces_closest_e12] = pts3d_on_edge12[
        pts3d_on_faces_closest_e12
    ]

    from od3d.cv.select import batched_index_select

    pts3d_on_face_closest_ids = (
        (pts3d[:, None] - pts3d_on_faces_closest).norm(dim=-1).min(dim=-1).indices
    )
    pts3d_on_face_closest = batched_index_select(
        pts3d_on_faces_closest,
        index=pts3d_on_face_closest_ids[:, None],
        dim=1,
    )[:, 0]
    pts3d_on_face_closest_inside = batched_index_select(
        pts3d_on_faces_closest_inside,
        index=pts3d_on_face_closest_ids[:, None],
        dim=1,
    )[:, 0]

    # pts3d_closest_dev_dist = (pts3d_on_face_closest - pts3d).norm(dim=-1)
    # verts_dev_dist = verts_dev.norm(dim=-1)

    pts3d_on_face_closest = pts3d_on_face_closest[pts3d_on_face_closest_inside]

    pts3d_dists = (pts3d - pts3d_on_face_closest).norm(dim=-1)

    pts3d_dist = pts3d_dists.mean()

    return pts3d_dist


# print((pts3d - pts3d_on_face_closest).norm(dim=-1).max())
# print(verts_dev.norm(dim=-1).max())
# , lines3d=[torch.stack([pts3d[:100], pts3d_on_face_closest[:100]],dim=-2 ]
# show_scene(pts3d=[verts, pts3d[:10000],  pts3d_on_face_closest[:10000] ], )
# print((pts3d_closest_dev_dist - verts_dev_dist).max())
# print((pts3d_closest_dev_dist - verts_dev_dist).max())

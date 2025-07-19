import logging

logger = logging.getLogger(__name__)

import torch
import numpy as np
from od3d.cv.geometry.objects3d.dmtet.dmtet_core import DMTet_Core
from od3d.cv.geometry.objects3d.meshes import Meshes
from typing import List

import torch
from typing import Union
from od3d.cv.geometry.objects3d.objects3d import (
    PROJECT_MODALITIES,
    FEATS_DISTR,
    FEATS_ACTIVATION,
)

from typing import Optional
from od3d.cv.geometry.objects3d.meshes_x_gaussians import Meshes_x_Gaussians
from od3d.cv.geometry.objects3d.meshes.meshes import (
    Meshes,
    FACE_BLEND_TYPE,
    OD3D_Meshes_Deform,
    RASTERIZER,
)
from omegaconf import DictConfig


class DMTet_x_Gaussians(Meshes_x_Gaussians):
    def __init__(
        self,
        verts: List[torch.Tensor],
        faces: List[torch.Tensor],
        feat_dim=128,
        objects_count=0,
        feats_objects: Union[bool, torch.Tensor] = False,
        feat_clutter_requires_param: Union[bool, torch.Tensor] = False,
        verts_uvs: List[torch.Tensor] = None,
        verts_coarse_count: int = 150,
        verts_coarse_prob_sigma: float = 0.01,
        feats_requires_grad=True,
        feat_clutter=False,
        feats_distribution=FEATS_DISTR.VON_MISES_FISHER,
        feats_activation=FEATS_ACTIVATION.NORM_DETACH,
        rgb: List[torch.Tensor] = None,
        verts_requires_grad=False,
        geodesic_prob_sigma=0.2,
        gaussian_splat_enabled=False,
        gaussian_splat_opacity=0.7,
        gaussian_splat_pts3d_size_rel_to_neighbor_dist=0.5,
        pt3d_raster_perspective_correct=False,
        device=None,
        dtype=None,
        rasterizer=RASTERIZER.NVDIFFRAST,
        face_blend_type=FACE_BLEND_TYPE.SOFT_SIGMOID_NORMALIZED,
        face_blend_count=2,
        face_opacity=1.0,
        face_opacity_face_sdf_sigma=1e-4,
        face_opacity_face_sdf_gamma=1e-4,
        instance_deform_net_config: DictConfig = None,
        gs_top_k=3,
        gs_scale=0.05,
        gs_opacity_requires_grad=False,
        gs_scale_requires_grad=False,
        gs_rotation_requires_grad=False,
        tet_res=16,
        sdf_symmetric=True,
        harmonic_functions_count=8,
        init_radius=1.0,
        **kwargs,
    ):
        super().__init__(
            verts=verts,
            faces=faces,
            feat_dim=feat_dim,
            objects_count=objects_count,
            feats_objects=None,
            verts_uvs=verts_uvs,
            feats_requires_grad=feats_requires_grad,
            feats_objects_requires_param=False,
            feat_clutter_requires_param=feat_clutter_requires_param,
            feat_clutter=feat_clutter,
            feats_distribution=feats_distribution,
            feats_activation=feats_activation,
            rgb=rgb,
            verts_requires_grad=verts_requires_grad,
            verts_requires_param=False,
            verts_coarse_count=verts_coarse_count,
            verts_coarse_prob_sigma=verts_coarse_prob_sigma,
            geodesic_prob_sigma=geodesic_prob_sigma,
            gaussian_splat_enabled=gaussian_splat_enabled,
            gaussian_splat_opacity=gaussian_splat_opacity,
            gaussian_splat_pts3d_size_rel_to_neighbor_dist=gaussian_splat_pts3d_size_rel_to_neighbor_dist,
            pt3d_raster_perspective_correct=pt3d_raster_perspective_correct,
            device=device,
            dtype=dtype,
            rasterizer=rasterizer,
            face_blend_type=face_blend_type,
            face_blend_count=face_blend_count,
            face_opacity=face_opacity,
            face_opacity_face_sdf_sigma=face_opacity_face_sdf_sigma,
            face_opacity_face_sdf_gamma=face_opacity_face_sdf_gamma,
            instance_deform_net_config=instance_deform_net_config,
            gs_top_k=gs_top_k,
            gs_scale=gs_scale,
            gs_opacity_requires_grad=gs_opacity_requires_grad,
            gs_scale_requires_grad=gs_scale_requires_grad,
            gs_rotation_requires_grad=gs_rotation_requires_grad,
        )

        self.init_radius = init_radius
        self.mesh_update_jitter_scale = 0.05

        self.marching_tets = DMTet_Core()
        self.sdf_coordmlps = torch.nn.ModuleList()
        self.feat_coordmlps = torch.nn.ModuleList()

        sdf_coord_mlp_cfg = DictConfig(
            {
                "num_layers": 5,
                "hidden_dim": 256,
                "out_dim": 1,
                "dropout": 0,
                "activation": None,  # None
                "symmetrize": sdf_symmetric,
            },
        )

        feat_coord_mlp_cfg = DictConfig(
            {
                "num_layers": 5,
                "hidden_dim": 256,
                "out_dim": feat_dim,
                "dropout": 0,
                "activation": None,  # feats_activation
                "symmetrize": False,
            },
        )

        # note: import on-the-fly to avoid circular import
        from od3d.models.heads.coordmlp import CoordMLP

        for m in range(self.meshes_count):
            # grid_scale = self.get_ranges()[m]
            # embedder_scalar = 2 * np.pi / grid_scale * 0.9  # originally (-0.5*s, 0.5*s) rescale to (-pi, pi) * 0.9
            self.sdf_coordmlps.append(
                CoordMLP(
                    in_dims=[0],
                    in_upsample_scales=[],
                    config=sdf_coord_mlp_cfg,
                    n_harmonic_functions=harmonic_functions_count,
                    embed_concat_pts=True,
                ).to(device, dtype),
            )

            self.feat_coordmlps.append(
                CoordMLP(
                    in_dims=[0],
                    in_upsample_scales=[],
                    config=feat_coord_mlp_cfg,
                    n_harmonic_functions=harmonic_functions_count,
                    embed_concat_pts=True,
                ).to(device, dtype),
            )

        if not verts_requires_grad:
            for param in self.sdf_coordmlps.parameters():
                param.requires_grad = False
        if not feats_requires_grad:
            for param in self.feat_coordmlps.parameters():
                param.requires_grad = False

        from pathlib import Path

        tets = np.load(
            Path(__file__).parent.resolve().joinpath(f"./{str(tet_res)}_tets.npz"),
        )
        if dtype is None:
            dtype = torch.float

        self.tets_scale = self.init_radius * 2.2
        self.tets_verts = (
            torch.tensor(tets["vertices"], dtype=dtype, device=device)
        ) * self.tets_scale  # verts original scale (-0.5, 0.5)
        self.tets_faces = torch.tensor(tets["indices"], dtype=torch.long, device=device)
        # self.generate_edges()
        self.update_dmtet(device=device, dtype=dtype, require_grad=False)

    def set_verts_requires_grad(self, verts_requires_grad):
        self.verts_requires_grad = verts_requires_grad
        for param in self.sdf_coordmlps.parameters():
            param.requires_grad = verts_requires_grad

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.marching_tets.to(*args, **kwargs)
        self.tets_verts = self.tets_verts.to(*args, **kwargs)
        self.tets_faces = self.tets_faces.to(*args, **kwargs)

    def cuda(self, *args, **kwargs):
        super().cuda(*args, **kwargs)
        self.marching_tets.cuda(*args, **kwargs)
        self.tets_verts = self.tets_verts.cuda(*args, **kwargs)
        self.tets_faces = self.tets_faces.cuda(*args, **kwargs)

    def eval(self, *args, **kwargs):
        super().eval(*args, **kwargs)
        self.update_dmtet(require_grad=False)

    def update_verts(self, require_grad=True):
        self.update_dmtet(require_grad=require_grad)
        # pass

    def update_dmtet(
        self,
        device=None,
        dtype=None,
        require_grad=None,
        require_feats_grad=None,
    ):
        if require_grad is None:
            require_grad = self.verts_requires_grad
        if require_feats_grad is None:
            require_feats_grad = require_grad
        # kaolin.non_commercial.FlexiCubes
        # https://kaolin.readthedocs.io/en/latest/modules/kaolin.non_commercial.html#kaolin.non_commercial.FlexiCubes
        verts = []
        faces = []
        feats = []
        if device is None:
            device = self.device
        if dtype is None:
            dtype = torch.get_default_dtype()

        for m in range(self.meshes_count):
            if require_grad:
                tets_verts = self.get_uniform_jittered_tets_verts(m)
                tets_sdfs = self.get_sdf(pts=tets_verts, object_id=m)
                _verts, _faces, _uvs, _uv_idx = self.marching_tets(
                    tets_verts,
                    tets_sdfs,
                    self.tets_faces,
                )

            else:
                with torch.no_grad():
                    if not require_feats_grad:
                        tets_verts = (
                            self.tets_verts.clone()
                        )  # self.get_uniform_jittered_tets_verts(m)
                    else:
                        tets_verts = self.get_uniform_jittered_tets_verts(m)
                    tets_sdfs = self.get_sdf(pts=tets_verts, object_id=m)
                    _verts, _faces, _uvs, _uv_idx = self.marching_tets(
                        tets_verts,
                        tets_sdfs,
                        self.tets_faces,
                    )

            # Safety check for empty meshes from marching tetrahedra, 
            # I implemented this after getting errors from initializing the values to 0, the values of sdf must be positive, negative and zero
            if _verts.shape[0] == 0:
                logger.warning(f"Marching tetrahedra produced empty mesh for object {m}")
                # Create minimal dummy mesh to prevent pipeline failure
                _verts = torch.zeros((1, 3), device=device, dtype=dtype)
                _faces = torch.zeros((0, 3), device=device, dtype=torch.long)
                
            if require_feats_grad:
                # logger.info(m)
                # logger.info(_verts.shape)
                _feats = self.get_feats(pts=_verts.detach(), object_id=m)
            else:
                with torch.no_grad():
                    _feats = self.get_feats(pts=_verts.detach(), object_id=m)

            verts.append(_verts)
            faces.append(_faces)
            feats.append(_feats)

        factory_kwargs = {"device": device, "dtype": dtype}

        self.verts_counts = [_verts.shape[0] for _verts in verts]
        self.verts_counts_max = max(self.verts_counts)

        if not self.feat_clutter_requires_param:
            if require_feats_grad:
                # logger.info(m)
                # logger.info(_verts.shape)
                _feat_clutter = self.get_feats(
                    pts=(torch.ones_like(_verts[0:1])).detach()
                    * (self.tets_scale / 2.0),
                    object_id=0,
                )[0]
            else:
                with torch.no_grad():
                    _feat_clutter = self.get_feats(
                        pts=(torch.ones_like(_verts[0:1])).detach()
                        * (self.tets_scale / 2.0),
                        object_id=0,
                    )[0]
            self.feat_clutter = _feat_clutter.to(**factory_kwargs)

        self.meshes_count = len(verts)
        self.verts = torch.cat([_verts for _verts in verts], dim=0).to(**factory_kwargs)
        self.feats_objects = torch.cat([_feats for _feats in feats], dim=0).to(
            **factory_kwargs,
        )

        self.faces = torch.cat([_faces for _faces in faces], dim=0).to(device=device)

        self.faces_counts = [_faces.shape[0] for _faces in faces]
        self.verts_counts_acc_from_0 = [0] + [
            sum(self.verts_counts[: i + 1]) for i in range(self.meshes_count)
        ]
        self.faces_counts_acc_from_0 = [0] + [
            sum(self.faces_counts[: i + 1]) for i in range(self.meshes_count)
        ]

        self.verts_count = self.verts_counts_acc_from_0[-1]
        self.faces_counts_max = max(self.faces_counts)

        self.mask_verts_not_padded = torch.ones(
            size=[len(self), self.verts_counts_max],
            dtype=torch.bool,
            device=self.device,
        )

        for i in range(len(self)):
            self.mask_verts_not_padded[i, self.verts_counts[i] :] = False

        import matplotlib.pyplot as plt

        color_ = plt.get_cmap("tab20", len(self))
        self.feats_rgb_object_id = []
        for i in range(len(self)):
            self.feats_rgb_object_id.extend([color_(i)] * self.verts_counts[i])

        self.update_verts_coarse()
        #### UPDATE MESHES x Gaussians
        # self.verts_gs_opacity = torch.nn.Parameter(torch.ones((self.verts_count, )).to(**factory_kwargs),
        #                                            requires_grad=self.gs_opacity_requires_grad)
        #
        # self.gs_top_k = self.gs_top_k
        # self.verts_gs_scale = torch.nn.Parameter(self.gs_scale * torch.ones((self.verts_count, 3)).to(**factory_kwargs),
        #                                          requires_grad=self.gs_scale_requires_grad)
        #
        # gs_rotation_init = torch.zeros((self.verts_count, 4)).to(**factory_kwargs)
        # gs_rotation_init[:, 0] = 1.
        # self.verts_gs_rotation = torch.nn.Parameter(gs_rotation_init,
        #                                             requires_grad=self.gs_rotation_requires_grad)

    def get_uniform_jittered_tets_verts(self, object_id):
        pts = self.tets_verts.clone()  #  self.get_verts_with_mesh_id(object_id)
        if self.mesh_update_jitter_scale > 0:
            jitter = (
                (torch.rand(3, device=pts.device) - 0.5)
                * self.tets_scale
                * self.mesh_update_jitter_scale
            )
            pts = pts + jitter
        return pts

    def get_rand_jittered_mesh_verts(self, object_id):
        pts = self.get_verts_with_mesh_id(object_id, clone=True).detach()
        if self.mesh_update_jitter_scale > 0:
            jitter = (
                (torch.rand_like(pts, device=pts.device) - 0.5)
                * self.tets_scale
                * self.mesh_update_jitter_scale
            )
            pts = pts + jitter
        return pts

    def get_sdf(self, pts, object_id):
        """
        Args:
            pts (torch.Tensor): BxNx3
        Returns:
            sdf (torch.Tensor): BxN
        """



        """***********************************************************************************************************************"""
         #Original
        #sdf_init = pts.detach().norm(dim=-1, keepdim=True) - self.init_radius   

        #@NOTE, this is another thing I tried, by trying to get an sdf with chair-like template that include holes as a starting point. 
        # Initialize SDF with chair-like template that includes holes, If  you want random initialization, use the commented code and comment this one.
        sdf_init = self._get_chair_template_sdf(pts)

        #sdf_init = torch.zeros_like(pts.detach().norm(dim=-1, keepdim=True ))

        # OR to random values
        #sdf_init = torch.rand_like(pts.detach().norm(dim=-1, keepdim=True)) * self.tets_scale

        from od3d.data.batch_datatypes import OD3D_ModelData

        sdf_delta = self.sdf_coordmlps[object_id](
            OD3D_ModelData(pts3d=pts[None,]),
        ).feat[0]
        sdf_vals = sdf_init + sdf_delta

        #@NOTE, This is where I implemented the denoising, basically just clamping the values within a certain range, I tried using 
        # guassian smmoothening via convolution, but produced same result. So I just left it like this
        # denoise the sdf, clamping the values with -tets_scale and tets_scale   
        sdf_vals = torch.clamp(sdf_vals, min=-self.tets_scale, max=self.tets_scale)
        
        # Safeguard: ensure there are always some negative values to guarantee a surface
        if sdf_vals.min() >= 0:
            center_mask = torch.norm(pts, dim=-1) < self.init_radius * 0.2
            sdf_vals = torch.where(center_mask, sdf_vals - 0.1, sdf_vals)
        
        return sdf_vals

    def _get_chair_template_sdf(self, pts):
        """
        Initialize SDF with chair-like template that includes holes
        """
        x, y, z = pts.unbind(-1)
        
        # a basic sphere sdf
        sphere_sdf = torch.sqrt(x**2 + y**2 + z**2) - self.init_radius * 0.5
        
        # for hole formation in the bottom center
        scale = self.init_radius * 0.3
        
        # Create a hole
        hole_center_dist = torch.sqrt(x**2 + z**2)
        hole_radius = 0.15 * scale
        hole_depth = 0.1 * scale
        
        # Bottom region mask
        bottom_mask = (y < -0.05 * scale) & (y > -0.25 * scale)
        
        # Strong hole carving - make SDF more positive in hole region
        hole_mask = bottom_mask & (hole_center_dist < hole_radius)
        hole_adjustment = torch.where(hole_mask, hole_depth, 0.0)
        
        # Gradual transition at hole edges
        edge_mask = bottom_mask & (hole_center_dist < hole_radius * 1.5) & (hole_center_dist >= hole_radius)
        edge_adjustment = torch.where(edge_mask, 
                                     hole_depth * 0.5 * (1.0 - (hole_center_dist - hole_radius) / (hole_radius * 0.5)), 
                                     0.0)
        
        # Apply hole modifications
        chair_sdf = sphere_sdf + hole_adjustment + edge_adjustment
        
        return chair_sdf.unsqueeze(-1)

    def get_feats(self, pts, object_id):
        from od3d.data.batch_datatypes import OD3D_ModelData

        feats = self.feat_coordmlps[object_id](OD3D_ModelData(pts3d=pts[None,])).feat[0]
        return feats

    def get_sdf_gradient(self, object_id):
        num_samples = 5000
        sample_points = (
            torch.rand(num_samples, 3, device=self.verts.device) - 0.5
        ) * self.tets_scale

        mesh_verts = self.get_rand_jittered_mesh_verts(object_id=object_id)

        rand_idx = torch.randperm(len(mesh_verts), device=mesh_verts.device)[:5000]
        mesh_verts = mesh_verts[rand_idx]
        sample_points = torch.cat([sample_points, mesh_verts], 0)
        sample_points.requires_grad = True
        y = self.get_sdf(pts=sample_points, object_id=object_id)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        try:
            gradients = torch.autograd.grad(
                outputs=[y],
                inputs=sample_points,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
        except RuntimeError:  # For validation, we have disabled gradient calculation.
            return torch.zeros_like(sample_points)
        return gradients

    def get_geo_sdf_reg_loss(self, objects_ids):
        regs_losses = []
        for object_id in objects_ids:
            regs_losses.append(
                (
                    (self.get_sdf_gradient(object_id=object_id).norm(dim=-1) - 1) ** 2
                ).mean(),
            )
        regs_losses = torch.stack(regs_losses)    
        return regs_losses



#@NOTE, added topological loss, but it didn't work, that was the last run with chair handles. 
# I will delete it, since I implement this to work explicitely for chair with holes, and it didn't work as expected.
    def get_topology_loss(self, objects_ids):
       
        topology_losses = []
        for object_id in objects_ids:
            # Sample points inside the expected hole region
            hole_region_pts = self._sample_chair_hole_region()
            
            # Get SDF values at these points
            sdf_vals = self.get_sdf(hole_region_pts, object_id)
            
            # Strong loss encourages positive SDF values (outside surface) in hole region
            # Use exponential loss for more aggressive hole formation
            hole_loss = torch.exp(-sdf_vals.clamp(min=0)).mean()
            
            # Also add a loss to maintain chair structure outside hole region
            structure_pts = self._sample_chair_structure_region()
            structure_sdf = self.get_sdf(structure_pts, object_id)
            structure_loss = torch.clamp(structure_sdf, min=0).mean()
            
            # Add direct hole-carving loss
            hole_carve_loss = self._get_hole_carving_loss(object_id)
            
            
            total_loss = 2.0 * hole_loss + 0.3 * structure_loss + hole_carve_loss
            topology_losses.append(total_loss)
        
        return torch.stack(topology_losses)
    
    def _get_hole_carving_loss(self, object_id):
       
        device = self.verts.device
        scale = self.init_radius * 0.3
        
        # Create a grid in the hole region
        n_grid = 20
        x_range = torch.linspace(-0.15 * scale, 0.15 * scale, n_grid, device=device)
        z_range = torch.linspace(-0.15 * scale, 0.15 * scale, n_grid, device=device)
        y_val = -0.1 * scale  # Fixed y in hole region
        
        x_grid, z_grid = torch.meshgrid(x_range, z_range, indexing='ij')
        x_flat = x_grid.flatten()
        z_flat = z_grid.flatten()
        y_flat = torch.full_like(x_flat, y_val)
        
        # Only keep points within the hole radius
        hole_radius = 0.15 * scale
        radii = torch.sqrt(x_flat**2 + z_flat**2)
        hole_mask = radii < hole_radius
        
        if hole_mask.sum() > 0:
            hole_pts = torch.stack([x_flat[hole_mask], y_flat[hole_mask], z_flat[hole_mask]], dim=-1)
            hole_pts = hole_pts.unsqueeze(0)  # Add batch dimension
            
            # Get SDF values and encourage them to be positive (outside surface)
            sdf_vals = self.get_sdf(hole_pts, object_id)
            
            
            hole_carve_loss = torch.mean(torch.clamp(-sdf_vals, min=0) ** 2)
            return hole_carve_loss
        else:
            return torch.tensor(0.0, device=device)
    
    def _sample_chair_hole_region(self):
        
        # Define hole region to match the SDF initialization
        n_samples = 1000
        device = self.verts.device
        
        # Sample in the bottom center region where we carved the hole
        scale = self.init_radius * 0.3
        hole_radius = 0.15 * scale
        
        # Sample in cylindrical region for hole
        theta = torch.rand(n_samples, device=device) * 2 * torch.pi
        r = torch.sqrt(torch.rand(n_samples, device=device)) * hole_radius
        
        x = r * torch.cos(theta)
        z = r * torch.sin(theta)
        
        # Sample in the bottom region where hole should be
        y = torch.rand(n_samples, device=device) * 0.2 * scale - 0.15 * scale
        
        return torch.stack([x, y, z], dim=-1)
    
    def _sample_chair_structure_region(self):
        
        n_samples = 1000
        device = self.verts.device
        
        # Sample points around the sphere structure, avoiding the hole region
        pts = torch.randn(n_samples, 3, device=device) * 0.3 * self.init_radius
        
        # Filter out points that are in the hole region
        x, y, z = pts.unbind(-1)
        hole_mask = (y < 0) & (torch.sqrt(x**2 + z**2) < 0.08 * self.init_radius)
        
        # Keep only points outside the hole region
        valid_mask = ~hole_mask
        if valid_mask.sum() > 0:
            pts = pts[valid_mask][:min(n_samples, valid_mask.sum())]
        
        return pts

    #
    # @property
    # def verts(self):
    #     return self.verts_params
    #
    # @verts.setter
    # def verts(self, value):
    #     self.verts_params = value
    #
    # def init_sdf(self):
    #     """
    #     """
    #     # calculate distance for each vertex of the tetrahedra to the meshes
    #     # 1. for each vertex, calculate the distance to the faces,
    #     # 2. for each vertex, calculate the distance to the vertices,
    #     # 3. take the minimum distance
    #     # 4. then check if inside for each vertex and determine the sign of the distance
    #
    #
    #     pass
    #
    # @classmethod
    # def read_from_ply_files(
    #     cls,
    #     fpaths_meshes: List[Path],
    #     fpaths_meshes_tforms: List[Path] = None,
    #     device=None,
    #     normalize_scale=False,
    #     **kwargs,
    # ):
    #     meshes = []
    #     for i, fpath_mesh in enumerate(fpaths_meshes):
    #         mesh = Mesh.load_from_file(fpath=fpath_mesh, device=device)
    #         if fpaths_meshes_tforms is not None and fpaths_meshes_tforms[i] is not None:
    #             mesh_tform = torch.load(fpaths_meshes_tforms[i]).to(device)
    #             mesh.verts = transf3d_broadcast(pts3d=mesh.verts, transf4x4=mesh_tform)
    #         if normalize_scale:
    #             mesh.verts /= mesh.verts.abs().max()
    #         meshes.append(mesh)
    #
    # def get_init_sdf(self, pts):
    #     """
    #         Args:
    #             pts (torch.Tensor): BxMxNx3
    #         Returns:
    #             sdf (torch.Tensor): BxMxN
    #     """
    #     init_radius = 1.
    #     sdf = init_radius - pts.norm(dim=-1, keepdim=True)  # init sdf is a sphere centered at origin
    #     return sdf
    #
    # def get_sdf(self, pts, objects_ids=None):
    #     """
    #     Args:
    #         pts (torch.Tensor): BxNx3
    #     Returns:
    #         sdf (torch.Tensor): BxN
    #     """
    #
    #
    #     B = pts.shape[0]
    #     if objects_ids is None and len(self) > 1:
    #         raise ValueError("objects_ids must be provided when there are multiple objects")
    #
    #     if objects_ids is None and len(self) == 1:
    #         objects_ids
    #
    #     sdf_vals = torch.zeros(pts.shape[:-1], device=pts.device)
    #     for m in range(self.meshes_count):
    #         sdf_vals[m] = self.sdf_mlps[m](pts[m])
    #
    #     return sdf_vals
    #
    # def update_mesh_with_sdf(self, objects_ids=None, jitter_grid=True):
    #     # Run DM tet to get a base mesh
    #     v_deformed = self.verts
    #
    #     if jitter_grid and self.mesh_update_jitter_grid > 0:
    #         jitter = (torch.rand(1, device=v_deformed.device) * 2 - 1) * self.mesh_update_jitter_grid * self.grid_scale
    #         v_deformed = v_deformed + jitter
    #
    #     self.current_sdf = self.get_sdf(v_deformed, total_iter=total_iter)
    #     verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, self.current_sdf, self.indices)
    #     self.mesh_verts = verts
    #     return mesh.make_mesh(verts[None], faces[None], uvs[None], uv_idx[None], material=None)
    #


"""
    def __init__(self, grid_res, scale, num_layers=None, hidden_size=None, embedder_freq=None, embed_concat_pts=True,
                 init_sdf=None, jitter_grid=0., symmetrize=False):
        super(DMTets, self).__init__()

        self.grid_res = grid_res
        self.marching_tets = DMTet()
        self.grid_scale = scale
        self.init_sdf = init_sdf
        self.jitter_grid = jitter_grid
        self.symmetrize = symmetrize
        self.load_tets(self.grid_res, self.grid_scale)

        embedder_scalar = 2 * np.pi / self.grid_scale * 0.9  # originally (-0.5*s, 0.5*s) rescale to (-pi, pi) * 0.9
        self.mlp = CoordMLP(
            3,
            1,
            num_layers,
            nf=hidden_size,
            dropout=0,
            activation=None,
            min_max=None,
            n_harmonic_functions=embedder_freq,
            embedder_scalar=embedder_scalar,
            embed_concat_pts=embed_concat_pts)

    def load_tets(self, grid_res=None, scale=None):
        if grid_res is None:
            grid_res = self.grid_res
        else:
            self.grid_res = grid_res
        if scale is None:
            scale = self.grid_scale
        else:
            self.grid_scale = scale
        tets = np.load('data/tets/{}_tets.npz'.format(grid_res))
        self.verts = torch.tensor(tets['vertices'], dtype=torch.float32,
                                  device='cuda') * scale  # verts original scale (-0.5, 0.5)
        self.indices = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        self.generate_edges()

    def get_sdf(self, pts=None, total_iter=0):
        if pts is None:
            pts = self.verts
        if self.symmetrize:
            xs, ys, zs = pts.unbind(-1)
            pts = torch.stack([xs.abs(), ys, zs], -1)  # mirror -x to +x
        sdf = self.mlp(pts)

        if self.init_sdf is None:
            pass
        elif type(self.init_sdf) in [float, int]:
            sdf = sdf + self.init_sdf
        elif self.init_sdf == 'sphere':
            init_radius = self.grid_scale * 0.25
            init_sdf = init_radius - pts.norm(dim=-1, keepdim=True)  # init sdf is a sphere centered at origin
            sdf = sdf + init_sdf
        elif self.init_sdf == 'ellipsoid':
            rxy = self.grid_scale * 0.15
            xs, ys, zs = pts.unbind(-1)
            init_sdf = rxy - torch.stack([xs, ys, zs / 2], -1).norm(dim=-1,
                                                                    keepdim=True)  # init sdf is approximately an ellipsoid centered at origin
            sdf = sdf + init_sdf
        else:
            raise NotImplementedError

        return sdf

    def get_sdf_gradient(self):
        num_samples = 5000
        sample_points = (torch.rand(num_samples, 3, device=self.verts.device) - 0.5) * self.grid_scale
        mesh_verts = self.mesh_verts.detach() + (torch.rand_like(self.mesh_verts) - 0.5) * 0.1 * self.grid_scale
        rand_idx = torch.randperm(len(mesh_verts), device=mesh_verts.device)[:5000]
        mesh_verts = mesh_verts[rand_idx]
        sample_points = torch.cat([sample_points, mesh_verts], 0)
        sample_points.requires_grad = True
        y = self.get_sdf(pts=sample_points)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        try:
            gradients = torch.autograd.grad(
                outputs=[y],
                inputs=sample_points,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        except RuntimeError:  # For validation, we have disabled gradient calculation.
            return torch.zeros_like(sample_points)
        return gradients

    def get_sdf_reg_loss(self):
        reg_loss = ((self.get_sdf_gradient().norm(dim=-1) - 1) ** 2).mean()
        return reg_loss

    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    def getMesh(self, material=None, total_iter=0, jitter_grid=True):
        # Run DM tet to get a base mesh
        v_deformed = self.verts

        # if self.FLAGS.deform_grid:
        #     v_deformed = self.verts + 2 / (self.grid_res * 2) * torch.tanh(self.deform)
        # else:
        #     v_deformed = self.verts
        if jitter_grid and self.jitter_grid > 0:
            jitter = (torch.rand(1, device=v_deformed.device) * 2 - 1) * self.jitter_grid * self.grid_scale
            v_deformed = v_deformed + jitter

        self.current_sdf = self.get_sdf(v_deformed, total_iter=total_iter)
        verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, self.current_sdf, self.indices)
        self.mesh_verts = verts
        return mesh.make_mesh(verts[None], faces[None], uvs[None], uv_idx[None], material)

"""

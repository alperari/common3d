## [Common3D: Self-Supervised Learning of 3D Morphable Models for Common Objects in Neural Feature Space [CVPR'25]](https://genintel.github.io/common3d)

```commandline
@InProceedings{Sommer_2025_CVPR,
    author    = {Sommer, Leonhard and D\"unkel, Olaf and Theobalt, Christian and Kortylewski, Adam},
    title     = {Common3D: Self-Supervised Learning of 3D Morphable Models for Common Objects in Neural Feature Space},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {6468-6479}
}
```

### Code Setup
- Setup Local
  - `bash setup/shuri.sh`
  - `config/platform/local.yaml` from `config/platform/slurm.yaml`
  - `wandb init`
    - verify `od3d debug hello-world`

- Setup Slurm
  - `config/platform/slurm.yaml`
  - `~/.ssh/config` with `config/platform/ssh-config-template`
  - `od3d platform setup -p slurm` # set install_od3d=true in slurm.yaml

### Dataset Setup

  1) Download
     - `od3d dataset setup -d [co3d|pascal3d|objectnet3d]`
  2) Extract Meta Data per Frame/Sequence (e.g. camera, quality, etc.)
     - `od3d dataset extract-meta -d [co3d|pascal3d|objectnet3d]`
  3) Preprocess (e.g. point cloud, mesh, masks, etc.)
     - `od3d dataset preprocess -d [co3d|pascal3d|objectnet3d]`

  - Visualize
    - `od3d dataset visualize -d [co3d|pascal3d|objectnet3d]`

### Benchmark

To evaluate the method run

  - `od3d bench run -b co3d_common3d -a co3d_refs -p slurm`
  - `od3d bench run -b co3d_common3d -a categories/cross/all,co3d_refs -p slurm`.

To see the current status on slurm use

  - `od3d bench status-slurm`.

To stop a job running on slurm use

  - `od3d bench stop-slurm -j <job-name>`.

### Media
`od3d dataset save-sequences-as-video -d co3d_no_zsp_aligned_visual`
`od3d dataset visualize-category-sequences -d co3d_no_zsp_aligned_visual`
`od3d dataset visualize-category-meshes -d co3d_no_zsp_aligned_visual`
`od3d dataset visualize-category-pcls -d co3d_no_zsp_aligned_visual`

### Tables
`od3d table multiple -b co3d_common3d -a co3d_refs -m test/pascal3d_test/pose/acc_pi6 -l 24`

### Documentation
- [Coordinate Frames](docs/coordinate_frames/README.md)
- [Contributing](docs/contributing/README.md)

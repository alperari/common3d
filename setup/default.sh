#!/bin/bash

# copy ssh tempalte

#wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
#sudo sh cuda_12.4.0_550.54.14_linux.run

#CUDA_HOME={cfg.platform.path_cuda}
#PATH_OD3D={cfg.platform.path_od3d}

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python3, python3-dev, python3-venv

#sudo add-apt-repository ppa:deadsnakes/ppa
#sudo apt update
#sudo apt-get install python3.11, python3.11-venv
#alias python3=python3.11

sudo apt install nvidia-driver-570 nvidia-dkms-570
sudo apt install libgtk2.0-dev pkg-config # for cv2

CUDA_HOME=/usr/local/cuda-12.2
CUDA_HOME=/usr/local/cuda-12.4
PATH_OD3D=/home/sommerl/PycharmProjects/od3d
CUDA_VERSION=$(basename "${CUDA_HOME}")

# ${{PATH_OD3D}}
# PATH=${{CUDA_HOME}}/bin:${{PATH}}
# LD_LIBRARY_PATH=${{CUDA_HOME}}/lib64:${{LD_LIBRARY_PATH}}
# export PATH
# export LD_LIBRARY_PATH
export CUDA_HOME
export CPATH=$CPATH:${CUDA_HOME}/targets/x86_64-linux/include # pycuda requires this
export LIBRARY_PATH=$LIBRARY_PATH:${CUDA_HOME}/targets/x86_64-linux/lib # pycuda requires this
export CC=/usr/bin/gcc-13 # only required for ubuntu 24
export CXX=/usr/bin/g++-13 # only required for ubuntu 24

# echo PATH=${{PATH}}
# echo LD_LIBRARY_PATH=${{LD_LIBRARY_PATH}}
echo CUDA_HOME=${CUDA_HOME}

git pull

git submodule init
git submodule update
git submodule foreach 'git fetch origin; git checkout $(git rev-parse --abbrev-ref HEAD); git reset --hard origin/$(git rev-parse --abbrev-ref HEAD); git submodule update --recursive; git clean -dfx'
# git submodule update --init --recursive


# Install OD3D in venv
VENV_NAME="venv_od3d_${CUDA_VERSION}"
export VENV_NAME
if [[ -d "${VENV_NAME}" ]]; then
    echo "Venv already exists at ${PATH_OD3D}/${VENV_NAME}."
    source ${PATH_OD3D}/${VENV_NAME}/bin/activate
else
    echo "Creating venv at ${PATH_OD3D}/${VENV_NAME}."
    python3 -m venv ${PATH_OD3D}/${VENV_NAME}
    source ${PATH_OD3D}/${VENV_NAME}/bin/activate
fi

pip install pip --upgrade
pip install wheel

if [[ "${CUDA_HOME}" == *"12.4"* ]]; then
    echo "installing for CUDA 12.4"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install pytorch3d==0.7.8+pt2.6.0cu124 --extra-index-url https://miropsota.github.io/torch_packages_builder
    pip install torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
    pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html # needs to remain 2.5.1

    # torch 2.5.1
    #pip install pytorch3d==0.7.8+pt2.5.1cu124 --extra-index-url https://miropsota.github.io/torch_packages_builder
    #pip install torch_cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
    #pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html

elif [[ "${CUDA_HOME}" == *"12.2"* ]]; then
    echo "installing for CUDA 12.2" # 12.2
    echo "doesnt work cause of flash attn"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install pytorch3d==0.7.8+pt2.5.1cu121 --extra-index-url https://miropsota.github.io/torch_packages_builder
    pip install torch_cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
    pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html
else
    echo "installing for CUDA 11"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    pip install pytorch3d@git+https://github.com/facebookresearch/pytorch3d@stable
    # pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt1131/download.html
    pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
    pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu117.html
fi

pip install -e .

# TRELLIS extensions
#cd third_party/TRELLIS
##git submodule init
#git submodule update
#pip install psutil
#pip install flash_attn
#pip install easydict rembg onnxruntime
#pip install onnxruntime --upgrade
#pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
#pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh xatlas pyvista pymeshfix igraph transformers
#pip install spconv-cu118 # despite cuda 12
#pip install diff-gaussian-rasterization@third_party/mip-splatting/submodules/diff-gaussian-rasterization
# /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/
#ln -s $(pwd)/third_party/TRELLIS/trellis $(pwd)/src/trellis

# HUNYUAN3D-2 extensions
#pip install git+https://github.com/Tencent/Hunyuan3D-2.git
#pip install third_party/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
#pip install third_party/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer # mesh_processor

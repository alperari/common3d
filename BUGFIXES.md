


# BUG 1
### Symptom
    cv2.imshow(...) freeze
### Cause
    python packages av and opencv-python clash (imshow freezes if av is installed)
### Solution
    pip uninstall av


# BUG 2
### Symptom
    RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the ‘spawn’ start method
### Cause
    Sharing GPU memory across multiprocesses
### Solution
    Don't use GPU with dataloader


# BUG 3

### Symptom (installing pytorch3d)
    build_ext
        error: [Errno 2] No such file or directory: '/usr/local/cuda/bin/nvcc'

### Cause
    nvcc not available in nvidia runtime image
### Solution
    use nvidia devel image, e.g. nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04


# BUG 4

[Open3D WARNING] GLFW Error: X11: The DISPLAY environment variable is missing
[Open3D WARNING] Failed to initialize GLFW
Segmentation fault (core dumped)

export DISPLAY=:0.0;

# BUG 5

end of file reading mesh

/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/mesh/alpha500/remote/107_12754_22765/mesh.ply


/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/mesh/alpha500/remote/107_12754_22765/mesh.ply
107_12754_22765
105_12562_23650

# CUDNN=cudnn-linux-x86_64-8.9.5.29_cuda11-archive
#cp ${CUDNN}/include/cudnn*.h ${CUDA_HOME}/include
#cp ${CUDNN}/lib/libcudnn* ${CUDA_HOME}/lib64
#chmod a+r ${CUDA_HOME}/include/cudnn*.h ${CUDA_HOME}/lib64/libcudnn*

# BUG 6

wrong cuda device

CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER
if CUDA_DEVICE_ORDER is not set,


#  BUG 7
py38_cu113_pyt1110
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
py310_cu117_pyt1131
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt1131/download.html

pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.5.1cu121
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.5.1cu124

pip install pytorch3d==0.7.8+pt2.5.1cu124 -f https://miropsota.github.io/torch_packages_builder

# BUG 8
torch cluster wrong CUDA version
import torch
print(torch.__version__)
print(torch.version.cuda)

pip install torch_cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install torch_cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html


# BUG 9
ssh-agent hijacked by gnome keyring

ssh-agent
problem: SSH_AUTH_SOCK=/run/user/19104/keyring/ssh


prevent gnome-keyring from being locked which disable ssh-auth-socket:
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

nano ~/.config/autostart/gnome-keyring-ssh.desktop
[Desktop Entry]
Type=Application
Name=Disable GNOME Keyring SSH
Exec=sh -c "true"
X-GNOME-Autostart-enabled=false

nano ~/.bashrc
if ! pgrep -u "$USER" ssh-agent > /dev/null; then
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519
fi


not tested:
nano ~/.ssh/config
IdentityFile ~/.ssh/id_ed25519


problem identified: kerberos new ticket after 10 hours required


# BUG 10

pip install imageio-ffmpeg
pip install moviepy


# BUG 11

NO SPACE LEFT ON DEVICE

tail -n 100 /var/log/auth.log

/var/log/syslog

sudo su
echo "" > /var/log/kern.log
echo "" > /var/log/syslog
service syslog restart
journalctl --vacuum-size=50M

/var/log/auth.logk

sudo su
echo "" > /var/log/auth.log
service syslog restart
journalctl --vacuum-size=50M


docker eating up all space
docker system df

/var/lib/docker
containers:
docker rm -v -f $(docker ps -qa)
images:
docker image remove -f $(sudo docker images -a -q)

docker container prune
docker image prune

docker volume prune

/var/lib/docker/overlay2
docker system prune



/var/log/auth.log

2025-03-01T11:10:12.552129+01:00 roycoffee gnome-keyring-daemon[3019]: GLib-GIO: fail: Error accepting connection: Too many open files
2025-03-01T11:10:12.552172+01:00 roycoffee gnome-keyring-daemon[3019]: GLib-GIO: fail: Error accepting connection: Too many open files
2025-03-01T11:10:12.552215+01:00 roycoffee gnome-keyring-daemon[3019]: GLib-GIO: fail: Error accepting connection: Too many open files
2025-03-01T11:10:12.552258+01:00 roycoffee gnome-keyring-daemon[3019]: GLib-GIO: fail: Error accepting connection: Too many open files
2025-03-01T11:10:12.552301+01:00 roycoffee gnome-keyring-daemon[3019]: GLib-GIO: fail: Error accepting connection: Too many open files


#  BUG 12 EGL/OPENGL MISSING

In file included from /home/sommerl/PycharmProjects/od3d/venv_od3d_cuda-12.4/lib/python3.12/site-packages/nvdiffrast/common/glutil.cpp:14:
/home/sommerl/PycharmProjects/od3d/venv_od3d_cuda-12.4/lib/python3.12/site-packages/nvdiffrast/common/glutil.h:36:10: fatal error: EGL/egl.h: No such file or directory
   36 | #include <EGL/egl.h>

sudo apt install freeglut3-dev

(egl wayland useful? egl-wayland package in Ubuntu)

# BUG 13 OpenEXR codec disabled for CV

img = cv2.imread(str(self.fpath_mask), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
cv2.error: OpenCV(4.11.0) /io/opencv/modules/imgcodecs/src/grfmt_exr.cpp:103: error: (-213:The function/feature is not implemented) imgcodecs: OpenEXR codec is disabled. You can enable it via 'OPENCV_IO_ENABLE_OPENEXR' option. Refer for details and cautions here: https://github.com/opencv/opencv/issues/21326 in function 'initOpenEXR'


sudo apt-get install openexr
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# BUG 14 Cython speed up does not work

warning: failed to delete old cython speedups. Please delete all *.so files from the directories

in registry: ctrl+alt+shift+'/' -> registry :
disable
python.debug.enable.cython.speedups

# BUG 15 Not enough SWAP MEMORY
swapon --show
/dev/nvme0n1p4
# 1MB * 1K= 1GB, 1MB * 30K = 30GB
sudo dd if=/dev/nvme0n1p3 of=/swapfile count=30K bs=1M
sudo mkswap /swapfile
sudo chown root:root /swapfile
sudo chmod 600 /swapfile
sudo swapon /swapfile

sudo swapoff /swapfile
sudo dd if=/dev/nvme0n1p3 of=/swapfile bs=1M count=1024 oflag=append conv=notrunc
sudo mkswap /swapfile
sudo swapon /swapfile

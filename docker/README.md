# Setup Docker without ROOT and with GPU

```
sudo usermod -aG docker ${USER}

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

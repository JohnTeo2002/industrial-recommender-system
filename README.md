## Quick Start

### Installation

### Install Docker engine

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [signed-by=/usr/share/keyrings/docker.asc] https://download.docker.com/linux/ubuntu stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### Install the NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Inject the runtime into the Docker configuration mapping and restart the engine
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Scaffold Your Production Repository Matrix

```bash
mkdir -p industrial-recommender-system/{config,data/{raw/train,raw/valid},src,feature_store,triton_serving,notebooks}
cd industrial-recommender-system
```

### Run the Containerized Environment Live

```bash
sudo docker run -it --gpus all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8888:8888 \
  -v $(pwd):/workspace/project \
  --ipc=host \
  nvcr.io/nvidia/merlin/merlin-tensorflow:nightly /bin/bash
```

### Execute the initialization script to populate the storage pipeline with 100k eCommerce actions using Merlin's synthetic transactional engine
```bash
cd /workspace/project
```

```python
python src/generate_data.py
```

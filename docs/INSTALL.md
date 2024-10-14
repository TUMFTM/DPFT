# Installation
Instructions on how to install the required environment to run the DPFT model. The recommended installation method is the docker installation to ensure a consistent environment across different machines and operating systems. However, instructions to install the environment locally are also provided.

## 🐋 docker (recommended)
Instructions on how to install the required environment in a docker container.

#### 1. Install docker - see [here](https://docs.docker.com/engine/install/ubuntu/) (tested with 25.0.3)
Docker needs to be installed on your system to run the recommended installation.

```
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

```
 sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

#### 2. Install nvidia container toolkit - see [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (tested with 1.14.5)
To build and run the model inside a docker container, the GPU must be accessible from inside the docker container. Therefore, the NVIDIA Container Toolkit is required.
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```
sudo apt-get update
```

```
sudo apt-get install -y nvidia-container-toolkit
```

```
sudo nvidia-ctk runtime configure --runtime=docker
```

```
sudo systemctl restart docker
```

#### 3. Build docker image
First, you have to build the docker image from the provided Dockerfile.
```
docker build -t dpft:0.0.1 .
```

#### 4. Run docker container
After that, you can create a docker container from this image and mount the required directories.
```
docker run \
    --name dpft \
    -it \
    --gpus all \
    -e DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v <path to repository>:/app \
    -v <path to data>:/data \
    dpft:0.0.1 bash
```


## 💻 local
Instructions on how to install the required environment locally.

#### 1. Install prerequisites
First, you should ensure that the following packages are installed on your system.
```
sudo apt install -y \
    build-essential \
    libgl1 \
    libegl1 \
    libx11-dev \
    libglib2.0-0 \
    libgomp1 \
    libxcb-xinerama0-dev \
    ninja-build \
    python-is-python3 \
    wget \
    unzip
```

#### 2. Install nvidia driver - see [here](https://ubuntu.com/server/docs/nvidia-drivers-installationv) (tested with 560.35.03)
Please note that the DPFT model needs a GPU and the associated drivers to run. Make sure that you have Nvidia drivers installed on your system.
```
sudo ubuntu-drivers install
```

#### 3. Install CUDA - see [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu) (tested with 11.7)
The build process and model execution rely on CUDA to run. Therefore, ensure that CUDA is installed on your system.
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```

#### 4. Install Pytorch - see [here](https://pytorch.org/get-started/previous-versions/) (tested with 1.13)
Pytorch is used as model framework and has to be installed on your system.
```
python -m pip install torch==1.cd 13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

#### 5. Install MultiScaleDeformableAttention - see [here](https://github.com/fundamentalvision/Deformable-DETR) 
The sensor fusion uses deformable attention, so this needs to be installed to execute the model.
```
wget https://github.com/fundamentalvision/Deformable-DETR/archive/main.zip
unzip main.zip
cd Deformable-DETR-main/models/ops/
sh ./make.sh
python -m pip install .
```

#### 6. Install DPFT
Finally, the DPFT project can be installed as follows.
```
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r ./requirements.txt
python -m pip install -e .
```

## <span style="color:red">⁉️</span> FAQ
<details>
<summary>No CUDA runtime is found</summary>

1. Install nvidia-container-runtime
    ```
    sudo apt-get install nvidia-container-runtime
    ```

2. Edit/create the /etc/docker/daemon.json with content:
    ```
    {
        "runtimes": {
            "nvidia": {
                "path": "/usr/bin/nvidia-container-runtime",
                "runtimeArgs": []
            } 
        },
        "default-runtime": "nvidia" 
    }
    ```

3. Restart docker daemon:
    ```
    sudo systemctl restart docker
    ```

    Reference: https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime


    If this doesn't solve the problem, try:
    ```
    DOCKER_BUILDKIT=0 docker build -t dprt:0.0.1 .
    ```

    Reference: https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime

</details>

<details>
<summary>fatal error: cusolverDn.h: No such file or directory</summary>

1. Export CUDA path:
    ```
    export PATH=/usr/local/cuda/bin:$PATH
    ```

    Reference: https://github.com/microsoft/DeepSpeed/issues/2684

</details>

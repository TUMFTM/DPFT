[![Linux](https://img.shields.io/badge/os-linux-blue.svg)](https://www.linux.org/)
[![Docker](https://badgen.net/badge/icon/docker?icon=docker&label)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/downloads/)

# Dual Perspective Fusion Transformer
![](/docs/figs/DPFT_Model_Overview.svg "DPFT Model Overview")


## Prerequisites
- ubuntu
- docker
- nvidia-docker

## Dataset
1. Get the dataset from https://github.com/kaist-avelab/K-Radar
2. Structure the dataset according to

    <details>
    <summary>K-Radar Data Structure</summary>

    ```
    .
    |
    +---data/
    |   |
    |   +---kradar/
    |   |   |
    |   |   +---raw/
    |   |   |   |
    |   |   |   +---1/
    |   |   |   |   |
    |   |   |   |   +---cam-front/
    |   |   |   |   |
    |   |   |   |   +---cam-left/
    |   |   |   |   |
    |   |   |   |   +---cam-rear/
    |   |   |   |   |
    |   |   |   |   +---cam-right/
    |   |   |   |   |
    |   |   |   |   +---description.txt
    |   |   |   |   |
    |   |   |   |   +---info_calib/
    |   |   |   |   |
    |   |   |   |   +---info_frames/
    |   |   |   |   |
    |   |   |   |   +---info_label/
    |   |   |   |   |
    |   |   |   |   +---info_label_v2/
    |   |   |   |   |
    |   |   |   |   +---info_matching/
    |   |   |   |   |
    |   |   |   |   +---os1-128/
    |   |   |   |   |
    |   |   |   |   +---os2-64/
    |   |   |   |   |
    |   |   |   |   +---radar_tesseract/
    |   |   |   |   |
    |   |   |   |   +---...
    |   |   |   |
    |   |   |   +---2, 3... 

    ```

    </details>

## Install
```
docker build -t dprt:0.0.1 .
```

## Run
```
docker run \
    --name dprt \
    -it \
    --gpus all \
    -e DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v <path to repository>:/app \
    -v <path to data>:/data \
    dprt:0.0.1 bash
```

## Prepare
```
python -m dprt.prepare --src /data/kradar/raw/ --cfg /app/config/kradar.json --dst /data/kradar/processed/
```

## Train
```
python -m dprt.train --src /data/kradar/processed/ --cfg /app/config/kradar.json
```

## Evaluate
```
python -m dprt.evaluate --src /data/kradar/processed/ --cfg /app/config/kradar.json --checkpoint /app/log/<path to checkpoint>
```

## Citation
If DPFT is useful or relevant to your research, please kindly recognize our contributions by citing our paper:
```bibtex
@article{fent2024dpft,
    title={DPFT: Dual Perspective Fusion Transformer for Camera-Radar-based Object Detection}, 
    author={Felix Fent and Andras Palffy and Holger Caesar},
    journal={arXiv preprint arXiv:2404.03015},
    year={2024}
}
```

## FAQ
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

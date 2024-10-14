<div align="center">

<h1>DPFT</h1>

Dual Perspective Fusion Transformer

[![Linux](https://img.shields.io/badge/os-linux-blue.svg)](https://www.linux.org/)
[![Docker](https://badgen.net/badge/icon/docker?icon=docker&label)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-blue.svg)](https://arxiv.org/abs/2404.03015)

![](/docs/figs/DPFT_Opening_Figure.jpg "DPFT Opening Figure")
</div>

## 📄 Overview
The perception of autonomous vehicles has to be efficient, robust, and cost-effective. However, cameras are not robust against severe weather conditions, lidar sensors are expensive, and the performance of radar-based perception is still inferior to the others. Camera-radar fusion methods have been proposed to address this issue, but these are constrained by the typical sparsity of radar point clouds and often designed for radars without elevation information. We propose a novel camera-radar fusion approach called Dual Perspective Fusion Transformer (DPFT), designed to overcome these limitations. Our method leverages lower-level radar data (the radar cube) instead of the processed point clouds to preserve as much information as possible and employs projections in both the camera and ground planes to effectively use radars with elevation information and simplify the fusion with camera data. As a result, DPFT has demonstrated state-of-the-art performance on the K-Radar dataset while showing remarkable robustness against adverse weather conditions and maintaining a low inference time.

## 🏆 Results
#### 3D object detection on the K-Radar dataset

| Model | Modality | Total | Normal | Overcast | Fog  | Rain | Sleet | LightSnow | HeavySnow | Revision |
|-------|----------|-------|--------|----------|------|------|-------|-----------|-----------|----------|
| DPFT  | C + R    | 56.1  | 55.7   | 59.4     | 63.1 | 49.0 | 51.6  | 50.5      | 50.5      | v1.0     |
| DPFT  | C + R    | 50.5  | 51.1   | 45.2     | 64.2 | 39.9 | 42.9  | 42.4      | 51.1      | v2.0     |



## 💿 Dataset
This project is based on the [K-Radar](https://github.com/kaist-avelab/K-Radar) dataset. To set it up correctly, you should follow these two steps:

1. Get the dataset from https://github.com/kaist-avelab/K-Radar
2. Structure the dataset accordly

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

## 💾 Install
We recommend using a docker based installation to ensure a consistent development environment but also provide instructions for a local installation. Therefore, check our more detailed [installation instructions](/docs/INSTALL.md)

```
docker build -t dprt:0.0.1 .
```

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

## 🔨 Usage
The usage of our model consists of three major steps.

### 1. Prepare
First, you have to prepare the training and evaluation data by pre-processing the raw dataset. This will not only deduce the essential information from the original dataset but also reduces the data size from 16 TB to only 670 GB.
```
python -m dprt.prepare --src /data/kradar/raw/ --cfg /app/config/kradar.json --dst /data/kradar/processed/
```

```
python -m dprt.prepare 
  --src <Path to the raw dataset folder>
  --cfg <Path to the configuration file>
  --dst <Path to save the processed dataset>
```

### 2. Train
Second, train the DPFT model on the previously prepared data or continue with a specific model training.
```
python -m dprt.train --src /data/kradar/processed/ --cfg /app/config/kradar.json
```

```
python -m dprt.train
  --src <Path to the processed dataset folder>
  --cfg <Path to the configuration file>
  --dst <Path to save the training log>
  --checkpoint <Path to a model checkpoint to resume training from>
```

### 3. Evaluate
Third, evaluate the model performance of a previously trained model checkpoint.
```
python -m dprt.evaluate --src /data/kradar/processed/ --cfg /app/config/kradar.json --checkpoint /app/log/<path to checkpoint>
```

```
python -m dprt.evaluate 
  --src <Path to the processed dataset folder>
  --cfg <Path to the configuration file>
  --dst <Path to save the evaluation log>
  --checkpoint <Path to the model checkpoint to evaluate>
```

## 📃 Citation
If DPFT is useful or relevant to your research, please kindly recognize our contributions by citing our paper:
```bibtex
@article{fent2024dpft,
    title={DPFT: Dual Perspective Fusion Transformer for Camera-Radar-based Object Detection}, 
    author={Felix Fent and Andras Palffy and Holger Caesar},
    journal={arXiv preprint arXiv:2404.03015},
    year={2024}
}
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

FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Define build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG DEBCONF_NOWARNINGS="yes"
ARG PIP_ROOT_USER_ACTION=ignore
ARG WORKDIR=/app

# Set environment variables
ENV PATH=/usr/local/cuda/bin:$PATH

# Set working directroy
WORKDIR $WORKDIR

# install packages
RUN apt-get update && apt-get install -y -q --no-install-recommends \
    build-essential \
    vim \
    git \
    libgl1 \
    libegl1 \
    libx11-dev \
    libglib2.0-0 \
    libgomp1 \
    libxcb-xinerama0-dev \
    ninja-build \
    qt5-default \
    wget \
    unzip \
 && rm -rf /var/lib/apt/lists/*

# Build and install MultiScaleDeformableAttention
RUN wget https://github.com/fundamentalvision/Deformable-DETR/archive/main.zip && \
    unzip main.zip && \
    cd Deformable-DETR-main/models/ops/ && \
    sh ./make.sh && \
    python -m pip install . && \
    cd $WORKDIR \
 && rm -rf Deformable-DETR-main main.zip

# Install python packages
COPY . $WORKDIR
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r ./requirements.txt && \
    python -m pip install -e . \
 && rm -rf ~/.cache/pip/*

CMD ["bash"]

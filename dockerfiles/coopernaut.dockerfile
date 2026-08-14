FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    vim \
    tmux \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -O /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

ENV PATH=/opt/conda/bin:$PATH

RUN conda create -y -n coopernaut python=3.7

# Use the environment by default
ENV PATH=/opt/conda/envs/coopernaut/bin:$PATH

RUN pip install --upgrade pip setuptools wheel

# PyTorch CUDA 11.0 build
RUN pip install \
    torch==1.7.1+cu110 \
    torchvision==0.8.2+cu110 \
    torchaudio==0.7.2 \
    -f https://download.pytorch.org/whl/torch_stable.html


WORKDIR /workspace

# Install Coopernaut dependencies only
COPY AutoCastSim/requirements.txt /AutoCastSim/requirements.txt

WORKDIR /AutoCastSim
RUN pip install -r requirements.txt
RUN pip install open3d

# CARLA python API
RUN pip install carla==0.9.15

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

WORKDIR /workspace/coopernaut

CMD ["/bin/bash"]
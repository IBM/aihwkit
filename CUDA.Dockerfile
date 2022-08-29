# Build argumnets
ARG CUDA_VER=11.7
ARG UBUNTU_VER=22.04

# Download the base image
FROM nvidia/cuda:${CUDA_VER}.0-devel-ubuntu${UBUNTU_VER}
# you can check for all available images at https://hub.docker.com/r/nvidia/cuda/tags

# Install as root
USER root

# Install dependencies
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install --yes \
    --no-install-recommends \
    bash \
    bash-completion \
    cmake \
    curl \
    git \
    libopenblas-dev \
    linux-headers-$(uname -r) \
    nano \
    python3 python3-dev python3-pip python-is-python3 \
    sudo \
    wget

RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Add a user `${USERNAME}` so that you're not developing as the `root` user
ENV USERNAME=ibm
RUN echo "${USERNAME}"
RUN useradd ${USERNAME} \
    --create-home \
    --shell=/bin/bash \
    --user-group && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd

# Change to your user
USER ${USERNAME}
WORKDIR /home/${USERNAME}

# Install python packages as your user
RUN pip install --upgrade pip
RUN pip install pybind11 scikit-build
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Set path of python packages
RUN echo 'export PATH=$HOME/.local/bin:$PATH' >> /home/${USERNAME}/.bashrc

# clone the repo and change to the source directory
RUN git clone https://github.com/IBM/aihwkit.git
WORKDIR /home/${USERNAME}/aihwkit

# Default value for NVIDIA RTX A5000, find your own GPU model and replace it
# use the k: https://developer.nvidia.com/cuda-gpus
ARG CUDA_ARCH=86
ARG NUM_CPU_THREADS=8
RUN echo "Detected CUDA_ARCHITECTURE is = ${CUDA_ARCH}"
# Build and install IBM aihwkit
RUN pip install . --install-option="-DUSE_CUDA=ON" --install-option="-DRPU_CUDA_ARCHITECTURES="${CUDA_ARCH}"" --install-option="-DRPU_BLAS=OpenBLAS"

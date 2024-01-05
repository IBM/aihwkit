# Stage 0: Intel MKL
ARG CUDA_VERSION=12.2.2
FROM intel/oneapi-basekit AS mkl-env

# Stage 1: Build dependencies
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS build-env

# Install as root
USER root

# Install dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install --yes \
    --no-install-recommends \
    cmake \
    git \
    linux-headers-$(uname -r) \
    python3 python3-dev python3-pip python-is-python3 && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python build dependencies
RUN pip install --no-cache-dir --upgrade --no-warn-script-location pip && \
    pip install --no-cache-dir --no-warn-script-location pybind11 scikit-build protobuf mypy

ARG USERNAME=coder
ARG USERID=1000
ARG GROUPID=1000

# Add user
RUN groupadd -g ${GROUPID} ${USERNAME} && \
    useradd ${USERNAME} \
    --create-home \
    --uid ${USERID} \
    --gid ${GROUPID} \
    --shell=/bin/bash

# Copy MKL libraries from mkl-env
COPY --from=mkl-env /opt/intel/oneapi/mkl /opt/intel/oneapi/mkl

# Copy IBM aihwkit and build
COPY . /aihwkit
RUN chown -R ${USERNAME}:${USERNAME} /aihwkit

# Change to your user
USER ${USERNAME}

# Install PyTorch
RUN pip install --no-cache-dir --no-warn-script-location torch torchvision

# Build aihwkit
WORKDIR /aihwkit
ENV MKLROOT /opt/intel/oneapi/mkl/latest
ENV CUDACXX /usr/local/cuda/bin/nvcc
ARG CUDA_ARCH=86
RUN python setup.py install --user -j$(nproc) \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
    -DRPU_BLAS=MKL \
    -DINTEL_MKL_DIR=${MKLROOT} \
    -DUSE_CUDA=ON \
    -DRPU_CUDA_ARCHITECTURES=${CUDA_ARCH}

# Stage 2: Final runtime environment
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04

# Copy from build-env
COPY --from=build-env /opt/intel/oneapi/mkl/latest/lib/intel64 /opt/intel/oneapi/mkl/latest/lib/intel64

# Install as root
USER root

# Install utilities
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install --yes \
    --no-install-recommends \
    bash-completion \
    curl \
    git \
    nano \
    python3 python3-pip python-is-python3 \
    sudo \
    vim \
    wget && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create a user
ARG USERNAME=coder
ARG USERID=1000
ARG GROUPID=1000
RUN groupadd -g ${GROUPID} ${USERNAME} && \
    useradd ${USERNAME} \
    --create-home \
    --uid ${USERID} \
    --gid ${GROUPID} \
    --shell=/bin/bash && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd

# Copy aihwkit and .local from build-env
COPY --from=build-env /home/${USERNAME}/.local /home/${USERNAME}/.local

# Install Python packages
RUN pip install --no-cache-dir --upgrade --no-warn-script-location pip

# Install Python packages
RUN pip install --no-cache-dir --no-warn-script-location \
    matplotlib

# Environment variables
ENV PATH /home/${USERNAME}/.local/bin:$PATH
ENV LD_LIBRARY_PATH /opt/intel/oneapi/mkl/latest/lib/intel64:${LD_LIBRARY_PATH}

# Switch to your user
USER ${USERNAME}

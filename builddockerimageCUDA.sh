#!/bin/bash

############################################
## Shell script to build Docker IBM aihwkit 
## CUDA enabled docker image for your GPU
############################################

# Get CUDA_ARCH for your GPU from `nvidia-smi`
CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv | sed -n '2 p' | tr -d '.')

# Get number of cpu threads for build
NUM_CPU_THREADS=$(nproc)

echo "Detected CUDA_ARCH is $CUDA_ARCH"
echo "Detected NUM_CPU_THREADS is $NUM_CPU_THREADS"

# Build Container
docker build \
--tag aihwkit:$CUDA_ARCH-cuda$CUDA_VER-ubuntu$UBUNTU_VER \
--build-arg CUDA_ARCH=$CUDA_ARCH \
--build-arg CUDA_VER=11.7 \
--build-arg UBUNTU_VER=22.04 \
--build-arg NUM_CPU_THREADS=$NUM_CPU_THREADS \
--file CUDA.Dockerfile .

echo "Docker image aihwkit:$CUDA_ARCH-cuda$CUDA_VER-ubuntu$UBUNTU_VER is built successfully"
#!/bin/bash
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.."
source scripts/activate.sh

python setup.py build_ext \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
    --inplace \
    -DRPU_BLAS=OpenBLAS \
    -j16 \
    -DUSE_CUDA=ON \
    -DRPU_CUDA_ARCHITECTURES=75

#!/bin/bash
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../venv/bin/activate"

export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14
export CUDAHOSTCXX=/usr/bin/g++-14
export CUDA_HOME="$PWD/.cuda"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PWD/src"

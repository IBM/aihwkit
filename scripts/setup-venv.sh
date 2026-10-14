#!/bin/bash
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.."

python3.14 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev,examples]'

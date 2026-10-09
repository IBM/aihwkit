# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Setup.py for `aihwkit`."""

import os
from typing import Optional

from setuptools import find_packages
from skbuild import setup

INSTALL_REQUIRES = [
    "torch{}".format(os.getenv("TORCH_VERSION_SPECIFIER", ">=1.9")),
    "torchvision",
    "scipy",
    "requests>=2.25,<3",
    "numpy>=1.22",
    "protobuf>=4.21.6",
]


def get_cuda_version() -> Optional[str]:
    """Get the ``<major><minor>`` CUDA version (e.g. ``126``) of the installed torch."""
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    if torch.version.cuda is None:
        return None
    major, minor = torch.version.cuda.split(".")[:2]
    return f"{major}{minor}"


def get_version() -> str:
    """Get the package version.

    When building with ``USE_CUDA`` enabled, a ``+cu<cuda_version>`` local version label is
    appended (e.g. ``1.2.0+cu126``), so that CUDA wheels are distinguishable from CPU ones.
    """
    version_path = os.path.join(os.path.dirname(__file__), "src", "aihwkit", "VERSION.txt")
    with open(version_path, encoding="utf-8") as version_file:
        version = version_file.read().strip()

    if os.getenv("USE_CUDA", "").upper() in ("ON", "1", "TRUE", "YES", "Y"):
        cuda_version = get_cuda_version()
        if cuda_version is None:
            raise RuntimeError("USE_CUDA is enabled but no CUDA-enabled torch is installed")
        version = f"{version}+cu{cuda_version}"

    return version


def get_long_description() -> str:
    """Get the package long description."""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(readme_path, encoding="utf-8") as readme_file:
        return readme_file.read().strip()


setup(
    name="aihwkit",
    version=get_version(),
    description="IBM Analog Hardware Acceleration Kit",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/IBM/aihwkit",
    author="IBM Research",
    author_email="aihwkit@us.ibm.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
    keywords=[
        "ai",
        "analog",
        "rpu",
        "torch",
        "memristor",
        "pcm",
        "reram",
        "crossbar",
        "in-memory",
        "nvm",
        "non-von-neumann",
        "non-volatile memory",
        "phase-change material",
    ],
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"aihwkit": ["VERSION.txt"]},
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.7",
    zip_safe=False,
    extras_require={
        "visualization": ["matplotlib>=3.0"],
        "fitting": ["lmfit"],
        "bert": ["transformers", "evaluate", "datasets", "wandb", "tensorboard"],
    },
)

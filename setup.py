# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Setup.py for `aihwkit`."""

import os

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


def get_version() -> str:
    """Get the package version."""
    version_path = os.path.join(os.path.dirname(__file__), "src", "aihwkit", "VERSION.txt")
    with open(version_path, encoding="utf-8") as version_file:
        return version_file.read().strip()


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
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
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

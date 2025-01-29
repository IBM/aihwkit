# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 13: Experiment training (3 fully connected layers).

Creation and execution of a Basic Training experiment, where the model uses
3 fully connected analog layers.
"""
# pylint: disable=invalid-name

import os

# Imports from PyTorch.
import torch
from torch.nn import Flatten, Sigmoid, LogSoftmax

# Imports from aihwkit.
from torchvision.datasets import FashionMNIST

from aihwkit.experiments import BasicTraining
from aihwkit.experiments.runners import LocalRunner
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice
from aihwkit.simulator.rpu_base import cuda

# Check device
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Path where the datasets will be stored.
PATH_DATASET = os.path.join("data", "DATASET")

# Network definition.
INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10

# Training parameters.
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.05


def main():
    """Create and execute an experiment."""
    model = AnalogSequential(
        Flatten(),
        AnalogLinear(
            INPUT_SIZE,
            HIDDEN_SIZES[0],
            True,
            rpu_config=SingleRPUConfig(device=ConstantStepDevice()),
        ),
        Sigmoid(),
        AnalogLinear(
            HIDDEN_SIZES[0],
            HIDDEN_SIZES[1],
            True,
            rpu_config=SingleRPUConfig(device=ConstantStepDevice()),
        ),
        Sigmoid(),
        AnalogLinear(
            HIDDEN_SIZES[1],
            OUTPUT_SIZE,
            True,
            rpu_config=SingleRPUConfig(device=ConstantStepDevice()),
        ),
        LogSoftmax(dim=1),
    )

    # Create the training Experiment.
    experiment = BasicTraining(
        dataset=FashionMNIST, model=model, epochs=EPOCHS, batch_size=BATCH_SIZE
    )

    # Create the runner and execute the experiment.
    runner = LocalRunner(device=DEVICE)
    results = runner.run(experiment, dataset_root=PATH_DATASET)
    print(results)


if __name__ == "__main__":
    # Execute only if run as the entry point into the program.
    main()

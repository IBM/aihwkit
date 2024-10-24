# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 13: Custom experiment training (3 fully connected layers).

Example of customization of a Basic Training experiment, adding the use of
a scheduler to the base experiment.
"""
# pylint: disable=invalid-name

import os

# Imports from PyTorch.
import torch
from torch.nn import Flatten, Sigmoid, LogSoftmax

# Imports from aihwkit.
from torch.optim.lr_scheduler import StepLR
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


class BasicTrainingWithScheduler(BasicTraining):
    """Custom BasicTraining that allows using a scheduler.

    This is an example on how to extend BasicTraining. In this case, we change
    the training algorithm in order to support using a scheduler.
    """

    scheduler = None

    def train(
        self, training_loader, validation_loader, model, optimizer, loss_function, epochs, device
    ):
        # Initialize the custom scheduler.
        self.scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

        return super().train(
            training_loader, validation_loader, model, optimizer, loss_function, epochs, device
        )

    def training_step(self, training_loader, model, optimizer, loss_function, device):
        super().training_step(training_loader, model, optimizer, loss_function, device)

        # Decay learning rate if needed.
        self.scheduler.step()


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
    experiment = BasicTrainingWithScheduler(
        dataset=FashionMNIST, model=model, epochs=EPOCHS, batch_size=BATCH_SIZE
    )

    # Create the runner and execute the experiment.
    runner = LocalRunner(device=DEVICE)
    results = runner.run(experiment, dataset_root=PATH_DATASET)
    print(results)


if __name__ == "__main__":
    # Execute only if run as the entry point into the program.
    main()

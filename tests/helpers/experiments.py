# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=missing-function-docstring,too-few-public-methods

"""Models helpers for aihwkit tests."""

from typing import Any

from torch.nn import (
    BatchNorm2d,
    Conv2d,
    Flatten,
    Linear,
    LogSoftmax,
    MaxPool2d,
    Module,
    ReLU,
    Sigmoid,
    Tanh,
)
from torchvision.datasets import FashionMNIST, SVHN

from aihwkit.experiments import BasicTraining
from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential
from aihwkit.simulator.presets import (
    CapacitorPreset,
    ReRamSBPreset,
    TikiTakaEcRamPreset,
    TikiTakaReRamSBPreset,
    EcRamPreset,
    IdealizedPreset,
)


class FullyConnectedFashionMNIST:
    """3 fully-connected layers; with FashionMNIST."""

    def get_experiment(self, real: bool = False, rpu_config: Any = CapacitorPreset):
        """Return a BasicTraining experiment."""
        argv = {
            "dataset": FashionMNIST,
            "model": self.get_model(rpu_config),
            "epochs": 30,
            "batch_size": 8,
            "learning_rate": 0.01,
        }

        if not real:
            argv["epochs"] = 1

        return BasicTraining(**argv)

    def get_model(self, rpu_config: Any = CapacitorPreset) -> Module:
        return AnalogSequential(
            Flatten(),
            AnalogLinear(784, 256, bias=True, rpu_config=rpu_config()),
            Sigmoid(),
            AnalogLinear(256, 128, bias=True, rpu_config=rpu_config()),
            Sigmoid(),
            AnalogLinear(128, 10, bias=True, rpu_config=rpu_config()),
            LogSoftmax(dim=1),
        )


class FullyConnectedFashionMNISTTikiTaka:
    """3 fully-connected layers; with FashionMNIST and tiki-taka."""

    def get_experiment(self, real: bool = False, rpu_config: Any = TikiTakaReRamSBPreset):
        """Return a BasicTraining experiment."""
        argv = {
            "dataset": FashionMNIST,
            "model": self.get_model(rpu_config),
            "epochs": 30,
            "batch_size": 8,
            "learning_rate": 0.01,
        }

        if not real:
            argv["epochs"] = 1

        return BasicTraining(**argv)

    def get_model(self, rpu_config: Any = TikiTakaReRamSBPreset) -> Module:
        return AnalogSequential(
            Flatten(),
            AnalogLinear(784, 256, bias=True, rpu_config=rpu_config()),
            Sigmoid(),
            AnalogLinear(256, 128, bias=True, rpu_config=rpu_config()),
            Sigmoid(),
            AnalogLinear(128, 10, bias=True, rpu_config=rpu_config()),
            LogSoftmax(dim=1),
        )


class LeNet5FashionMNIST:
    """LeNet5; with FashionMNIST."""

    def get_experiment(self, real: bool = False, rpu_config: Any = EcRamPreset):
        """Return a BasicTraining experiment."""
        argv = {
            "dataset": FashionMNIST,
            "model": self.get_model(rpu_config),
            "epochs": 30,
            "batch_size": 8,
            "learning_rate": 0.01,
        }

        if not real:
            argv["epochs"] = 1

        return BasicTraining(**argv)

    def get_model(self, rpu_config: Any = EcRamPreset) -> Module:
        return AnalogSequential(
            AnalogConv2d(
                in_channels=1, out_channels=16, kernel_size=5, stride=1, rpu_config=rpu_config()
            ),
            Tanh(),
            MaxPool2d(kernel_size=2),
            AnalogConv2d(
                in_channels=16, out_channels=32, kernel_size=5, stride=1, rpu_config=rpu_config()
            ),
            Tanh(),
            MaxPool2d(kernel_size=2),
            Tanh(),
            Flatten(),
            AnalogLinear(in_features=512, out_features=128, rpu_config=rpu_config()),
            Tanh(),
            AnalogLinear(in_features=128, out_features=10, rpu_config=rpu_config()),
            LogSoftmax(dim=1),
        )


class LeNet5SVHN:
    """LeNet5; with SVHN."""

    def get_experiment(self, real: bool = False, rpu_config: Any = ReRamSBPreset):
        """Return a BasicTraining experiment."""
        argv = {
            "dataset": SVHN,
            "model": self.get_model(rpu_config),
            "epochs": 30,
            "batch_size": 8,
            "learning_rate": 0.01,
        }

        if not real:
            argv["epochs"] = 1

        return BasicTraining(**argv)

    def get_model(self, rpu_config: Any = ReRamSBPreset) -> Module:
        return AnalogSequential(
            AnalogConv2d(
                in_channels=3, out_channels=16, kernel_size=5, stride=1, rpu_config=rpu_config()
            ),
            Tanh(),
            MaxPool2d(kernel_size=2),
            AnalogConv2d(
                in_channels=16, out_channels=32, kernel_size=5, stride=1, rpu_config=rpu_config()
            ),
            Tanh(),
            MaxPool2d(kernel_size=2),
            Tanh(),
            Flatten(),
            AnalogLinear(in_features=800, out_features=128, rpu_config=rpu_config()),
            Tanh(),
            AnalogLinear(in_features=128, out_features=10, rpu_config=rpu_config()),
            LogSoftmax(dim=1),
        )


class Vgg8SVHN:
    """Vgg8; with SVHN."""

    def get_experiment(self, real: bool = False, rpu_config: Any = IdealizedPreset):
        """Return a BasicTraining experiment."""
        argv = {
            "dataset": SVHN,
            "model": self.get_model(rpu_config),
            "epochs": 20,
            "batch_size": 10,
            "learning_rate": 0.01,
        }

        if not real:
            argv["epochs"] = 1

        return BasicTraining(**argv)

    def get_model(self, rpu_config: Any = IdealizedPreset) -> Module:
        return AnalogSequential(
            Conv2d(in_channels=3, out_channels=48, kernel_size=3, stride=1, padding=1),
            ReLU(),
            AnalogConv2d(
                in_channels=48,
                out_channels=48,
                kernel_size=3,
                stride=1,
                padding=1,
                rpu_config=rpu_config(),
            ),
            BatchNorm2d(48),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            AnalogConv2d(
                in_channels=48,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
                rpu_config=rpu_config(),
            ),
            ReLU(),
            AnalogConv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
                rpu_config=rpu_config(),
            ),
            BatchNorm2d(96),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            AnalogConv2d(
                in_channels=96,
                out_channels=144,
                kernel_size=3,
                stride=1,
                padding=1,
                rpu_config=rpu_config(),
            ),
            ReLU(),
            AnalogConv2d(
                in_channels=144,
                out_channels=144,
                kernel_size=3,
                stride=1,
                padding=1,
                rpu_config=rpu_config(),
            ),
            BatchNorm2d(144),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            Flatten(),
            AnalogLinear(in_features=16 * 144, out_features=384, rpu_config=rpu_config()),
            ReLU(),
            Linear(in_features=384, out_features=10),
            LogSoftmax(dim=1),
        )


class Vgg8SVHNTikiTaka:
    """Vgg8; with SVHN (and Tiki-Taka)."""

    def get_experiment(self, real: bool = False, rpu_config: Any = TikiTakaEcRamPreset):
        """Return a BasicTraining experiment."""
        argv = {
            "dataset": SVHN,
            "model": self.get_model(rpu_config),
            "epochs": 20,
            "batch_size": 10,
            "learning_rate": 0.01,
        }

        if not real:
            argv["epochs"] = 1

        return BasicTraining(**argv)

    def get_model(self, rpu_config: Any = TikiTakaEcRamPreset) -> Module:
        return AnalogSequential(
            Conv2d(in_channels=3, out_channels=48, kernel_size=3, stride=1, padding=1),
            ReLU(),
            AnalogConv2d(
                in_channels=48,
                out_channels=48,
                kernel_size=3,
                stride=1,
                padding=1,
                rpu_config=rpu_config(),
            ),
            BatchNorm2d(48),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            AnalogConv2d(
                in_channels=48,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
                rpu_config=rpu_config(),
            ),
            ReLU(),
            AnalogConv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
                rpu_config=rpu_config(),
            ),
            BatchNorm2d(96),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            AnalogConv2d(
                in_channels=96,
                out_channels=144,
                kernel_size=3,
                stride=1,
                padding=1,
                rpu_config=rpu_config(),
            ),
            ReLU(),
            AnalogConv2d(
                in_channels=144,
                out_channels=144,
                kernel_size=3,
                stride=1,
                padding=1,
                rpu_config=rpu_config(),
            ),
            BatchNorm2d(144),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            Flatten(),
            AnalogLinear(in_features=16 * 144, out_features=384, rpu_config=rpu_config()),
            ReLU(),
            Linear(in_features=384, out_features=10),
            LogSoftmax(dim=1),
        )

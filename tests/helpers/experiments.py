# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-function-docstring,too-few-public-methods

"""Models helpers for aihwkit tests."""

from typing import Any

from torch.nn import (
    BatchNorm2d, Flatten, LogSoftmax, MaxPool2d, Module, ReLU, Sigmoid, Tanh
)
from torchvision.datasets import FashionMNIST, SVHN

from aihwkit.experiments import BasicTraining
from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential
from aihwkit.simulator.presets import (
    CapacitorPreset, ReRamSBPreset, TikiTakaReRamSBPreset,
    EcRamPreset, IdealizedPreset
)


class FullyConnectedFashionMNIST:
    """3 fully-connected layers; with FashionMNIST."""

    def get_experiment(
            self,
            real: bool = False,
            rpu_config: Any = CapacitorPreset
    ):
        """Return a BasicTraining experiment."""
        argv = {
            'dataset': FashionMNIST,
            'model': self.get_model(rpu_config),
            'epochs': 30,
            'batch_size': 8,
            'learning_rate': 0.01
        }

        if not real:
            argv['epochs'] = 1

        return BasicTraining(**argv)

    def get_model(self, rpu_config: Any = CapacitorPreset) -> Module:
        return AnalogSequential(
            Flatten(),
            AnalogLinear(784, 256, bias=True, rpu_config=rpu_config()),
            Sigmoid(),
            AnalogLinear(256, 128, bias=True, rpu_config=rpu_config()),
            Sigmoid(),
            AnalogLinear(128, 10, bias=True, rpu_config=rpu_config()),
            LogSoftmax(dim=1)
        )


class FullyConnectedFashionMNISTTikiTaka:
    """3 fully-connected layers; with FashionMNIST and tiki-taka."""

    def get_experiment(
            self,
            real: bool = False,
            rpu_config: Any = TikiTakaReRamSBPreset
    ):
        """Return a BasicTraining experiment."""
        argv = {
            'dataset': FashionMNIST,
            'model': self.get_model(rpu_config),
            'epochs': 30,
            'batch_size': 8,
            'learning_rate': 0.01
        }

        if not real:
            argv['epochs'] = 1

        return BasicTraining(**argv)

    def get_model(self, rpu_config: Any = TikiTakaReRamSBPreset) -> Module:
        return AnalogSequential(
            Flatten(),
            AnalogLinear(784, 256, bias=True, rpu_config=rpu_config()),
            Sigmoid(),
            AnalogLinear(256, 128, bias=True, rpu_config=rpu_config()),
            Sigmoid(),
            AnalogLinear(128, 10, bias=True, rpu_config=rpu_config()),
            LogSoftmax(dim=1)
        )


class LeNet5FashionMNIST:
    """LeNet5; with FashionMNIST."""

    def get_experiment(
            self,
            real: bool = False,
            rpu_config: Any = EcRamPreset
    ):
        """Return a BasicTraining experiment."""
        argv = {
            'dataset': FashionMNIST,
            'model': self.get_model(rpu_config),
            'epochs': 30,
            'batch_size': 8,
            'learning_rate': 0.01
        }

        if not real:
            argv['epochs'] = 1

        return BasicTraining(**argv)

    def get_model(self, rpu_config: Any = EcRamPreset) -> Module:
        return AnalogSequential(
            AnalogConv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1,
                         rpu_config=rpu_config(),
                         weight_scaling_omega=0.6),
            Tanh(),
            MaxPool2d(kernel_size=2),
            AnalogConv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1,
                         rpu_config=rpu_config(),
                         weight_scaling_omega=0.6),
            Tanh(),
            MaxPool2d(kernel_size=2),
            Tanh(),
            Flatten(),
            AnalogLinear(in_features=512, out_features=128, rpu_config=rpu_config(),
                         weight_scaling_omega=0.6),
            Tanh(),
            AnalogLinear(in_features=128, out_features=10, rpu_config=rpu_config(),
                         weight_scaling_omega=0.6),
            LogSoftmax(dim=1)
        )


class LeNet5SVHN:
    """LeNet5; with SVHN."""

    def get_experiment(
            self,
            real: bool = False,
            rpu_config: Any = ReRamSBPreset
    ):
        """Return a BasicTraining experiment."""
        argv = {
            'dataset': SVHN,
            'model': self.get_model(rpu_config),
            'epochs': 30,
            'batch_size': 8,
            'learning_rate': 0.01
        }

        if not real:
            argv['epochs'] = 1

        return BasicTraining(**argv)

    def get_model(self, rpu_config: Any = ReRamSBPreset) -> Module:
        return AnalogSequential(
            AnalogConv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1,
                         rpu_config=rpu_config(),
                         weight_scaling_omega=0.6),
            Tanh(),
            MaxPool2d(kernel_size=2),
            AnalogConv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1,
                         rpu_config=rpu_config(),
                         weight_scaling_omega=0.6),
            Tanh(),
            MaxPool2d(kernel_size=2),
            Tanh(),
            Flatten(),
            AnalogLinear(in_features=800, out_features=128, rpu_config=rpu_config(),
                         weight_scaling_omega=0.6),
            Tanh(),
            AnalogLinear(in_features=128, out_features=10, rpu_config=rpu_config(),
                         weight_scaling_omega=0.6),
            LogSoftmax(dim=1)
        )

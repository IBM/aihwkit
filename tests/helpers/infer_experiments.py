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
    Tanh,
    NLLLoss,
)
from torchvision.datasets import FashionMNIST, SVHN

from aihwkit.nn import (
    AnalogConv2d,
    AnalogLinear,
    AnalogSequential,
    AnalogConv2dMapped,
    AnalogLinearMapped,
)
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.presets.web import (
    OldWebComposerInferenceRPUConfig,
    WebComposerInferenceRPUConfig,
)
from aihwkit.experiments.experiments.inferencing import BasicInferencing


class Lenet5:
    """Hardware-aware LeNet5; with FashionMNIST."""

    def get_weight_template_id(self) -> str:
        raise NotImplementedError()

    def get_reference_accuracy(self) -> float:
        raise NotImplementedError()

    def get_experiment(
        self, real: bool = False, rpu_config: Any = InferenceRPUConfig()
    ) -> BasicInferencing:
        """Return a BasicInference experiment."""

        rpu_config = OldWebComposerInferenceRPUConfig()

        argv = {
            "dataset": FashionMNIST,
            "model": self.get_model(rpu_config),
            "batch_size": 8,
            "loss_function": NLLLoss,
            "weight_template_id": self.get_weight_template_id(),
            "inference_repeats": 10,
            "inference_time": 3600,
        }

        if not real:
            argv["inference_repeats"] = 2

        return BasicInferencing(**argv)

    def get_model(self, rpu_config: Any = InferenceRPUConfig()) -> Module:
        # set the InferenceRPUConfig
        channel = [16, 32, 512, 128]
        return AnalogSequential(
            AnalogConv2d(
                in_channels=1,
                out_channels=channel[0],
                kernel_size=5,
                stride=1,
                rpu_config=rpu_config,
            ),
            Tanh(),
            MaxPool2d(kernel_size=2),
            AnalogConv2d(
                in_channels=channel[0],
                out_channels=channel[1],
                kernel_size=5,
                stride=1,
                rpu_config=rpu_config,
            ),
            Tanh(),
            MaxPool2d(kernel_size=2),
            Tanh(),
            Flatten(),
            AnalogLinear(in_features=channel[2], out_features=channel[3], rpu_config=rpu_config),
            Tanh(),
            AnalogLinear(in_features=channel[3], out_features=10, rpu_config=rpu_config),
            LogSoftmax(dim=1),
        )


class Lenet5Mapped:
    """LeNet5; with FashionMNIST."""

    def get_weight_template_id(self) -> str:
        raise NotImplementedError()

    def get_reference_accuracy(self) -> float:
        raise NotImplementedError()

    def get_experiment(
        self, real: bool = False, rpu_config: Any = InferenceRPUConfig()
    ) -> BasicInferencing:
        """Return a BasicInference experiment."""

        rpu_config = WebComposerInferenceRPUConfig()

        argv = {
            "dataset": FashionMNIST,
            "model": self.get_model(rpu_config),
            "batch_size": 8,
            "loss_function": NLLLoss,
            "weight_template_id": self.get_weight_template_id(),
            "inference_repeats": 10,
            "inference_time": 3600,
        }

        if not real:
            argv["inference_repeats"] = 2

        return BasicInferencing(**argv)

    def get_model(self, rpu_config: Any = InferenceRPUConfig()) -> Module:
        # set the InferenceRPUConfig
        channel = [16, 32, 512, 128]
        return AnalogSequential(
            AnalogConv2dMapped(
                in_channels=1,
                out_channels=channel[0],
                kernel_size=5,
                stride=1,
                rpu_config=rpu_config,
            ),
            Tanh(),
            MaxPool2d(kernel_size=2),
            AnalogConv2dMapped(
                in_channels=channel[0],
                out_channels=channel[1],
                kernel_size=5,
                stride=1,
                rpu_config=rpu_config,
            ),
            Tanh(),
            MaxPool2d(kernel_size=2),
            Tanh(),
            Flatten(),
            AnalogLinearMapped(
                in_features=channel[2], out_features=channel[3], rpu_config=rpu_config
            ),
            Tanh(),
            AnalogLinearMapped(in_features=channel[3], out_features=10, rpu_config=rpu_config),
            LogSoftmax(dim=1),
        )


class DigitalTrainedLenet5(Lenet5):
    """Digitally trained lenet 5"""

    def get_reference_accuracy(self) -> float:
        """Reference for 100 samples without noise"""
        return 88.0

    def get_weight_template_id(self) -> str:
        return "digital-trained-lenet5"

    get_experiment = Lenet5.get_experiment
    get_model = Lenet5.get_model


class HwTrainedLenet5(Lenet5):
    """HWA trained lenet 5"""

    def get_reference_accuracy(self) -> float:
        """Reference for 100 samples without noise"""
        return 86.0

    def get_weight_template_id(self) -> str:
        return "hw-trained-lenet5"

    get_experiment = Lenet5.get_experiment
    get_model = Lenet5.get_model


class DigitalTrainedLenet5Mapped(Lenet5Mapped):
    """Digitally trained lenet 5 mapped"""

    def get_reference_accuracy(self) -> float:
        """Reference for 100 samples without noise"""
        return 89.0

    def get_weight_template_id(self) -> str:
        return "digital-trained-lenet5-mapped"

    get_experiment = Lenet5Mapped.get_experiment
    get_model = Lenet5Mapped.get_model


class HwTrainedLenet5Mapped(Lenet5Mapped):
    """HWA trained lenet 5 mapped"""

    def get_reference_accuracy(self) -> float:
        """Reference for 100 samples without noise"""
        return 87.0

    def get_weight_template_id(self) -> str:
        return "hwa-trained-lenet5-mapped"

    get_experiment = Lenet5Mapped.get_experiment
    get_model = Lenet5Mapped.get_model


class Vgg8:
    """Vgg8; with SVHN."""

    def get_weight_template_id(self) -> str:
        raise NotImplementedError()

    def get_reference_accuracy(self) -> float:
        raise NotImplementedError()

    def get_experiment(
        self, real: bool = False, rpu_config: Any = InferenceRPUConfig()
    ) -> BasicInferencing:
        """Return a BasicInference experiment."""

        rpu_config = OldWebComposerInferenceRPUConfig()

        argv = {
            "dataset": SVHN,
            "model": self.get_model(rpu_config),
            "batch_size": 128,
            "loss_function": NLLLoss,
            "weight_template_id": self.get_weight_template_id(),
            "inference_repeats": 10,
            "inference_time": 3600,
        }

        if not real:
            argv["inference_repeats"] = 2

        return BasicInferencing(**argv)

    def get_model(self, rpu_config: Any = InferenceRPUConfig()) -> Module:
        return AnalogSequential(
            Conv2d(in_channels=3, out_channels=48, kernel_size=3, stride=1, padding=1),
            ReLU(),
            AnalogConv2d(
                in_channels=48,
                out_channels=48,
                kernel_size=3,
                stride=1,
                padding=1,
                rpu_config=rpu_config,
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
                rpu_config=rpu_config,
            ),
            ReLU(),
            AnalogConv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
                rpu_config=rpu_config,
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
                rpu_config=rpu_config,
            ),
            ReLU(),
            AnalogConv2d(
                in_channels=144,
                out_channels=144,
                kernel_size=3,
                stride=1,
                padding=1,
                rpu_config=rpu_config,
            ),
            BatchNorm2d(144),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            Flatten(),
            AnalogLinear(in_features=16 * 144, out_features=384, rpu_config=rpu_config),
            ReLU(),
            Linear(in_features=384, out_features=10),
            LogSoftmax(dim=1),
        )


class Vgg8Mapped:
    """Vgg8; with SVHN. Mapped"""

    def get_weight_template_id(self) -> str:
        raise NotImplementedError()

    def get_reference_accuracy(self) -> float:
        raise NotImplementedError()

    def get_experiment(
        self, real: bool = False, rpu_config: Any = InferenceRPUConfig()
    ) -> BasicInferencing:
        """Return a BasicInference experiment."""

        rpu_config = WebComposerInferenceRPUConfig()

        argv = {
            "dataset": SVHN,
            "model": self.get_model(rpu_config),
            "batch_size": 128,
            "loss_function": NLLLoss,
            "weight_template_id": self.get_weight_template_id(),
            "inference_repeats": 10,
            "inference_time": 3600,
        }

        if not real:
            argv["inference_repeats"] = 2

        return BasicInferencing(**argv)

    def get_model(self, rpu_config: Any = InferenceRPUConfig()) -> Module:
        return AnalogSequential(
            AnalogConv2dMapped(
                in_channels=3,
                out_channels=48,
                kernel_size=3,
                stride=1,
                padding=1,
                rpu_config=rpu_config,
            ),
            ReLU(),
            AnalogConv2dMapped(
                in_channels=48,
                out_channels=48,
                kernel_size=3,
                stride=1,
                padding=1,
                rpu_config=rpu_config,
            ),
            BatchNorm2d(48),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            AnalogConv2dMapped(
                in_channels=48,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
                rpu_config=rpu_config,
            ),
            ReLU(),
            AnalogConv2dMapped(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
                rpu_config=rpu_config,
            ),
            BatchNorm2d(96),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            AnalogConv2dMapped(
                in_channels=96,
                out_channels=144,
                kernel_size=3,
                stride=1,
                padding=1,
                rpu_config=rpu_config,
            ),
            ReLU(),
            AnalogConv2dMapped(
                in_channels=144,
                out_channels=144,
                kernel_size=3,
                stride=1,
                padding=1,
                rpu_config=rpu_config,
            ),
            BatchNorm2d(144),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            Flatten(),
            AnalogLinearMapped(in_features=16 * 144, out_features=384, rpu_config=rpu_config),
            ReLU(),
            AnalogLinearMapped(in_features=384, out_features=10, rpu_config=rpu_config),
            LogSoftmax(dim=1),
        )


class HwTrainedVgg8Mapped(Vgg8Mapped):
    """Vgg8; with SVHN. Mapped"""

    def get_reference_accuracy(self) -> float:
        """Reference for 100 samples without noise"""
        # return 97.0
        return 96.0

    def get_weight_template_id(self) -> str:
        return "hwa-trained-vgg8-mapped"

    get_experiment = Vgg8Mapped.get_experiment
    get_model = Vgg8Mapped.get_model


class DigitalTrainedVgg8Mapped(Vgg8Mapped):
    """Vgg8; with SVHN."""

    def get_reference_accuracy(self) -> float:
        """Reference for 100 samples without noise"""
        return 95.0

    def get_weight_template_id(self) -> str:
        return "digital-trained-vgg8-mapped"

    get_experiment = Vgg8Mapped.get_experiment
    get_model = Vgg8Mapped.get_model


class HwTrainedVgg8(Vgg8):
    """Vgg8; with SVHN. Mapped"""

    def get_reference_accuracy(self) -> float:
        """Reference for 100 samples without noise"""
        return 93.0

    def get_weight_template_id(self) -> str:
        return "hw-trained-vgg8"

    get_experiment = Vgg8.get_experiment
    get_model = Vgg8.get_model


class DigitalTrainedVgg8(Vgg8):
    """Vgg8; with SVHN."""

    def get_reference_accuracy(self) -> float:
        """Reference for 100 samples without noise"""
        return 6.0

    def get_weight_template_id(self) -> str:
        return "digital-trained-vgg8"

    get_experiment = Vgg8.get_experiment
    get_model = Vgg8.get_model

# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
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
    BatchNorm2d, Conv2d, Flatten, Linear, LogSoftmax, MaxPool2d, Module,
    ReLU, Tanh, NLLLoss
)
from torchvision.datasets import FashionMNIST, SVHN

from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential
from aihwkit.simulator.presets.utils import PresetIOParameters
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import WeightNoiseType, WeightClipType
from aihwkit.inference import PCMLikeNoiseModel
from aihwkit.experiments.experiments.inferencing import BasicInferencing


class HwTrainedLenet5:
    """Hardware-aware LeNet5; with FashionMNIST."""

    def get_reference_accuracy(self):
        """ Reference for 100 samples without noise """
        return 86.0

    def get_experiment(
            self,
            real: bool = False,
            rpu_config: Any = InferenceRPUConfig()
    ):
        """Return a BasicInference experiment."""

        rpu_config = InferenceRPUConfig(forward=PresetIOParameters())
        rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
        rpu_config.clip.type = WeightClipType.FIXED_VALUE
        rpu_config.clip.fixed_value = 1.0
        rpu_config.forward.w_noise = 0.0175

        argv = {
            'dataset': FashionMNIST,
            'model': self.get_model(rpu_config),
            'batch_size': 8,
            'loss_function': NLLLoss,
            'weight_template_id': 'hw-trained-lenet5',
            'inference_repeats': 10,
            'inference_time': 3600
        }

        if not real:
            argv['inference_repeats'] = 2

        return BasicInferencing(**argv)

    def get_model(self, rpu_config: Any = InferenceRPUConfig) -> Module:
        # set the InferenceRPUConfig
        channel = [16, 32, 512, 128]
        return AnalogSequential(
           AnalogConv2d(in_channels=1, out_channels=channel[0], kernel_size=5, stride=1,
                        rpu_config=rpu_config),
           Tanh(),
           MaxPool2d(kernel_size=2),
           AnalogConv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=5, stride=1,
                        rpu_config=rpu_config),
           Tanh(),
           MaxPool2d(kernel_size=2),
           Tanh(),
           Flatten(),
           AnalogLinear(in_features=channel[2], out_features=channel[3], rpu_config=rpu_config),
           Tanh(),
           AnalogLinear(in_features=channel[3], out_features=10, rpu_config=rpu_config),
           LogSoftmax(dim=1)
        )


class DigitalTrainedLenet5:
    """Hardware-aware LeNet5; with FashionMNIST."""

    def get_reference_accuracy(self):
        """ Reference for 100 samples without noise """
        return 88.0

    def get_experiment(
            self,
            real: bool = False,
            rpu_config: Any = InferenceRPUConfig()
    ):
        """Return a BasicInference experiment."""

        rpu_config = InferenceRPUConfig(forward=PresetIOParameters())
        rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
        rpu_config.clip.type = WeightClipType.FIXED_VALUE
        rpu_config.clip.fixed_value = 1.0
        rpu_config.forward.w_noise = 0.0175

        argv = {
            'dataset': FashionMNIST,
            'model': self.get_model(rpu_config),
            'batch_size': 8,
            'loss_function': NLLLoss,
            'weight_template_id': 'digital-trained-lenet5',
            'inference_repeats': 10,
            'inference_time': 3600
        }

        if not real:
            argv['inference_repeats'] = 2

        return BasicInferencing(**argv)

    def get_model(self, rpu_config: Any = InferenceRPUConfig) -> Module:
        # set the InferenceRPUConfig
        channel = [16, 32, 512, 128]
        return AnalogSequential(
           AnalogConv2d(in_channels=1, out_channels=channel[0], kernel_size=5, stride=1,
                        rpu_config=rpu_config),
           Tanh(),
           MaxPool2d(kernel_size=2),
           AnalogConv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=5, stride=1,
                        rpu_config=rpu_config),
           Tanh(),
           MaxPool2d(kernel_size=2),
           Tanh(),
           Flatten(),
           AnalogLinear(in_features=channel[2], out_features=channel[3], rpu_config=rpu_config),
           Tanh(),
           AnalogLinear(in_features=channel[3], out_features=10, rpu_config=rpu_config),
           LogSoftmax(dim=1)
        )


class HwTrainedVgg8:
    """Vgg8; with SVHN."""

    def get_reference_accuracy(self):
        """ Reference for 100 samples without noise """
        return 93.0

    def get_experiment(
            self,
            real: bool = False,
            rpu_config: Any = InferenceRPUConfig()
    ):
        """Return a BasicInference experiment."""
        output_noise_strength = 0.03999999910593033
        adc = 9
        dac = 7
        programming_noise_scale = 1.0
        read_noise_scale = 1.0
        drift_scale = 1.0
        poly_first_order_coef = 1.965000033378601
        poly_second_order_coef = -1.1730999946594238
        poly_constant_coef = 0.26350000500679016
        rpu_config.forward.out_noise = output_noise_strength
        rpu_config.forward.out_res = 1/(2**adc - 2)
        rpu_config.forward.inp_res = 1/(2**dac - 2)
        rpu_config.noise_model.prog_noise_scale = programming_noise_scale
        rpu_config.noise_model.read_noise_scale = read_noise_scale
        rpu_config.noise_model.drift_scale = drift_scale
        rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0,
                                                   prog_coeff=[poly_constant_coef,
                                                               poly_first_order_coef,
                                                               poly_second_order_coef])

        argv = {
            'dataset': SVHN,
            'model': self.get_model(rpu_config),
            'batch_size': 128,
            'loss_function': NLLLoss,
            'weight_template_id': 'hw-trained-vgg8',
            'inference_repeats': 10,
            'inference_time': 3600

        }

        if not real:
            argv['inference_repeats'] = 2

        return BasicInferencing(**argv)

    def get_model(self, rpu_config: Any = InferenceRPUConfig) -> Module:
        return AnalogSequential(
            Conv2d(in_channels=3, out_channels=48,
                   kernel_size=3, stride=1, padding=1),
            ReLU(),
            AnalogConv2d(in_channels=48, out_channels=48,
                         kernel_size=3, stride=1, padding=1,
                         rpu_config=rpu_config),
            BatchNorm2d(48),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            AnalogConv2d(in_channels=48, out_channels=96,
                         kernel_size=3, stride=1, padding=1,
                         rpu_config=rpu_config),
            ReLU(),
            AnalogConv2d(in_channels=96, out_channels=96,
                         kernel_size=3, stride=1, padding=1,
                         rpu_config=rpu_config),
            BatchNorm2d(96),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            AnalogConv2d(in_channels=96, out_channels=144,
                         kernel_size=3, stride=1, padding=1,
                         rpu_config=rpu_config),
            ReLU(),
            AnalogConv2d(in_channels=144, out_channels=144,
                         kernel_size=3, stride=1, padding=1,
                         rpu_config=rpu_config),
            BatchNorm2d(144),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            Flatten(),
            AnalogLinear(in_features=16 * 144, out_features=384,
                         rpu_config=rpu_config),
            ReLU(),
            Linear(in_features=384, out_features=10),
            LogSoftmax(dim=1)
        )


class DigitalTrainedVgg8:
    """Vgg8; with SVHN."""

    def get_reference_accuracy(self):
        """ Reference for 100 samples without noise """
        return 6.0

    def get_experiment(
            self,
            real: bool = False,
            rpu_config: Any = InferenceRPUConfig()
    ):
        """Return a BasicInference experiment."""

        output_noise_strength = 0.03999999910593033
        adc = 9
        dac = 7
        programming_noise_scale = 1.0
        read_noise_scale = 1.0
        drift_scale = 1.0
        poly_first_order_coef = 1.965000033378601
        poly_second_order_coef = -1.1730999946594238
        poly_constant_coef = 0.26350000500679016
        rpu_config.forward.out_noise = output_noise_strength
        rpu_config.forward.out_res = 1/(2**adc - 2)
        rpu_config.forward.inp_res = 1/(2**dac - 2)
        rpu_config.noise_model.prog_noise_scale = programming_noise_scale
        rpu_config.noise_model.read_noise_scale = read_noise_scale
        rpu_config.noise_model.drift_scale = drift_scale
        rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0,
                                                   prog_coeff=[poly_constant_coef,
                                                               poly_first_order_coef,
                                                               poly_second_order_coef])

        argv = {
            'dataset': SVHN,
            'model': self.get_model(rpu_config),
            'batch_size': 128,
            'loss_function': NLLLoss,
            'weight_template_id': 'digital-trained-vgg8',
            'inference_repeats': 10,
            'inference_time': 3600

        }

        if not real:
            argv['inference_repeats'] = 2

        return BasicInferencing(**argv)

    def get_model(self, rpu_config: Any = InferenceRPUConfig) -> Module:
        return AnalogSequential(
            Conv2d(in_channels=3, out_channels=48,
                   kernel_size=3, stride=1, padding=1),
            ReLU(),
            AnalogConv2d(in_channels=48, out_channels=48,
                         kernel_size=3, stride=1, padding=1,
                         rpu_config=rpu_config),
            BatchNorm2d(48),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            AnalogConv2d(in_channels=48, out_channels=96,
                         kernel_size=3, stride=1, padding=1,
                         rpu_config=rpu_config),
            ReLU(),
            AnalogConv2d(in_channels=96, out_channels=96,
                         kernel_size=3, stride=1, padding=1,
                         rpu_config=rpu_config),
            BatchNorm2d(96),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            AnalogConv2d(in_channels=96, out_channels=144,
                         kernel_size=3, stride=1, padding=1,
                         rpu_config=rpu_config),
            ReLU(),
            AnalogConv2d(in_channels=144, out_channels=144,
                         kernel_size=3, stride=1, padding=1,
                         rpu_config=rpu_config),
            BatchNorm2d(144),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            Flatten(),
            AnalogLinear(in_features=16 * 144, out_features=384,
                         rpu_config=rpu_config),
            ReLU(),
            Linear(in_features=384, out_features=10),
            LogSoftmax(dim=1)
        )

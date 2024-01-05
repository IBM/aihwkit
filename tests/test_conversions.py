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

"""Tests for model conversions."""

from unittest import SkipTest

from torch import Tensor, manual_seed, randn
from torch.nn import Sequential, Linear, Conv2d, MaxPool2d, Tanh, LogSoftmax, Flatten, Module
from torch.nn.functional import mse_loss
from torch.optim import SGD
from torchvision.models import resnet18, alexnet
from parameterized import parameterized_class

from aihwkit.simulator.configs.configs import FloatingPointRPUConfig, InferenceRPUConfig
from aihwkit.nn import AnalogSequential, AnalogLinear, AnalogConv2d, AnalogWrapper
from aihwkit.nn.conversion import convert_to_analog, convert_to_digital

from .helpers.testcases import ParametrizedTestCase


@parameterized_class(
    [{"use_cuda": False, "name": "CPU"}, {"use_cuda": True, "name": "CUDA"}],
    class_name_func=lambda cls, _, params_dict: "%s%s" % (cls.__name__, params_dict["name"]),
)
class ConversionLayerTest(ParametrizedTestCase):
    """Linear layer abstractions tests."""

    @staticmethod
    def train_model_torch(model, loss_func, x_b, y_b):
        """Train the model with torch SGD."""
        opt = SGD(model.parameters(), lr=0.5)
        epochs = 100
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(x_b)
            loss = loss_func(pred, y_b)

            loss.backward()
            opt.step()

    def test_conversion_linear_sequential(self):
        """Test converting sequential and linear."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.3, 0.1]])
        y_b = Tensor([[0.3], [0.6]])

        manual_seed(4321)
        model = Sequential(Linear(4, 3), Linear(3, 3), Sequential(Linear(3, 1), Linear(1, 1)))
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            model = model.cuda()

        self.train_model_torch(model, loss_func, x_b, y_b)
        digital_loss = loss_func(model(x_b), y_b)
        analog_model = convert_to_analog(model, FloatingPointRPUConfig(), ensure_analog_root=False)
        self.assertEqual(analog_model[0].__class__, AnalogLinear)
        self.assertEqual(analog_model.__class__, AnalogSequential)
        self.assertTensorAlmostEqual(loss_func(analog_model(x_b), y_b), digital_loss)

        # convert back to digial
        new_model = convert_to_digital(analog_model)
        for new_mod, mod in zip(new_model.modules(), model.modules()):
            self.assertEqual(new_mod.__class__, mod.__class__)
            if hasattr(mod, "weight"):
                self.assertTensorAlmostEqual(new_mod.weight, mod.weight)
            if hasattr(mod, "bias") and mod.bias is not None:
                self.assertTensorAlmostEqual(new_mod.bias, mod.bias)

        self.train_model_torch(model, loss_func, x_b, y_b)
        self.train_model_torch(new_model, loss_func, x_b, y_b)
        self.assertTensorAlmostEqual(loss_func(new_model(x_b), y_b), loss_func(model(x_b), y_b))

    def test_conversion_lenet(self):
        """Convert Lenet"""

        manual_seed(4321)
        model = Sequential(
            Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1),
            Tanh(),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            Tanh(),
            MaxPool2d(kernel_size=2),
            Tanh(),
            Flatten(),
            Linear(in_features=512, out_features=128),
            Tanh(),
            Linear(in_features=128, out_features=10),
            LogSoftmax(dim=1),
        )
        analog_model = convert_to_analog(model, FloatingPointRPUConfig(), ensure_analog_root=False)
        x_input = randn((10, 1, 28, 28), device="cuda" if self.use_cuda else "cpu")

        if self.use_cuda:
            analog_model.cuda()
            model.cuda()

        self.assertEqual(analog_model[0].__class__, AnalogConv2d)
        self.assertEqual(analog_model.__class__, AnalogSequential)
        self.assertTensorAlmostEqual(analog_model(x_input), model(x_input))

        # convert back to digial
        new_model = convert_to_digital(analog_model)
        for new_mod, mod in zip(new_model.modules(), model.modules()):
            self.assertEqual(new_mod.__class__, mod.__class__)
            if hasattr(mod, "weight"):
                self.assertTensorAlmostEqual(new_mod.weight, mod.weight)
            if hasattr(mod, "bias") and mod.bias is not None:
                self.assertTensorAlmostEqual(new_mod.bias, mod.bias)

        self.assertTensorAlmostEqual(new_model(x_input), model(x_input))

    def test_conversion_lenet_module(self) -> None:
        """Convert Lenet with ensure_analog_root"""

        class LeNet(Module):
            """Simple module"""

            def __init__(self, my_model: Module):
                super().__init__()
                self.model = my_model

            def forward(self, x: Tensor) -> Tensor:
                """Forward pass"""
                return self.model(x)

        manual_seed(4321)
        model = Sequential(
            Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1),
            Tanh(),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            Tanh(),
            MaxPool2d(kernel_size=2),
            Tanh(),
            Flatten(),
            Linear(in_features=512, out_features=128),
            Tanh(),
            Linear(in_features=128, out_features=10),
            LogSoftmax(dim=1),
        )
        lenet = LeNet(model)

        analog_model = convert_to_analog(lenet, FloatingPointRPUConfig(), ensure_analog_root=True)
        x_input = randn((10, 1, 28, 28), device="cuda" if self.use_cuda else "cpu")

        if self.use_cuda:
            analog_model.cuda()
            model.cuda()

        self.assertTrue(isinstance(analog_model, AnalogWrapper))
        self.assertEqual(analog_model.model[0].__class__, AnalogConv2d)
        self.assertEqual(analog_model.model.__class__, AnalogSequential)
        self.assertTensorAlmostEqual(analog_model(x_input), lenet(x_input))

        # convert back to digial
        new_lenet = convert_to_digital(analog_model)
        for new_mod, mod in zip(new_lenet.modules(), lenet.modules()):
            self.assertEqual(new_mod.__class__.__name__, mod.__class__.__name__)
            if hasattr(mod, "weight"):
                self.assertTensorAlmostEqual(new_mod.weight, mod.weight)
            if hasattr(mod, "bias") and mod.bias is not None:
                self.assertTensorAlmostEqual(new_mod.bias, mod.bias)

        self.assertTensorAlmostEqual(new_lenet(x_input), lenet(x_input))

    def test_conversion_linear_sequential_specific(self):
        """Test converting sequential and linear."""

        def specfun(name, _, rpu_config):
            """special layer"""
            if name in ["0", "2.1"]:
                return InferenceRPUConfig()
            return rpu_config

        model = Sequential(Linear(4, 3), Linear(3, 3), Sequential(Linear(3, 1), Linear(1, 1)))
        if self.use_cuda:
            model = model.cuda()

        analog_model = convert_to_analog(
            model,
            FloatingPointRPUConfig(),
            specific_rpu_config_fun=specfun,
            ensure_analog_root=False,
        )
        self.assertEqual(analog_model[0].__class__, AnalogLinear)
        self.assertEqual(
            next(analog_model[0].analog_tiles()).rpu_config.__class__, InferenceRPUConfig
        )
        self.assertEqual(
            next(analog_model[2][1].analog_tiles()).rpu_config.__class__, InferenceRPUConfig
        )
        self.assertEqual(
            next(analog_model[1].analog_tiles()).rpu_config.__class__, FloatingPointRPUConfig
        )
        self.assertEqual(
            next(analog_model[2][0].analog_tiles()).rpu_config.__class__, FloatingPointRPUConfig
        )

    def test_conversion_torchvision_resnet(self):
        """Test converting resnet model from torchvision."""
        if self.use_cuda:
            raise SkipTest("Skipping for CUDA.")

        model = resnet18()

        analog_model = convert_to_analog(model, FloatingPointRPUConfig(), ensure_analog_root=False)
        self.assertEqual(analog_model.conv1.__class__, AnalogConv2d)
        self.assertEqual(analog_model.layer1.__class__, AnalogSequential)
        self.assertEqual(analog_model.layer1[0].conv1.__class__, AnalogConv2d)

        new_model = convert_to_digital(analog_model)
        for new_mod, mod in zip(new_model.modules(), model.modules()):
            self.assertEqual(new_mod.__class__, mod.__class__)
            if hasattr(mod, "weight"):
                self.assertTensorAlmostEqual(new_mod.weight, mod.weight)
            if hasattr(mod, "bias") and mod.bias is not None:
                self.assertTensorAlmostEqual(new_mod.bias, mod.bias)

    def test_conversion_torchvision_alexnet(self):
        """Test converting resnet model from torchvision"""
        if self.use_cuda:
            raise SkipTest("Skipping for CUDA.")

        model = alexnet()
        analog_model = convert_to_analog(model, FloatingPointRPUConfig(), ensure_analog_root=False)
        self.assertEqual(analog_model.features[0].__class__, AnalogConv2d)
        self.assertEqual(analog_model.classifier[6].__class__, AnalogLinear)
        self.assertEqual(analog_model.features.__class__, AnalogSequential)

        new_model = convert_to_digital(analog_model)
        for new_mod, mod in zip(new_model.modules(), model.modules()):
            self.assertEqual(new_mod.__class__, mod.__class__)
            if hasattr(mod, "weight"):
                self.assertTensorAlmostEqual(new_mod.weight, mod.weight)
            if hasattr(mod, "bias") and mod.bias is not None:
                self.assertTensorAlmostEqual(new_mod.bias, mod.bias)

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

"""Tests for model conversions."""

from unittest import SkipTest

from torch import Tensor, manual_seed
from torch.nn import Sequential, Linear
from torch.nn.functional import mse_loss
from torch.optim import SGD
from torchvision.models import resnet18, alexnet
from parameterized import parameterized_class

from aihwkit.simulator.configs.configs import FloatingPointRPUConfig
from aihwkit.nn import AnalogSequential, AnalogLinear, AnalogConv2d
from aihwkit.nn.conversion import convert_to_analog

from .helpers.testcases import ParametrizedTestCase


@parameterized_class([
    {'use_cuda': False, 'name': 'CPU'},
    {'use_cuda': True, 'name': 'CUDA'},
], class_name_func=lambda cls, _, params_dict: '%s%s' % (cls.__name__, params_dict['name']))
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
        model = Sequential(
            Linear(4, 3),
            Linear(3, 3),
            Sequential(
                Linear(3, 1),
                Linear(1, 1)
            )
        )
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            model = model.cuda()

        self.train_model_torch(model, loss_func, x_b, y_b)
        digital_loss = loss_func(model(x_b), y_b)
        analog_model = convert_to_analog(model, FloatingPointRPUConfig())
        self.assertEqual(analog_model[0].__class__, AnalogLinear)
        self.assertEqual(analog_model.__class__, AnalogSequential)
        self.assertTensorAlmostEqual(loss_func(analog_model(x_b), y_b), digital_loss)

    def test_conversion_torchvision_resnet(self):
        """Test converting resnet model from torchvision."""
        model = resnet18()
        if self.use_cuda:
            raise SkipTest('Skipping for CUDA.')

        analog_model = convert_to_analog(model, FloatingPointRPUConfig())
        self.assertEqual(analog_model.conv1.__class__, AnalogConv2d)
        self.assertEqual(analog_model.layer1.__class__, AnalogSequential)
        self.assertEqual(analog_model.layer1[0].conv1.__class__, AnalogConv2d)

    def test_conversion_torchvision_alexnet(self):
        """Test converting resnet model from torchvision"""
        model = alexnet()
        if self.use_cuda:
            raise SkipTest('Skipping for CUDA.')

        analog_model = convert_to_analog(model, FloatingPointRPUConfig())
        self.assertEqual(analog_model.features[0].__class__, AnalogConv2d)
        self.assertEqual(analog_model.classifier[6].__class__, AnalogLinear)
        self.assertEqual(analog_model.features.__class__, AnalogSequential)

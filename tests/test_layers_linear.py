# -*- coding: utf-8 -*-

# (C) Copyright 2020 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for linear layer."""

from torch import Tensor, manual_seed
from torch.nn import Sequential, Linear as torchLinear
from torch.nn.functional import mse_loss

from aihwkit.optim import AnalogSGD

from .helpers.decorators import parametrize_over_layers
from .helpers.layers import Linear, LinearCuda
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import FloatingPoint, ConstantStep, Inference


@parametrize_over_layers(
    layers=[Linear, LinearCuda],
    tiles=[FloatingPoint, ConstantStep, Inference],
    biases=[True, False]
)
class LinearLayerTest(ParametrizedTestCase):
    """Linear layer abstractions tests."""

    @staticmethod
    def train_model(model, loss_func, x_b, y_b):
        """Train the model."""
        opt = AnalogSGD(model.parameters(), lr=0.5)
        opt.regroup_param_groups(model)

        epochs = 100
        for _ in range(epochs):
            pred = model(x_b)
            loss = loss_func(pred, y_b)

            loss.backward()
            opt.step()
            opt.zero_grad()

    def test_single_analog_layer(self):
        """Check using a single layer."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2], [0.2, 0.4]])
        y_b = Tensor([[0.3], [0.6]])

        manual_seed(4321)
        model = self.get_layer(2, 1)
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            model = model.cuda()
        self.train_model(model, loss_func, x_b, y_b)

        self.assertLess(loss_func(model(x_b), y_b), 0.2)

    def test_single_analog_layer_sequential(self):
        """Check using a single layer as a Sequential."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2], [0.2, 0.4]])
        y_b = Tensor([[0.3], [0.6]])

        manual_seed(4321)
        model = Sequential(self.get_layer(2, 1))
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            model = model.cuda()
        self.train_model(model, loss_func, x_b, y_b)

        self.assertLess(loss_func(model(x_b), y_b), 0.2)

    def test_several_analog_layers(self):
        """Check using a several analog layers."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.3, 0.1]])
        y_b = Tensor([[0.3], [0.6]])

        manual_seed(4321)
        model = Sequential(
            self.get_layer(4, 2),
            self.get_layer(2, 1)
        )
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            model = model.cuda()
        self.train_model(model, loss_func, x_b, y_b)

        self.assertLess(loss_func(model(x_b), y_b), 0.2)

    def test_analog_and_digital(self):
        """Check mixing analog and digital layers."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2], [0.2, 0.4]])
        y_b = Tensor([[0.3], [0.6]])

        manual_seed(4321)
        model = Sequential(
            self.get_layer(2, 3),
            torchLinear(3, 3),
            self.get_layer(3, 1)
        )
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            model = model.cuda()
        self.train_model(model, loss_func, x_b, y_b)

        self.assertLess(loss_func(model(x_b), y_b), 0.2)

    def test_learning_rate_update(self):
        """Check the learning rate update is applied to tile."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2], [0.2, 0.4]])
        y_b = Tensor([[0.3], [0.6]])

        layer1 = self.get_layer(2, 3)
        layer2 = self.get_layer(3, 1)

        model = Sequential(layer1, layer2)
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            model = model.cuda()
        opt = AnalogSGD(model.parameters(), lr=0.5)
        opt.regroup_param_groups(model)

        new_lr = 0.07
        for param_group in opt.param_groups:
            param_group['lr'] = new_lr

        pred = model(x_b)
        loss = loss_func(pred, y_b)
        loss.backward()
        opt.step()

        self.assertAlmostEqual(layer1.analog_tile.get_learning_rate(), new_lr)

    def test_learning_rate_update_fn(self):
        """Check the learning rate update is applied to tile."""
        layer1 = self.get_layer(2, 3)
        layer2 = self.get_layer(3, 1)

        model = Sequential(layer1, layer2)
        if self.use_cuda:
            model = model.cuda()
        opt = AnalogSGD(model.parameters(), lr=0.5)
        opt.regroup_param_groups(model)

        new_lr = 0.07

        opt.set_learning_rate(new_lr)

        self.assertAlmostEqual(layer1.analog_tile.get_learning_rate(), new_lr)
        self.assertAlmostEqual(layer2.analog_tile.get_learning_rate(), new_lr)

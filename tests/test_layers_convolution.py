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

"""Tests for layer abstractions."""

from torch import randn
from torch.nn import Conv2d as torch_Conv2d, Sequential
from torch.nn.functional import mse_loss

from aihwkit.optim import AnalogSGD

from .helpers.decorators import parametrize_over_layers
from .helpers.layers import Conv2d, Conv2dCuda
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import FloatingPoint


@parametrize_over_layers(
    layers=[Conv2d, Conv2dCuda],
    tiles=[FloatingPoint],
    biases=[True, False]
)
class ConvolutionLayerTest(ParametrizedTestCase):
    """Convolution layer abstractions tests."""

    def get_digital_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2):
        """Return a digital layer."""
        layer = torch_Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             padding=padding,
                             bias=self.bias)
        if self.use_cuda:
            layer = layer.cuda()

        return layer

    def set_weights_from_digital_model(self, analog_model, digital_model):
        """Set the analog model weights based on the digital model."""
        weights, biases = self.get_weights_from_digital_model(analog_model, digital_model)
        analog_model.analog_tile.set_weights(weights, biases, realistic=False)

    @staticmethod
    def get_weights_from_digital_model(analog_model, digital_model):
        """Set the analog model weights based on the digital model."""
        weights = digital_model.weight.data.detach().reshape(
            [analog_model.out_features, analog_model.in_features])
        biases = None
        if digital_model.bias is not None:
            biases = digital_model.bias.data.detach()

        return weights, biases

    @staticmethod
    def train_model(model, loss_func, x_b, y_b):
        """Train the model."""
        opt = AnalogSGD(model.parameters(), lr=0.1)
        opt.regroup_param_groups(model)

        epochs = 10
        for _ in range(epochs):
            pred = model(x_b)
            loss = loss_func(pred, y_b)
            loss.backward()
            opt.step()
            opt.zero_grad()

    def test_torch_original_layer(self):
        """Test a single layer, having the digital layer as reference."""
        # This tests the forward pass
        model = self.get_digital_layer(in_channels=2, out_channels=3, kernel_size=4, padding=2)
        x = randn(3, 2, 4, 4)

        if self.use_cuda:
            x = x.cuda()

        y = model(x)

        analog_model = self.get_layer(in_channels=2, out_channels=3, kernel_size=4, padding=2)
        self.set_weights_from_digital_model(analog_model, model)

        y_analog = analog_model(x)
        self.assertTensorAlmostEqual(y_analog, y)

    def test_torch_train_original_layer(self):
        """Test the forward and update pass, having the digital layer as reference."""
        model = self.get_digital_layer(in_channels=2, out_channels=3, kernel_size=4, padding=2)
        analog_model = self.get_layer(in_channels=2, out_channels=3, kernel_size=4, padding=2)
        self.set_weights_from_digital_model(analog_model, model)

        loss_func = mse_loss
        y_b = randn(3, 3, 5, 5)
        x_b = randn(3, 2, 4, 4)

        if self.use_cuda:
            y_b = y_b.cuda()
            x_b = x_b.cuda()

        self.train_model(model, loss_func, x_b, y_b)
        self.train_model(analog_model, loss_func, x_b, y_b)

        weight, bias = self.get_weights_from_digital_model(analog_model, model)

        weight_analog, bias_analog = analog_model.analog_tile.get_weights(realistic=False)

        self.assertTensorAlmostEqual(weight_analog, weight)
        if analog_model.use_bias:
            self.assertTensorAlmostEqual(bias_analog, bias)

    def test_torch_train_original_layer_multiple(self):
        """Test the backward pass, having the digital layer as reference."""
        model = Sequential(
            self.get_digital_layer(in_channels=2, out_channels=2, kernel_size=4, padding=2),
            self.get_digital_layer(in_channels=2, out_channels=3, kernel_size=4, padding=2)
        )

        analog_model = Sequential(
            self.get_layer(in_channels=2, out_channels=2, kernel_size=4, padding=2),
            self.get_layer(in_channels=2, out_channels=3, kernel_size=4, padding=2)
        )

        for analog_layer, layer in zip(analog_model.children(), model.children()):
            self.set_weights_from_digital_model(analog_layer, layer)

        loss_func = mse_loss
        y_b = randn(3, 3, 6, 6)
        x_b = randn(3, 2, 4, 4)

        if self.use_cuda:
            y_b = y_b.cuda()
            x_b = x_b.cuda()

        self.train_model(model, loss_func, x_b, y_b)
        self.train_model(analog_model, loss_func, x_b, y_b)

        for analog_layer, layer in zip(analog_model.children(), model.children()):
            weight, bias = self.get_weights_from_digital_model(analog_layer, layer)

            weight_analog, bias_analog = analog_layer.analog_tile.get_weights(realistic=False)

            self.assertTensorAlmostEqual(weight_analog, weight)
            if analog_layer.use_bias:
                self.assertTensorAlmostEqual(bias_analog, bias)

    def test_layer_instantiation(self):
        """Test AnalogConv2d layer instantiation."""
        model = self.get_layer(in_channels=2, out_channels=3, kernel_size=4)

        # Assert the number of elements of the weights.
        tile_weights, tile_biases = model.analog_tile.get_weights()

        self.assertEqual(tile_weights.numel(), 2*3*4*4)
        if model.use_bias:
            self.assertEqual(tile_biases.numel(), 3)

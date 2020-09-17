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

from unittest import TestCase, skipIf

from numpy.testing import assert_array_almost_equal

from torch import randn
from torch.nn import Conv2d, Sequential
from torch.nn.functional import mse_loss

from aihwkit.nn.modules.conv import AnalogConv2d
from aihwkit.optim.analog_sgd import AnalogSGD
from aihwkit.simulator.rpu_base import cuda

from aihwkit.simulator.devices import FloatingPointResistiveDevice


class ConvolutionLayerMixin:
    """Helper for testing AnalogConv2d layer."""

    USE_CUDA = False

    def get_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2):
        """Return a layer."""
        raise NotImplementedError

    def get_digital_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2):
        """Return a digital layer."""
        raise NotImplementedError

    def set_weights_from_digital_model(self, analog_model, digital_model):
        """Set the analog model weights based on the digital model."""
        weights, biases = self.get_weights_from_digital_model(analog_model, digital_model)
        analog_model.analog_tile.set_weights(weights, biases, realistic=False)

    def assertTensorAlmostEqual(self, tensor_a, tensor_b):
        """Assert that two tensors are almost equal."""
        # pylint: disable=invalid-name
        array_a = tensor_a.detach().cpu().numpy()
        array_b = tensor_b.detach().cpu().numpy()
        assert_array_almost_equal(array_a, array_b)

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

        if self.USE_CUDA:
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

        if self.USE_CUDA:
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

        if self.USE_CUDA:
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


class AnalogConv2dTestNoBias(TestCase, ConvolutionLayerMixin):
    """Test for AnalogConv2d (no bias)."""

    def get_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2):
        """Return a layer."""
        return AnalogConv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                            resistive_device=FloatingPointResistiveDevice())

    def get_digital_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2):
        """Return a digital layer."""
        return Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False)


class AnalogConv2dTestBias(TestCase, ConvolutionLayerMixin):
    """Test for AnalogConv2d (bias)."""

    def get_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2):
        """Return a layer."""
        return AnalogConv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=True,
                            resistive_device=FloatingPointResistiveDevice())

    def get_digital_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2):
        """Return a digital layer."""
        return Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=True)


@skipIf(not cuda.is_compiled(), 'not compiled with CUDA support')
class CudaAnalogConv2dTestNoBias(AnalogConv2dTestNoBias):
    """Test for AnalogConv2d (no bias, CUDA)."""

    USE_CUDA = True

    def get_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2):
        """Return a layer."""
        return AnalogConv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=True,
                            resistive_device=FloatingPointResistiveDevice()).cuda()

    def get_digital_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2):
        """Return a digital layer."""
        return Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=True).cuda()


@skipIf(not cuda.is_compiled(), 'not compiled with CUDA support')
class CudaAnalogConv2dTestBias(AnalogConv2dTestBias):
    """Test for AnalogConv2d (bias, CUDA)."""

    USE_CUDA = True

    def get_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2):
        """Return a layer."""
        return AnalogConv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=True,
                            resistive_device=FloatingPointResistiveDevice()).cuda()

    def get_digital_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2):
        """Return a digital layer."""
        return Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=True).cuda()

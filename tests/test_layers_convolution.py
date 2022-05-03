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

"""Tests for layer abstractions."""

from torch import randn
from torch.nn import (Conv1d as torch_Conv1d, Conv2d as torch_Conv2d,
                      Conv3d as torch_Conv3d, Sequential)
from torch.nn.functional import mse_loss

from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import MappingParameter

from .helpers.decorators import parametrize_over_layers
from .helpers.layers import Conv1d, Conv1dCuda, Conv2d, Conv2dCuda, Conv3d, Conv3dCuda
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import FloatingPoint, Inference


class ConvolutionLayerTest(ParametrizedTestCase):
    """Generic class for helping testing analog convolution layers."""

    digital_layer_cls = torch_Conv1d

    def get_digital_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2):
        """Return a digital layer."""
        layer = self.digital_layer_cls(in_channels=in_channels,
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
        analog_model.set_weights(weights, biases, force_exact=True)

    @staticmethod
    def get_weights_from_digital_model(analog_model, digital_model):
        """Set the analog model weights based on the digital model."""
        weights = digital_model.weight.data.detach().reshape(
            [analog_model.out_features, analog_model.in_features]).cpu()
        biases = None
        if digital_model.bias is not None:
            biases = digital_model.bias.data.detach().cpu()

        return weights, biases

    @staticmethod
    def get_weights_from_analog_model(analog_model):
        """Set the analog model weights based on the digital model."""
        weights, biases = analog_model.get_weights(force_exact=True)
        return weights, biases

    @staticmethod
    def train_model(model, loss_func, x_b, y_b):
        """Train the model."""
        opt = AnalogSGD(model.parameters(), lr=0.1)
        opt.regroup_param_groups(model)

        epochs = 10
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(x_b)
            loss = loss_func(pred, y_b)
            loss.backward()
            opt.step()


@parametrize_over_layers(
    layers=[Conv1d, Conv1dCuda],
    tiles=[FloatingPoint, Inference],
    biases=['analog', 'digital', None]
)
class Convolution1dLayerTest(ConvolutionLayerTest):
    """Tests for AnalogConv1d layer."""

    digital_layer_cls = torch_Conv1d

    def test_torch_original_layer(self):
        """Test a single layer, having the digital layer as reference."""
        # This tests the forward pass
        model = self.get_digital_layer(in_channels=2, out_channels=3, kernel_size=4, padding=2)
        x = randn(3, 2, 4)

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
        y_b = randn(3, 3, 5)
        x_b = randn(3, 2, 4)

        if self.use_cuda:
            y_b = y_b.cuda()
            x_b = x_b.cuda()

        self.train_model(model, loss_func, x_b, y_b)
        self.train_model(analog_model, loss_func, x_b, y_b)

        weight, bias = self.get_weights_from_digital_model(analog_model, model)

        weight_analog, bias_analog = self.get_weights_from_analog_model(analog_model)

        self.assertTensorAlmostEqual(weight_analog, weight)
        if self.bias:
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
        y_b = randn(3, 3, 6)
        x_b = randn(3, 2, 4)

        if self.use_cuda:
            y_b = y_b.cuda()
            x_b = x_b.cuda()

        self.train_model(model, loss_func, x_b, y_b)
        self.train_model(analog_model, loss_func, x_b, y_b)

        for analog_layer, layer in zip(analog_model.children(), model.children()):
            weight, bias = self.get_weights_from_digital_model(analog_layer, layer)
            weight_analog, bias_analog = self.get_weights_from_analog_model(analog_layer)

            self.assertTensorAlmostEqual(weight_analog, weight)
            if self.bias:
                self.assertTensorAlmostEqual(bias_analog, bias)

    def test_out_scaling_alpha_learning(self):
        """Check if out scaling alpha are learning."""
        rpu_config = InferenceRPUConfig(mapping=MappingParameter(
            weight_scaling_omega=0.6,
            learn_out_scaling_alpha=True))

        analog_model = Sequential(
            self.get_layer(in_channels=2, out_channels=2, kernel_size=4,
                           padding=2, rpu_config=rpu_config),
            self.get_layer(in_channels=2, out_channels=3, kernel_size=4,
                           padding=2, rpu_config=rpu_config)
        )

        loss_func = mse_loss
        y_b = randn(3, 3, 6)
        x_b = randn(3, 2, 4)

        if self.use_cuda:
            y_b = y_b.cuda()
            x_b = x_b.cuda()

        initial_out_scaling_alpha_0 = analog_model[0].analog_tile.get_out_scaling_alpha().clone()
        initial_out_scaling_alpha_1 = analog_model[1].analog_tile.get_out_scaling_alpha().clone()

        self.train_model(analog_model, loss_func, x_b, y_b)

        learned_out_scaling_alpha_0 = analog_model[0].analog_tile.get_out_scaling_alpha().clone()
        learned_out_scaling_alpha_1 = analog_model[1].analog_tile.get_out_scaling_alpha().clone()

        self.assertIsNotNone(analog_model[0].analog_tile.get_out_scaling_alpha().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_alpha_0, learned_out_scaling_alpha_0)
        self.assertIsNotNone(analog_model[1].analog_tile.get_out_scaling_alpha().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_alpha_1, learned_out_scaling_alpha_1)

    def test_out_scaling_alpha_learning_columnwise(self):
        """Check if out scaling alpha are learning."""
        rpu_config = InferenceRPUConfig(mapping=MappingParameter(
            weight_scaling_omega=0.6,
            learn_out_scaling_alpha=True,
            weight_scaling_omega_columnwise=True))

        analog_model = Sequential(
            self.get_layer(in_channels=2, out_channels=2, kernel_size=4,
                           padding=2, rpu_config=rpu_config),
            self.get_layer(in_channels=2, out_channels=3, kernel_size=4,
                           padding=2, rpu_config=rpu_config)
        )

        loss_func = mse_loss
        y_b = randn(3, 3, 6)
        x_b = randn(3, 2, 4)

        if self.use_cuda:
            y_b = y_b.cuda()
            x_b = x_b.cuda()

        initial_out_scaling_alpha_0 = analog_model[0].analog_tile.get_out_scaling_alpha().clone()
        initial_out_scaling_alpha_1 = analog_model[1].analog_tile.get_out_scaling_alpha().clone()

        self.train_model(analog_model, loss_func, x_b, y_b)

        learned_out_scaling_alpha_0 = analog_model[0].analog_tile.get_out_scaling_alpha().clone()
        learned_out_scaling_alpha_1 = analog_model[1].analog_tile.get_out_scaling_alpha().clone()

        self.assertIsNotNone(analog_model[0].analog_tile.get_out_scaling_alpha().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_alpha_0, learned_out_scaling_alpha_0)
        self.assertIsNotNone(analog_model[1].analog_tile.get_out_scaling_alpha().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_alpha_1, learned_out_scaling_alpha_1)

    def test_layer_instantiation(self):
        """Test AnalogConv2d layer instantiation."""
        model = self.get_layer(in_channels=2, out_channels=3, kernel_size=4)

        # Assert the number of elements of the weights.
        tile_weights, tile_biases = model.analog_tile.get_weights()

        self.assertEqual(tile_weights.numel(), 2*3*4)
        if model.analog_bias:
            self.assertEqual(tile_biases.numel(), 3)


@parametrize_over_layers(
    layers=[Conv2d, Conv2dCuda],
    tiles=[FloatingPoint, Inference],
    biases=['analog', 'digital', None]
)
class Convolution2dLayerTest(ConvolutionLayerTest):
    """Tests for AnalogConv2d layer."""

    digital_layer_cls = torch_Conv2d

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
        weight_analog, bias_analog = self.get_weights_from_analog_model(analog_model)

        self.assertTensorAlmostEqual(weight_analog, weight)
        if self.bias:
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

            weight_analog, bias_analog = self.get_weights_from_analog_model(analog_layer)

            self.assertTensorAlmostEqual(weight_analog, weight)
            if self.bias:
                self.assertTensorAlmostEqual(bias_analog, bias)

    def test_out_scaling_alpha_learning(self):
        """Check if out scaling alpha are learning."""
        rpu_config = InferenceRPUConfig(mapping=MappingParameter(
            weight_scaling_omega=0.6,
            learn_out_scaling_alpha=True))

        analog_model = Sequential(
            self.get_layer(in_channels=2, out_channels=2, kernel_size=4,
                           padding=2, rpu_config=rpu_config),
            self.get_layer(in_channels=2, out_channels=3, kernel_size=4,
                           padding=2, rpu_config=rpu_config)
        )

        loss_func = mse_loss
        y_b = randn(3, 3, 6, 6)
        x_b = randn(3, 2, 4, 4)

        if self.use_cuda:
            y_b = y_b.cuda()
            x_b = x_b.cuda()

        initial_out_scaling_alpha_0 = analog_model[0].analog_tile.get_out_scaling_alpha().clone()
        initial_out_scaling_alpha_1 = analog_model[1].analog_tile.get_out_scaling_alpha().clone()

        self.train_model(analog_model, loss_func, x_b, y_b)

        learned_out_scaling_alpha_0 = analog_model[0].analog_tile.get_out_scaling_alpha().clone()
        learned_out_scaling_alpha_1 = analog_model[1].analog_tile.get_out_scaling_alpha().clone()

        self.assertIsNotNone(analog_model[0].analog_tile.get_out_scaling_alpha().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_alpha_0, learned_out_scaling_alpha_0)
        self.assertIsNotNone(analog_model[1].analog_tile.get_out_scaling_alpha().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_alpha_1, learned_out_scaling_alpha_1)

    def test_out_scaling_alpha_learning_columnwise(self):
        """Check if out scaling alpha are learning."""
        rpu_config = InferenceRPUConfig(mapping=MappingParameter(
            weight_scaling_omega=0.6,
            learn_out_scaling_alpha=True,
            weight_scaling_omega_columnwise=True))

        analog_model = Sequential(
            self.get_layer(in_channels=2, out_channels=2, kernel_size=4,
                           padding=2, rpu_config=rpu_config),
            self.get_layer(in_channels=2, out_channels=3, kernel_size=4,
                           padding=2, rpu_config=rpu_config)
        )

        loss_func = mse_loss
        y_b = randn(3, 3, 6, 6)
        x_b = randn(3, 2, 4, 4)

        if self.use_cuda:
            y_b = y_b.cuda()
            x_b = x_b.cuda()

        initial_out_scaling_alpha_0 = analog_model[0].analog_tile.get_out_scaling_alpha().clone()
        initial_out_scaling_alpha_1 = analog_model[1].analog_tile.get_out_scaling_alpha().clone()

        self.train_model(analog_model, loss_func, x_b, y_b)

        learned_out_scaling_alpha_0 = analog_model[0].analog_tile.get_out_scaling_alpha().clone()
        learned_out_scaling_alpha_1 = analog_model[1].analog_tile.get_out_scaling_alpha().clone()

        self.assertIsNotNone(analog_model[0].analog_tile.get_out_scaling_alpha().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_alpha_0, learned_out_scaling_alpha_0)
        self.assertIsNotNone(analog_model[1].analog_tile.get_out_scaling_alpha().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_alpha_1, learned_out_scaling_alpha_1)

    def test_layer_instantiation(self):
        """Test AnalogConv2d layer instantiation."""
        model = self.get_layer(in_channels=2, out_channels=3, kernel_size=4)

        # Assert the number of elements of the weights.
        tile_weights, tile_biases = model.analog_tile.get_weights()

        self.assertEqual(tile_weights.numel(), 2*3*4*4)
        if model.analog_bias:
            self.assertEqual(tile_biases.numel(), 3)


@parametrize_over_layers(
    layers=[Conv3d, Conv3dCuda],
    tiles=[FloatingPoint, Inference],
    biases=['analog', 'digital', None]
)
class Convolution3dLayerTest(ConvolutionLayerTest):
    """Tests for AnalogConv3d layer."""

    digital_layer_cls = torch_Conv3d

    def test_torch_original_layer(self):
        """Test a single layer, having the digital layer as reference."""
        # This tests the forward pass
        model = self.get_digital_layer(in_channels=2, out_channels=3, kernel_size=4, padding=2)
        x = randn(3, 2, 4, 5, 6)

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
        y_b = randn(3, 3, 5, 5, 5)
        x_b = randn(3, 2, 4, 4, 4)

        if self.use_cuda:
            y_b = y_b.cuda()
            x_b = x_b.cuda()

        self.train_model(model, loss_func, x_b, y_b)
        self.train_model(analog_model, loss_func, x_b, y_b)

        weight, bias = self.get_weights_from_digital_model(analog_model, model)
        weight_analog, bias_analog = self.get_weights_from_analog_model(analog_model)

        self.assertTensorAlmostEqual(weight_analog, weight)
        if self.bias:
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
        y_b = randn(3, 3, 6, 6, 6)
        x_b = randn(3, 2, 4, 4, 4)

        if self.use_cuda:
            y_b = y_b.cuda()
            x_b = x_b.cuda()

        self.train_model(model, loss_func, x_b, y_b)
        self.train_model(analog_model, loss_func, x_b, y_b)

        for analog_layer, layer in zip(analog_model.children(), model.children()):
            weight, bias = self.get_weights_from_digital_model(analog_layer, layer)
            weight_analog, bias_analog = self.get_weights_from_analog_model(analog_layer)

            self.assertTensorAlmostEqual(weight_analog, weight)
            if self.bias:
                self.assertTensorAlmostEqual(bias_analog, bias)

    def test_out_scaling_alpha_learning(self):
        """Check if out scaling alpha are learning."""
        rpu_config = InferenceRPUConfig(mapping=MappingParameter(
            weight_scaling_omega=0.6,
            learn_out_scaling_alpha=True))

        analog_model = Sequential(
            self.get_layer(in_channels=2, out_channels=2, kernel_size=4,
                           padding=2, rpu_config=rpu_config),
            self.get_layer(in_channels=2, out_channels=3, kernel_size=4,
                           padding=2, rpu_config=rpu_config)
        )

        loss_func = mse_loss
        y_b = randn(3, 3, 6, 6, 6)
        x_b = randn(3, 2, 4, 4, 4)

        if self.use_cuda:
            y_b = y_b.cuda()
            x_b = x_b.cuda()

        initial_out_scaling_alpha_0 = analog_model[0].analog_tile.get_out_scaling_alpha().clone()
        initial_out_scaling_alpha_1 = analog_model[1].analog_tile.get_out_scaling_alpha().clone()

        self.train_model(analog_model, loss_func, x_b, y_b)

        learned_out_scaling_alpha_0 = analog_model[0].analog_tile.get_out_scaling_alpha().clone()
        learned_out_scaling_alpha_1 = analog_model[1].analog_tile.get_out_scaling_alpha().clone()

        self.assertEqual(initial_out_scaling_alpha_0.numel(), 1)
        self.assertIsNotNone(analog_model[0].analog_tile.get_out_scaling_alpha().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_alpha_0, learned_out_scaling_alpha_0)

        self.assertEqual(initial_out_scaling_alpha_1.numel(), 1)
        self.assertIsNotNone(analog_model[1].analog_tile.get_out_scaling_alpha().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_alpha_1, learned_out_scaling_alpha_1)

    def test_out_scaling_alpha_learning_columnwise(self):
        """Check if out scaling alpha are learning."""
        rpu_config = InferenceRPUConfig(mapping=MappingParameter(
            weight_scaling_omega=0.6,
            learn_out_scaling_alpha=True,
            weight_scaling_omega_columnwise=True))

        analog_model = Sequential(
            self.get_layer(in_channels=2, out_channels=2, kernel_size=4,
                           padding=2, rpu_config=rpu_config),
            self.get_layer(in_channels=2, out_channels=3, kernel_size=4,
                           padding=2, rpu_config=rpu_config)
        )

        loss_func = mse_loss
        y_b = randn(3, 3, 6, 6, 6)
        x_b = randn(3, 2, 4, 4, 4)

        if self.use_cuda:
            y_b = y_b.cuda()
            x_b = x_b.cuda()

        initial_out_scaling_alpha_0 = analog_model[0].analog_tile.get_out_scaling_alpha().clone()
        initial_out_scaling_alpha_1 = analog_model[1].analog_tile.get_out_scaling_alpha().clone()

        self.train_model(analog_model, loss_func, x_b, y_b)

        learned_out_scaling_alpha_0 = analog_model[0].analog_tile.get_out_scaling_alpha().clone()
        learned_out_scaling_alpha_1 = analog_model[1].analog_tile.get_out_scaling_alpha().clone()

        self.assertGreaterEqual(initial_out_scaling_alpha_0.numel(), 1)
        self.assertIsNotNone(analog_model[0].analog_tile.get_out_scaling_alpha().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_alpha_0, learned_out_scaling_alpha_0)

        self.assertGreaterEqual(initial_out_scaling_alpha_1.numel(), 1)
        self.assertIsNotNone(analog_model[1].analog_tile.get_out_scaling_alpha().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_alpha_1, learned_out_scaling_alpha_1)

    def test_layer_instantiation(self):
        """Test AnalogConv2d layer instantiation."""
        model = self.get_layer(in_channels=2, out_channels=3, kernel_size=4)

        # Assert the number of elements of the weights.
        tile_weights, tile_biases = model.analog_tile.get_weights()

        self.assertEqual(tile_weights.numel(), 2*3*4*4*4)
        if model.analog_bias:
            self.assertEqual(tile_biases.numel(), 3)

# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for layer abstractions."""

from unittest import SkipTest

import pytest
from torch import randn
from torch.nn import (
    Conv1d as torch_Conv1d,
    Conv2d as torch_Conv2d,
    Conv3d as torch_Conv3d,
    Sequential,
)
from torch.nn.functional import mse_loss

from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.configs import InferenceRPUConfig
from aihwkit.simulator.parameters import (
    MappingParameter,
    IOParameters,
    WeightModifierParameter,
    WeightModifierType,
)
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.inference.noise.custom import StateIndependentNoiseModel
from aihwkit.nn.conversion import convert_to_analog

from .helpers.decorators import parametrize_over_layers
from .helpers.layers import Conv1d, Conv1dCuda, Conv2d, Conv2dCuda, Conv3d, Conv3dCuda
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import FloatingPoint, Inference, TorchInference, Custom


class ConvolutionLayerTest(ParametrizedTestCase):
    """Generic class for helping testing analog convolution layers."""

    digital_layer_cls = torch_Conv1d

    def get_digital_layer(
        self, in_channels=2, out_channels=3, kernel_size=4, padding=2
    ):
        """Return a digital layer."""
        layer = self.digital_layer_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=self.bias,
        )
        if self.use_cuda:
            layer = layer.cuda()

        return layer

    def set_weights_from_digital_model(self, analog_model, digital_model):
        """Set the analog model weights based on the digital model."""
        weights, biases = self.get_weights_from_digital_model(
            analog_model, digital_model
        )
        analog_model.set_weights(weights, biases)

    @staticmethod
    def get_weights_from_digital_model(analog_model, digital_model):
        """Set the analog model weights based on the digital model."""
        weights = (
            digital_model.weight.data.detach()
            .reshape([analog_model.out_features, analog_model.in_features])
            .cpu()
        )
        biases = None
        if digital_model.bias is not None:
            biases = digital_model.bias.data.detach().cpu()

        return weights, biases

    @staticmethod
    def get_weights_from_analog_model(analog_model):
        """Set the analog model weights based on the digital model."""
        weights, biases = analog_model.get_weights()
        return weights, biases

    @staticmethod
    def train_model(model, loss_func, x_b, y_b):
        """Train the model."""
        opt = AnalogSGD(model.parameters(), lr=0.1)

        epochs = 10
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(x_b)
            loss = loss_func(pred, y_b)
            loss.backward()
            opt.step()

    def base_test_inference_modifier(self, torch_model, x_b):
        """tests whether modifier are used"""

        rpu_config = InferenceRPUConfig(
            mapping=MappingParameter(
                weight_scaling_omega=0.0,
                learn_out_scaling=False,
                weight_scaling_columnwise=False,
            ),
            modifier=WeightModifierParameter(
                type=WeightModifierType.ADD_NORMAL, std_dev=1.0
            ),
            forward=IOParameters(is_perfect=True),
        )

        model = convert_to_analog(torch_model, rpu_config)

        if self.use_cuda:
            x_b = x_b.cuda()
            model = model.cuda()

        opt = AnalogSGD(model.parameters(), lr=0.0)
        opt.step()

        model.eval()
        y_eval1 = model(x_b)
        y_eval2 = model(x_b)

        model.train()
        y_train1 = model(x_b)
        opt.step()
        y_train2 = model(x_b)

        self.assertNotAlmostEqualTensor(y_train1, y_train2)
        self.assertTensorAlmostEqual(y_eval2, y_eval1)

    def base_test_drift_compensation(self, torch_model, x_b):
        """tests whether drift compensation is performed"""

        rpu_config = InferenceRPUConfig(
            mapping=MappingParameter(
                weight_scaling_omega=0.0,
                learn_out_scaling=False,
                weight_scaling_columnwise=False,
            ),
            forward=IOParameters(is_perfect=True),
            drift_compensation=GlobalDriftCompensation(),
            noise_model=StateIndependentNoiseModel(
                prog_noise_scale=0.0,
                read_noise_scale=0.0,
                drift_nu_std=0.0,
                drift_nu_mean=0.1,
            ),
        )

        model = convert_to_analog(torch_model, rpu_config)

        rpu_config.drift_compensation = None
        model_without = convert_to_analog(torch_model, rpu_config)

        if self.use_cuda:
            x_b = x_b.cuda()
            model = model.cuda()
            model_without = model_without.cuda()

        model.eval()
        y_before = model(x_b)
        model.drift_analog_weights(1000.0)
        y_after = model(x_b)

        model_without.eval()
        y_without_before = model_without(x_b)
        model_without.drift_analog_weights(1000.0)
        y_without_after = model_without(x_b)

        self.assertTensorAlmostEqual(y_before, y_without_before)
        self.assertTensorAlmostEqual(y_before, y_after)
        self.assertNotAlmostEqualTensor(y_after, y_without_after)
        self.assertNotAlmostEqualTensor(y_without_before, y_without_after)


@parametrize_over_layers(
    layers=[Conv1d, Conv1dCuda],
    tiles=[FloatingPoint, Inference],
    biases=["analog", "digital", None],
)
class Convolution1dLayerTest(ConvolutionLayerTest):
    """Tests for AnalogConv1d layer."""

    digital_layer_cls = torch_Conv1d

    def test_torch_original_layer(self):
        """Test a single layer, having the digital layer as reference."""
        # This tests the forward pass
        model = self.get_digital_layer(
            in_channels=2, out_channels=3, kernel_size=4, padding=2
        )
        x = randn(3, 2, 4)

        if self.use_cuda:
            x = x.cuda()

        y = model(x)

        analog_model = self.get_layer(
            in_channels=2, out_channels=3, kernel_size=4, padding=2
        )
        self.set_weights_from_digital_model(analog_model, model)

        y_analog = analog_model(x)
        self.assertTensorAlmostEqual(y_analog, y)

    def test_torch_train_original_layer(self):
        """Test the forward and update pass, having the digital layer as reference."""
        model = self.get_digital_layer(
            in_channels=2, out_channels=3, kernel_size=4, padding=2
        )
        analog_model = self.get_layer(
            in_channels=2, out_channels=3, kernel_size=4, padding=2
        )
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
            self.get_digital_layer(
                in_channels=2, out_channels=2, kernel_size=4, padding=2
            ),
            self.get_digital_layer(
                in_channels=2, out_channels=3, kernel_size=4, padding=2
            ),
        )

        analog_model = Sequential(
            self.get_layer(in_channels=2, out_channels=2, kernel_size=4, padding=2),
            self.get_layer(in_channels=2, out_channels=3, kernel_size=4, padding=2),
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
            weight_analog, bias_analog = self.get_weights_from_analog_model(
                analog_layer
            )

            self.assertTensorAlmostEqual(weight_analog, weight)
            if self.bias:
                self.assertTensorAlmostEqual(bias_analog, bias)

    @pytest.mark.skip(reason="This test is randomly failing.")
    def test_out_scaling_learning(self):
        """Check if out scaling are learning."""
        rpu_config = InferenceRPUConfig(
            mapping=MappingParameter(
                learn_out_scaling=True, out_scaling_columnwise=False
            )
        )

        analog_model = Sequential(
            self.get_layer(
                in_channels=2,
                out_channels=2,
                kernel_size=4,
                padding=2,
                rpu_config=rpu_config,
            ),
            self.get_layer(
                in_channels=2,
                out_channels=3,
                kernel_size=4,
                padding=2,
                rpu_config=rpu_config,
            ),
        )

        loss_func = mse_loss
        y_b = randn(3, 3, 6)
        x_b = randn(3, 2, 4)

        if self.use_cuda:
            y_b = y_b.cuda()
            x_b = x_b.cuda()

        analog_tile_0 = next(analog_model[0].analog_tiles())
        analog_tile_1 = next(analog_model[1].analog_tiles())

        initial_out_scaling_0 = analog_tile_0.get_learned_out_scales().clone()
        initial_out_scaling_1 = analog_tile_1.get_learned_out_scales().clone()
        self.assertEqual(initial_out_scaling_0.numel(), 1)
        self.assertEqual(initial_out_scaling_1.numel(), 1)

        self.train_model(analog_model, loss_func, x_b, y_b)

        learned_out_scaling_0 = analog_tile_0.get_learned_out_scales().clone()
        learned_out_scaling_1 = analog_tile_1.get_learned_out_scales().clone()

        self.assertIsNotNone(analog_tile_0.get_learned_out_scales().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_0, learned_out_scaling_0)
        self.assertIsNotNone(analog_tile_1.get_learned_out_scales().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_1, learned_out_scaling_1)

    @pytest.mark.skip(reason="This test is randomly failing.")
    def test_out_scaling_learning_columnwise(self):
        """Check if out scaling alpha are learning."""
        rpu_config = InferenceRPUConfig(
            mapping=MappingParameter(
                weight_scaling_omega=0.6,
                learn_out_scaling=True,
                weight_scaling_columnwise=True,
            )
        )

        analog_model = Sequential(
            self.get_layer(
                in_channels=2,
                out_channels=2,
                kernel_size=4,
                padding=2,
                rpu_config=rpu_config,
            ),
            self.get_layer(
                in_channels=2,
                out_channels=3,
                kernel_size=4,
                padding=2,
                rpu_config=rpu_config,
            ),
        )

        loss_func = mse_loss
        y_b = randn(3, 3, 6)
        x_b = randn(3, 2, 4)

        if self.use_cuda:
            y_b = y_b.cuda()
            x_b = x_b.cuda()

        analog_tile_0 = next(analog_model[0].analog_tiles())
        analog_tile_1 = next(analog_model[1].analog_tiles())

        initial_out_scaling_0 = analog_tile_0.get_learned_out_scales().clone()
        initial_out_scaling_1 = analog_tile_1.get_learned_out_scales().clone()

        self.train_model(analog_model, loss_func, x_b, y_b)

        learned_out_scaling_0 = analog_tile_0.get_learned_out_scales().clone()
        learned_out_scaling_1 = analog_tile_1.get_learned_out_scales().clone()

        self.assertIsNotNone(analog_tile_0.get_learned_out_scales().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_0, learned_out_scaling_0)
        self.assertIsNotNone(analog_tile_1.get_learned_out_scales().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_1, learned_out_scaling_1)

    def test_layer_instantiation(self):
        """Test AnalogConv2d layer instantiation."""
        model = self.get_layer(in_channels=2, out_channels=3, kernel_size=4)

        # Assert the number of elements of the weights.
        tile_weights, tile_biases = model.get_weights()

        self.assertEqual(tile_weights.numel(), 2 * 3 * 4)
        if next(model.analog_tiles()).analog_bias:
            self.assertEqual(tile_biases.numel(), 3)


@parametrize_over_layers(
    layers=[Conv1d, Conv1dCuda], tiles=[Inference], biases=["digital"]
)
class Convolution1dLayerTestInference(ConvolutionLayerTest):
    """Tests for AnalogConv1d layer specific for inference."""

    digital_layer_cls = torch_Conv1d

    def test_drift_compensation(self):
        """tests the drift compensation"""

        x_b = randn(3, 2, 4)
        torch_model = self.get_digital_layer(
            in_channels=2, out_channels=2, kernel_size=4, padding=2
        )
        self.base_test_drift_compensation(torch_model, x_b)

    def test_inference_modifier(self):
        """tests the modifier function"""
        x_b = randn(3, 2, 4)
        torch_model = self.get_digital_layer(
            in_channels=2, out_channels=2, kernel_size=4, padding=2
        )
        self.base_test_inference_modifier(torch_model, x_b)


@parametrize_over_layers(
    layers=[Conv2d, Conv2dCuda],
    tiles=[FloatingPoint, Inference, TorchInference, Custom],
    biases=["analog", "digital", None],
)
class Convolution2dLayerTest(ConvolutionLayerTest):
    """Tests for AnalogConv2d layer."""

    digital_layer_cls = torch_Conv2d

    def test_torch_original_layer(self):
        """Test a single layer, having the digital layer as reference."""
        # This tests the forward pass
        model = self.get_digital_layer(
            in_channels=2, out_channels=3, kernel_size=4, padding=2
        )
        x = randn(3, 2, 4, 4)

        if self.use_cuda:
            x = x.cuda()

        y = model(x)

        analog_model = self.get_layer(
            in_channels=2, out_channels=3, kernel_size=4, padding=2
        )
        self.set_weights_from_digital_model(analog_model, model)

        y_analog = analog_model(x)
        self.assertTensorAlmostEqual(y_analog, y)

    def test_torch_original_layer_indexed(self):
        """Test a single layer, having the digital layer as reference."""
        # This tests the forward pass
        if not self.get_rpu_config().tile_class.supports_indexed:
            raise SkipTest("Indexed not supported")

        model = self.get_digital_layer(
            in_channels=2, out_channels=3, kernel_size=4, padding=2
        )
        x = randn(3, 2, 4, 4)

        if self.use_cuda:
            x = x.cuda()

        y = model(x)

        analog_model = self.get_layer(
            in_channels=2, out_channels=3, kernel_size=4, padding=2
        )
        analog_model.use_indexed = True
        self.set_weights_from_digital_model(analog_model, model)

        y_analog = analog_model(x)
        self.assertTensorAlmostEqual(y_analog, y)

    def test_torch_original_layer_not_indexed(self):
        """Test a single layer, having the digital layer as reference."""
        # This tests the forward pass
        model = self.get_digital_layer(
            in_channels=2, out_channels=3, kernel_size=4, padding=2
        )
        x = randn(3, 2, 4, 4)

        if self.use_cuda:
            x = x.cuda()

        y = model(x)

        analog_model = self.get_layer(
            in_channels=2, out_channels=3, kernel_size=4, padding=2
        )
        analog_model.use_indexed = False
        self.set_weights_from_digital_model(analog_model, model)

        y_analog = analog_model(x)
        self.assertTensorAlmostEqual(y_analog, y)

    def test_torch_train_original_layer(self):
        """Test the forward and update pass, having the digital layer as reference."""
        model = self.get_digital_layer(
            in_channels=2, out_channels=3, kernel_size=4, padding=2
        )
        analog_model = self.get_layer(
            in_channels=2, out_channels=3, kernel_size=4, padding=2
        )
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
            self.get_digital_layer(
                in_channels=2, out_channels=2, kernel_size=4, padding=2
            ),
            self.get_digital_layer(
                in_channels=2, out_channels=3, kernel_size=4, padding=2
            ),
        )

        analog_model = Sequential(
            self.get_layer(in_channels=2, out_channels=2, kernel_size=4, padding=2),
            self.get_layer(in_channels=2, out_channels=3, kernel_size=4, padding=2),
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

            weight_analog, bias_analog = self.get_weights_from_analog_model(
                analog_layer
            )

            self.assertTensorAlmostEqual(weight_analog, weight)
            if self.bias:
                self.assertTensorAlmostEqual(bias_analog, bias)

    @pytest.mark.skip(reason="This test is randomly failing.")
    def test_out_scaling_learning(self):
        """Check if out scaling alpha are learning."""
        rpu_config = InferenceRPUConfig(
            mapping=MappingParameter(weight_scaling_omega=0.6, learn_out_scaling=True)
        )

        analog_model = Sequential(
            self.get_layer(
                in_channels=2,
                out_channels=2,
                kernel_size=4,
                padding=2,
                rpu_config=rpu_config,
            ),
            self.get_layer(
                in_channels=2,
                out_channels=3,
                kernel_size=4,
                padding=2,
                rpu_config=rpu_config,
            ),
        )

        loss_func = mse_loss
        y_b = randn(3, 3, 6, 6)
        x_b = randn(3, 2, 4, 4)

        if self.use_cuda:
            y_b = y_b.cuda()
            x_b = x_b.cuda()

        analog_tile_0 = next(analog_model[0].analog_tiles())
        analog_tile_1 = next(analog_model[1].analog_tiles())

        initial_out_scaling_0 = analog_tile_0.get_learned_out_scales().clone()
        initial_out_scaling_1 = analog_tile_1.get_learned_out_scales().clone()

        self.train_model(analog_model, loss_func, x_b, y_b)

        learned_out_scaling_0 = analog_tile_0.get_learned_out_scales().clone()
        learned_out_scaling_1 = analog_tile_1.get_learned_out_scales().clone()

        self.assertIsNotNone(analog_tile_0.get_learned_out_scales().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_0, learned_out_scaling_0)
        self.assertIsNotNone(analog_tile_1.get_learned_out_scales().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_1, learned_out_scaling_1)

    @pytest.mark.skip(reason="This test is randomly failing.")
    def test_out_scaling_learning_columnwise(self):
        """Check if out scaling alpha are learning."""
        rpu_config = InferenceRPUConfig(
            mapping=MappingParameter(
                weight_scaling_omega=0.6,
                learn_out_scaling=True,
                weight_scaling_columnwise=True,
            )
        )

        analog_model = Sequential(
            self.get_layer(
                in_channels=2,
                out_channels=2,
                kernel_size=4,
                padding=2,
                rpu_config=rpu_config,
            ),
            self.get_layer(
                in_channels=2,
                out_channels=3,
                kernel_size=4,
                padding=2,
                rpu_config=rpu_config,
            ),
        )

        loss_func = mse_loss
        y_b = randn(3, 3, 6, 6)
        x_b = randn(3, 2, 4, 4)

        if self.use_cuda:
            y_b = y_b.cuda()
            x_b = x_b.cuda()

        analog_tile_0 = next(analog_model[0].analog_tiles())
        analog_tile_1 = next(analog_model[1].analog_tiles())

        initial_out_scaling_0 = (
            analog_tile_0.get_learned_out_scales().clone()
        )  # pylint: disable=unused-variable
        initial_out_scaling_1 = (
            analog_tile_1.get_learned_out_scales().clone()
        )  # pylint: disable=unused-variable

        self.train_model(analog_model, loss_func, x_b, y_b)

        learned_out_scaling_0 = (
            analog_tile_0.get_learned_out_scales().clone()
        )  # pylint: disable=unused-variable
        learned_out_scaling_1 = (
            analog_tile_1.get_learned_out_scales().clone()
        )  # pylint: disable=unused-variable

        self.assertIsNotNone(analog_tile_0.get_learned_out_scales().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_0, learned_out_scaling_0)
        self.assertIsNotNone(analog_tile_1.get_learned_out_scales().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_1, learned_out_scaling_1)

    def test_layer_instantiation(self):
        """Test AnalogConv2d layer instantiation."""
        model = self.get_layer(in_channels=2, out_channels=3, kernel_size=4)

        # Assert the number of elements of the weights.
        tile_weights, tile_biases = model.get_weights()

        self.assertEqual(tile_weights.numel(), 2 * 3 * 4 * 4)
        if next(model.analog_tiles()).analog_bias:
            self.assertEqual(tile_biases.numel(), 3)


@parametrize_over_layers(
    layers=[Conv2d, Conv2dCuda], tiles=[Inference, TorchInference], biases=["digital"]
)
class Convolution2dLayerTestInference(ConvolutionLayerTest):
    """Tests for AnalogConv2d layer specific for infernence."""

    digital_layer_cls = torch_Conv2d

    def test_drift_compensation(self):
        """tests the drift compensation"""
        x_b = randn(3, 2, 4, 4)
        torch_model = self.get_digital_layer(
            in_channels=2, out_channels=2, kernel_size=4, padding=2
        )
        self.base_test_drift_compensation(torch_model, x_b)

    def test_inference_modifier(self):
        """tests the modifier function"""
        x_b = randn(3, 2, 4, 4)
        torch_model = self.get_digital_layer(
            in_channels=2, out_channels=2, kernel_size=4, padding=2
        )
        self.base_test_inference_modifier(torch_model, x_b)


@parametrize_over_layers(
    layers=[Conv3d, Conv3dCuda],
    tiles=[FloatingPoint, Inference],
    biases=["analog", "digital", None],
)
class Convolution3dLayerTest(ConvolutionLayerTest):
    """Tests for AnalogConv3d layer."""

    digital_layer_cls = torch_Conv3d

    def test_torch_original_layer(self):
        """Test a single layer, having the digital layer as reference."""
        # This tests the forward pass
        model = self.get_digital_layer(
            in_channels=2, out_channels=3, kernel_size=4, padding=2
        )
        x = randn(3, 2, 4, 5, 6)

        if self.use_cuda:
            x = x.cuda()

        y = model(x)

        analog_model = self.get_layer(
            in_channels=2, out_channels=3, kernel_size=4, padding=2
        )
        self.set_weights_from_digital_model(analog_model, model)

        y_analog = analog_model(x)
        self.assertTensorAlmostEqual(y_analog, y)

    def test_torch_train_original_layer(self):
        """Test the forward and update pass, having the digital layer as reference."""
        model = self.get_digital_layer(
            in_channels=2, out_channels=3, kernel_size=4, padding=2
        )
        analog_model = self.get_layer(
            in_channels=2, out_channels=3, kernel_size=4, padding=2
        )
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
            self.get_digital_layer(
                in_channels=2, out_channels=2, kernel_size=4, padding=2
            ),
            self.get_digital_layer(
                in_channels=2, out_channels=3, kernel_size=4, padding=2
            ),
        )

        analog_model = Sequential(
            self.get_layer(in_channels=2, out_channels=2, kernel_size=4, padding=2),
            self.get_layer(in_channels=2, out_channels=3, kernel_size=4, padding=2),
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
            weight_analog, bias_analog = self.get_weights_from_analog_model(
                analog_layer
            )

            self.assertTensorAlmostEqual(weight_analog, weight)
            if self.bias:
                self.assertTensorAlmostEqual(bias_analog, bias)

    @pytest.mark.skip(reason="This test is randomly failing.")
    def test_out_scaling_learning(self):
        """Check if out scaling alpha are learning."""
        rpu_config = InferenceRPUConfig(
            mapping=MappingParameter(
                learn_out_scaling=True, out_scaling_columnwise=False
            )
        )

        analog_model = Sequential(
            self.get_layer(
                in_channels=2,
                out_channels=2,
                kernel_size=4,
                padding=2,
                rpu_config=rpu_config,
            ),
            self.get_layer(
                in_channels=2,
                out_channels=3,
                kernel_size=4,
                padding=2,
                rpu_config=rpu_config,
            ),
        )

        loss_func = mse_loss
        y_b = randn(3, 3, 6, 6, 6)
        x_b = randn(3, 2, 4, 4, 4)

        if self.use_cuda:
            y_b = y_b.cuda()
            x_b = x_b.cuda()

        analog_tile_0 = next(analog_model[0].analog_tiles())
        analog_tile_1 = next(analog_model[1].analog_tiles())

        initial_out_scaling_0 = analog_tile_0.get_learned_out_scales().clone()
        initial_out_scaling_1 = analog_tile_1.get_learned_out_scales().clone()

        self.train_model(analog_model, loss_func, x_b, y_b)

        learned_out_scaling_0 = analog_tile_0.get_learned_out_scales().clone()
        learned_out_scaling_1 = analog_tile_1.get_learned_out_scales().clone()

        self.assertEqual(initial_out_scaling_0.numel(), 1)
        self.assertIsNotNone(analog_tile_0.get_learned_out_scales().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_0, learned_out_scaling_0)

        self.assertEqual(initial_out_scaling_1.numel(), 1)
        self.assertIsNotNone(analog_tile_1.get_learned_out_scales().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_1, learned_out_scaling_1)

    @pytest.mark.skip(reason="This test is randomly failing.")
    def test_out_scaling_learning_columnwise(self):
        """Check if out scaling alpha are learning."""
        rpu_config = InferenceRPUConfig(
            mapping=MappingParameter(
                learn_out_scaling=True, out_scaling_columnwise=True
            )
        )

        analog_model = Sequential(
            self.get_layer(
                in_channels=2,
                out_channels=2,
                kernel_size=4,
                padding=2,
                rpu_config=rpu_config,
            ),
            self.get_layer(
                in_channels=2,
                out_channels=3,
                kernel_size=4,
                padding=2,
                rpu_config=rpu_config,
            ),
        )

        loss_func = mse_loss
        y_b = randn(3, 3, 6, 6, 6)
        x_b = randn(3, 2, 4, 4, 4)

        if self.use_cuda:
            y_b = y_b.cuda()
            x_b = x_b.cuda()

        analog_tile_0 = next(analog_model[0].analog_tiles())
        analog_tile_1 = next(analog_model[1].analog_tiles())

        initial_out_scaling_0 = analog_tile_0.get_learned_out_scales().clone()
        initial_out_scaling_1 = analog_tile_1.get_learned_out_scales().clone()

        self.train_model(analog_model, loss_func, x_b, y_b)

        learned_out_scaling_0 = analog_tile_0.get_learned_out_scales().clone()
        learned_out_scaling_1 = analog_tile_1.get_learned_out_scales().clone()

        self.assertGreaterEqual(initial_out_scaling_0.numel(), 1)
        self.assertIsNotNone(analog_tile_0.get_learned_out_scales().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_0, learned_out_scaling_0)

        self.assertGreaterEqual(initial_out_scaling_1.numel(), 1)
        self.assertIsNotNone(analog_tile_1.get_learned_out_scales().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_1, learned_out_scaling_1)

    def test_layer_instantiation(self):
        """Test AnalogConv2d layer instantiation."""
        model = self.get_layer(in_channels=2, out_channels=3, kernel_size=4)

        # Assert the number of elements of the weights.
        tile_weights, tile_biases = model.get_weights()

        self.assertEqual(tile_weights.numel(), 2 * 3 * 4 * 4 * 4)
        if next(model.analog_tiles()).analog_bias:
            self.assertEqual(tile_biases.numel(), 3)


@parametrize_over_layers(
    layers=[Conv3d, Conv3dCuda], tiles=[Inference], biases=["digital"]
)
class Convolution3dLayerTestInference(ConvolutionLayerTest):
    """Tests for AnalogConv2d layer specific for infernence."""

    digital_layer_cls = torch_Conv3d

    def test_drift_compensation(self):
        """tests the drift compensation"""
        x_b = randn(3, 2, 4, 4, 4)
        torch_model = self.get_digital_layer(
            in_channels=2, out_channels=2, kernel_size=4, padding=2
        )
        self.base_test_drift_compensation(torch_model, x_b)

    def test_inference_modifier(self):
        """tests the modifier function"""
        x_b = randn(3, 2, 4, 4, 4)
        torch_model = self.get_digital_layer(
            in_channels=2, out_channels=2, kernel_size=4, padding=2
        )
        self.base_test_inference_modifier(torch_model, x_b)

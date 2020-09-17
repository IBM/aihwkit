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

"""Tests for general functionality of layers."""

from unittest import TestCase, skipIf

from numpy.testing import assert_array_almost_equal, assert_raises

from torch import Tensor

from aihwkit.nn.modules.conv import AnalogConv2d
from aihwkit.nn.modules.linear import AnalogLinear

from aihwkit.simulator.devices import ConstantStepResistiveDevice
from aihwkit.simulator.parameters import ConstantStepResistiveDeviceParameters
from aihwkit.simulator.rpu_base import cuda


class LayerTestMixin:
    """Helper for layer tests."""

    def get_layer(self, **kwargs):
        """Return a layer."""
        raise NotImplementedError

    def assertTensorAlmostEqual(self, tensor_a, tensor_b):
        """Assert that two tensors are almost equal."""
        # pylint: disable=invalid-name
        array_a = tensor_a.detach().cpu().numpy()
        array_b = tensor_b.detach().cpu().numpy()
        assert_array_almost_equal(array_a, array_b)

    def assertNotAlmostEqualTensor(self, tensor_a, tensor_b):
        """Assert that two tensors are not equal."""
        # pylint: disable=invalid-name
        assert_raises(AssertionError, self.assertTensorAlmostEqual, tensor_a, tensor_b)

    def test_realistic_weights_bias(self):
        """Test using realistic weights (bias)."""
        layer = self.get_layer(realistic_read_write=True,
                               bias=True)

        shape = layer.weight.shape
        # Check that the tile weights are equal from the layer weights, as
        # the weights are synced after being set.
        tile_weights, tile_biases = layer.analog_tile.get_weights()
        self.assertTensorAlmostEqual(layer.weight, tile_weights.reshape(shape))
        self.assertTensorAlmostEqual(layer.bias, tile_biases)

        # 1. Set the layer weights and biases.
        user_weights = Tensor(layer.out_features, layer.in_features).uniform_(-0.5, 0.5)
        user_biases = Tensor(layer.out_features).uniform_(-0.5, 0.5)
        layer.set_weights(user_weights, user_biases)

        # Check that the tile weights are equal from the layer weights, as
        # the weights are synced after being set.
        tile_weights, tile_biases = layer.analog_tile.get_weights()
        self.assertTensorAlmostEqual(layer.weight, tile_weights.reshape(shape))
        self.assertTensorAlmostEqual(layer.bias, tile_biases)

        # Check that the tile weights are different than the user-specified
        # weights, as it is realistic.
        self.assertNotAlmostEqualTensor(user_weights, tile_weights.reshape(shape))
        self.assertNotAlmostEqualTensor(user_biases, tile_biases)

        # 2. Get the layer weights and biases.
        gotten_weights, gotten_biases = layer.get_weights()

        # Check that the tile weights are different than the gotten
        # weights, as it is realistic.
        self.assertNotAlmostEqualTensor(gotten_weights, tile_weights.reshape(shape))
        self.assertNotAlmostEqualTensor(gotten_biases, tile_biases)

    def test_not_realistic_weights_bias(self):
        """Test using non realistic weights (bias)."""
        layer = self.get_layer(realistic_read_write=False, bias=True)

        shape = layer.weight.shape
        # Check that the tile weights are equal from the layer weights, as
        # the weights are synced after being set.
        tile_weights, tile_biases = layer.analog_tile.get_weights()
        self.assertTensorAlmostEqual(layer.weight, tile_weights.reshape(shape))
        self.assertTensorAlmostEqual(layer.bias, tile_biases)

        # 1. Set the layer weights and biases.
        user_weights = Tensor(layer.out_features, layer.in_features).uniform_(-0.5, 0.5)
        user_biases = Tensor(layer.out_features).uniform_(-0.5, 0.5)
        layer.set_weights(user_weights, user_biases)

        # Check that the tile weights are equal from the layer weights, as
        # the weights are synced after being set.
        tile_weights, tile_biases = layer.analog_tile.get_weights()
        self.assertTensorAlmostEqual(layer.weight, tile_weights.reshape(shape))
        self.assertTensorAlmostEqual(layer.bias, tile_biases)

        # Check that the tile weights are equal to the user-specified
        # weights, as it is not realistic.
        self.assertTensorAlmostEqual(user_weights, tile_weights)
        self.assertTensorAlmostEqual(user_biases, tile_biases)

        # 2. Get the layer weights and biases.
        gotten_weights, gotten_biases = layer.get_weights()

        # Check that the tile weights are equal than the gotten
        # weights, as it is not realistic.
        self.assertTensorAlmostEqual(gotten_weights, tile_weights)
        self.assertTensorAlmostEqual(gotten_biases, tile_biases)

    def test_realistic_weights_no_bias(self):
        """Test using realistic weights (no bias)."""
        layer = self.get_layer(realistic_read_write=True,
                               bias=False)

        shape = layer.weight.shape

        # Check that the tile weights are equal from the layer weights, as
        # the weights are synced after being set.
        tile_weights, _ = layer.analog_tile.get_weights()
        self.assertTensorAlmostEqual(layer.weight, tile_weights.reshape(shape))

        # 1. Set the layer weights and biases.
        user_weights = Tensor(layer.out_features, layer.in_features).uniform_(-0.5, 0.5)
        layer.set_weights(user_weights)

        # Check that the tile weights are equal from the layer weights, as
        # the weights are synced after being set.
        tile_weights, _ = layer.analog_tile.get_weights()
        self.assertTensorAlmostEqual(layer.weight, tile_weights.reshape(shape))

        # Check that the tile weights are different than the user-specified
        # weights, as it is realistic.
        self.assertNotAlmostEqualTensor(user_weights, tile_weights)

        # 2. Get the layer weights and biases.
        gotten_weights, _ = layer.get_weights()

        # Check that the tile weights are different than the gotten
        # weights, as it is realistic.
        self.assertNotAlmostEqualTensor(gotten_weights, tile_weights)

    def test_not_realistic_weights_no_bias(self):
        """Test using non realistic weights (no bias)."""
        layer = self.get_layer(realistic_read_write=False,
                               bias=False)
        shape = layer.weight.shape

        # Check that the tile weights are equal from the layer weights, as
        # the weights are synced after being set.
        tile_weights, _ = layer.analog_tile.get_weights()
        self.assertTensorAlmostEqual(layer.weight, tile_weights.reshape(shape))

        # 1. Set the layer weights and biases.
        user_weights = Tensor(layer.out_features, layer.in_features).uniform_(-0.5, 0.5)
        layer.set_weights(user_weights)

        # Check that the tile weights are equal from the layer weights, as
        # the weights are synced after being set.
        tile_weights, _ = layer.analog_tile.get_weights()
        self.assertTensorAlmostEqual(layer.weight, tile_weights.reshape(shape))

        # Check that the tile weights are equal to the user-specified
        # weights, as it is not realistic.
        self.assertTensorAlmostEqual(user_weights, tile_weights)

        # 2. Get the layer weights and biases.
        gotten_weights, _ = layer.get_weights()

        # Check that the tile weights are equal than the gotten
        # weights, as it is not realistic.
        self.assertTensorAlmostEqual(gotten_weights, tile_weights)


class Conv2DTest(LayerTestMixin, TestCase):
    """Test AnalogConv2d."""

    def get_layer(self, in_channels=2, out_channels=3, kernel_size=4, **kwargs):
        """Return a layer."""
        # pylint: disable=arguments-differ
        return AnalogConv2d(
            in_channels, out_channels, kernel_size,
            resistive_device=ConstantStepResistiveDevice(
                ConstantStepResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0)),
            **kwargs)


class LinearTest(LayerTestMixin, TestCase):
    """Test AnalogLinear."""

    def get_layer(self, cols=3, rows=4, **kwargs):
        """Return a layer."""
        # pylint: disable=arguments-differ
        return AnalogLinear(
            cols, rows,
            resistive_device=ConstantStepResistiveDevice(
                ConstantStepResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0)),
            **kwargs)


@skipIf(not cuda.is_compiled(), 'not compiled with CUDA support')
class CudaConv2DTest(LayerTestMixin, TestCase):
    """Test AnalogConv2d (cuda)."""

    def get_layer(self, in_channels=2, out_channels=3, kernel_size=4, **kwargs):
        """Return a layer."""
        # pylint: disable=arguments-differ
        return AnalogConv2d(
            in_channels, out_channels, kernel_size,
            resistive_device=ConstantStepResistiveDevice(
                ConstantStepResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0)),
            **kwargs).cuda()


@skipIf(not cuda.is_compiled(), 'not compiled with CUDA support')
class CudaLinearTest(LayerTestMixin, TestCase):
    """Test AnalogLinear (cuda)."""

    def get_layer(self, cols=3, rows=4, **kwargs):
        """Return a layer."""
        # pylint: disable=arguments-differ
        return AnalogLinear(
            cols, rows,
            resistive_device=ConstantStepResistiveDevice(
                ConstantStepResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0)),
            **kwargs).cuda()

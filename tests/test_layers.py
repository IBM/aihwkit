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

from unittest import SkipTest

from torch import Tensor

from aihwkit.nn import AnalogSequential
from aihwkit.simulator.tiles import AnalogTile, CudaAnalogTile
from aihwkit.simulator.rpu_base import cuda

from .helpers.decorators import parametrize_over_layers
from .helpers.layers import Linear, Conv2d, LinearCuda, Conv2dCuda
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import ConstantStep


@parametrize_over_layers(
    layers=[Linear, Conv2d, LinearCuda, Conv2dCuda],
    tiles=[ConstantStep],
    biases=[True, False]
)
class AnalogLayerTest(ParametrizedTestCase):
    """Analog layers abstraction tests."""

    def test_realistic_weights(self):
        """Test using realistic weights."""
        layer = self.get_layer(realistic_read_write=True)

        shape = layer.weight.shape
        # Check that the tile weights are equal from the layer weights, as
        # the weights are synced after being set.
        tile_weights, tile_biases = layer.analog_tile.get_weights()
        self.assertTensorAlmostEqual(layer.weight, tile_weights.reshape(shape))
        if self.bias:
            self.assertTensorAlmostEqual(layer.bias, tile_biases)

        # 1. Set the layer weights and biases.
        user_weights = Tensor(layer.out_features, layer.in_features).uniform_(-0.5, 0.5)
        user_biases = Tensor(layer.out_features).uniform_(-0.5, 0.5)
        layer.set_weights(user_weights, user_biases)

        # Check that the tile weights are equal from the layer weights, as
        # the weights are synced after being set.
        tile_weights, tile_biases = layer.analog_tile.get_weights()
        self.assertTensorAlmostEqual(layer.weight, tile_weights.reshape(shape))
        if self.bias:
            self.assertTensorAlmostEqual(layer.bias, tile_biases)

        # Check that the tile weights are different than the user-specified
        # weights, as it is realistic.
        self.assertNotAlmostEqualTensor(user_weights, tile_weights.reshape(shape))
        if self.bias:
            self.assertNotAlmostEqualTensor(user_biases, tile_biases)

        # 2. Get the layer weights and biases.
        gotten_weights, gotten_biases = layer.get_weights()

        # Check that the tile weights are different than the gotten
        # weights, as it is realistic.
        self.assertNotAlmostEqualTensor(gotten_weights, tile_weights.reshape(shape))
        if self.bias:
            self.assertNotAlmostEqualTensor(gotten_biases, tile_biases)

    def test_not_realistic_weights(self):
        """Test using non realistic weights."""
        layer = self.get_layer(realistic_read_write=False)

        shape = layer.weight.shape
        # Check that the tile weights are equal from the layer weights, as
        # the weights are synced after being set.
        tile_weights, tile_biases = layer.analog_tile.get_weights()
        self.assertTensorAlmostEqual(layer.weight, tile_weights.reshape(shape))
        if self.bias:
            self.assertTensorAlmostEqual(layer.bias, tile_biases)

        # 1. Set the layer weights and biases.
        user_weights = Tensor(layer.out_features, layer.in_features).uniform_(-0.5, 0.5)
        user_biases = Tensor(layer.out_features).uniform_(-0.5, 0.5)
        layer.set_weights(user_weights, user_biases)

        # Check that the tile weights are equal from the layer weights, as
        # the weights are synced after being set.
        tile_weights, tile_biases = layer.analog_tile.get_weights()
        self.assertTensorAlmostEqual(layer.weight, tile_weights.reshape(shape))
        if self.bias:
            self.assertTensorAlmostEqual(layer.bias, tile_biases)

        # Check that the tile weights are equal to the user-specified
        # weights, as it is not realistic.
        self.assertTensorAlmostEqual(user_weights, tile_weights)
        if self.bias:
            self.assertTensorAlmostEqual(user_biases, tile_biases)

        # 2. Get the layer weights and biases.
        gotten_weights, gotten_biases = layer.get_weights()

        # Check that the tile weights are equal than the gotten
        # weights, as it is not realistic.
        self.assertTensorAlmostEqual(gotten_weights, tile_weights)
        if self.bias:
            self.assertTensorAlmostEqual(gotten_biases, tile_biases)

    def test_sequential_move_to_cuda(self):
        """Test sequential cuda."""
        if not cuda.is_compiled():
            raise SkipTest('not compiled with CUDA support')

        # Map the original tile classes to the expected ones after `cuda()`.
        tile_classes = {
            AnalogTile: CudaAnalogTile,
            CudaAnalogTile: CudaAnalogTile
        }

        layer = self.get_layer()
        expected_class = tile_classes[layer.analog_tile.__class__]

        # Create a container and move to cuda.
        model = AnalogSequential(layer)
        model.cuda()

        # Assert the tile has been moved to cuda.
        self.assertIsInstance(layer.analog_tile, expected_class)

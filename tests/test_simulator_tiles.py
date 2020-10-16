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

"""Tests for the high level simulator devices functionality."""

from torch import Tensor

from .helpers.decorators import parametrize_over_tiles
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import (
    FloatingPoint, Ideal, ConstantStep, LinearStep,
    ExpStep, Vector, Difference, Transfer, Inference,
    FloatingPointCuda, IdealCuda, ConstantStepCuda, LinearStepCuda,
    ExpStepCuda, VectorCuda, DifferenceCuda, TransferCuda, InferenceCuda
)


@parametrize_over_tiles([
    FloatingPoint,
    Ideal,
    ConstantStep,
    LinearStep,
    ExpStep,
    Vector,
    Difference,
    Transfer,
    Inference,
    FloatingPointCuda,
    IdealCuda,
    ConstantStepCuda,
    LinearStepCuda,
    ExpStepCuda,
    VectorCuda,
    DifferenceCuda,
    TransferCuda,
    InferenceCuda
])
class TileTest(ParametrizedTestCase):
    """Test floating point tile."""

    def test_bias(self):
        """Test instantiating a tile."""
        out_size = 2
        in_size = 3

        analog_tile = self.get_tile(out_size, in_size, bias=True)

        self.assertIsInstance(analog_tile.tile, self.simulator_tile_class)

        learning_rate = 0.123
        # Use [out_size, in_size] weights.
        weights = Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        biases = Tensor([-0.1, -0.2])

        # Set some properties in the simulators.Tile.
        analog_tile.set_learning_rate(0.123)
        analog_tile.set_weights(weights, biases)

        # Assert over learning rate.
        self.assertAlmostEqual(analog_tile.get_learning_rate(), learning_rate)
        self.assertAlmostEqual(analog_tile.get_learning_rate(),
                               analog_tile.tile.get_learning_rate())

        # Assert over weights and biases.
        tile_weights, tile_biases = analog_tile.get_weights()
        self.assertEqual(tile_weights.shape, (out_size, in_size))
        self.assertEqual(tile_biases.shape, (out_size,))
        self.assertTensorAlmostEqual(tile_weights, weights)
        self.assertTensorAlmostEqual(tile_biases, biases)

    def test_no_bias(self):
        """Test instantiating a floating point tile."""
        out_size = 2
        in_size = 3

        analog_tile = self.get_tile(out_size, in_size, bias=False)
        self.assertIsInstance(analog_tile.tile, self.simulator_tile_class)

        learning_rate = 0.123

        # Use [out_size, in_size] weights.
        weights = Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        # Set some properties in the simulators.Tile.
        analog_tile.set_learning_rate(0.123)
        analog_tile.set_weights(weights)

        # Assert over learning rate.
        self.assertAlmostEqual(analog_tile.get_learning_rate(), learning_rate)
        self.assertAlmostEqual(analog_tile.get_learning_rate(),
                               analog_tile.tile.get_learning_rate())

        # Assert over weights and biases.
        tile_weights, tile_biases = analog_tile.get_weights()
        self.assertEqual(tuple(tile_weights.shape), (out_size, in_size))
        self.assertEqual(tile_biases, None)
        self.assertTensorAlmostEqual(tile_weights, weights)

    def test_get_hidden_parameters(self):
        """Test getting hidden parameters."""
        analog_tile = self.get_tile(4, 5)

        hidden_parameters = analog_tile.get_hidden_parameters()
        field = self.first_hidden_field

        # Check that there are hidden parameters.
        if field:
            self.assertGreater(len(hidden_parameters), 0)
        else:
            self.assertEqual(len(hidden_parameters), 0)

        if field:
            # Check that one of the parameters is correct.
            self.assertIn(field, hidden_parameters.keys())
            self.assertEqual(hidden_parameters[field].shape,
                             (4, 5))
            self.assertTrue(all(val == 0.6 for val in
                                hidden_parameters[field].flatten()))

    def test_set_hidden_parameters(self):
        """Test setting hidden parameters."""
        analog_tile = self.get_tile(3, 4)

        hidden_parameters = analog_tile.get_hidden_parameters()
        field = self.first_hidden_field

        # Update one of the values of the hidden parameters.
        if field:
            # set higher as default otherwise hidden weight might change
            hidden_parameters[field][1][1] = 0.8

        analog_tile.set_hidden_parameters(hidden_parameters)

        # Check that the change was propagated to the tile.
        new_hidden_parameters = analog_tile.get_hidden_parameters()

        # Compare old and new hidden parameters tensors.
        for (_, old), (_, new) in zip(hidden_parameters.items(),
                                      new_hidden_parameters.items()):
            self.assertTrue(old.allclose(new))

        if field:
            self.assertEqual(new_hidden_parameters[field][1][1], 0.8)

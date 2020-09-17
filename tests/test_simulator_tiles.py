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

from unittest import TestCase, skipIf

from numpy.testing import assert_array_almost_equal

from torch import Tensor

from aihwkit.simulator.devices import ConstantStepResistiveDevice
from aihwkit.simulator.tiles import AnalogTile, FloatingPointTile
from aihwkit.simulator.parameters import ConstantStepResistiveDeviceParameters
from aihwkit.simulator.rpu_base import tiles, cuda


class TileTestMixin:
    """Test floating point tile."""

    simulator_tile_class = None

    def get_tile(self, out_size, in_size, **kwargs):
        """Return an analog tile of the specified dimensions."""
        raise NotImplementedError

    def assertTensorAlmostEqual(self, tensor_a, tensor_b):
        """Assert that two tensors are almost equal."""
        # pylint: disable=invalid-name
        array_a = tensor_a.detach().cpu().numpy()
        array_b = tensor_b.detach().cpu().numpy()
        assert_array_almost_equal(array_a, array_b)

    def test_bias(self):
        """Test instantiating a floating point tile."""
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

        # Check that there are hidden parameters.
        if isinstance(analog_tile, AnalogTile):
            self.assertGreater(len(hidden_parameters), 0)
        else:
            self.assertEqual(len(hidden_parameters), 0)

        if isinstance(analog_tile, AnalogTile):
            # Check that one of the parameters is correct.
            self.assertIn('max_bound', hidden_parameters.keys())
            self.assertEqual(hidden_parameters['max_bound'].shape,
                             (4, 5))
            self.assertTrue(all(val == 0.6 for val in
                                hidden_parameters['max_bound'].flatten()))

    def test_set_hidden_parameters(self):
        """Test setting hidden parameters."""
        analog_tile = self.get_tile(3, 4)

        hidden_parameters = analog_tile.get_hidden_parameters()

        # Update one of the values of the hidden parameters.
        if isinstance(analog_tile, AnalogTile):
            hidden_parameters['max_bound'][1][1] = 0.1

        analog_tile.set_hidden_parameters(hidden_parameters)

        # Check that the change was propagated to the tile.
        new_hidden_parameters = analog_tile.get_hidden_parameters()

        # Compare old and new hidden parameters tensors.
        for (_, old), (_, new) in zip(hidden_parameters.items(),
                                      new_hidden_parameters.items()):
            self.assertTrue(old.allclose(new))
        if isinstance(analog_tile, AnalogTile):
            self.assertEqual(new_hidden_parameters['max_bound'][1][1], 0.1)


class FloatingPointTileTest(TileTestMixin, TestCase):
    """Tests for FloatingPointTile."""

    simulator_tile_class = tiles.FloatingPointTile

    def get_tile(self, out_size, in_size, **kwargs):
        return FloatingPointTile(out_size, in_size, **kwargs)


class AnalogTileTest(TileTestMixin, TestCase):
    """Tests for AnalogTile."""

    simulator_tile_class = tiles.AnalogTile

    def get_tile(self, out_size, in_size, **kwargs):
        resistive_device = ConstantStepResistiveDevice(
            ConstantStepResistiveDeviceParameters(w_max_dtod=0,
                                                  w_min_dtod=0))

        return AnalogTile(out_size, in_size, resistive_device, **kwargs)


@skipIf(not cuda.is_compiled(), 'not compiled with CUDA support')
class CudaFloatingPointTileTest(TileTestMixin, TestCase):
    """Tests for FloatingPointTile (cuda)."""

    simulator_tile_class = tiles.FloatingPointTile

    def get_tile(self, out_size, in_size, **kwargs):
        return FloatingPointTile(out_size, in_size, **kwargs).cuda()


@skipIf(not cuda.is_compiled(), 'not compiled with CUDA support')
class CudaAnalogTileTest(TileTestMixin, TestCase):
    """Tests for AnalogTile (cuda)."""

    simulator_tile_class = tiles.AnalogTile

    def get_tile(self, out_size, in_size, **kwargs):
        resistive_device = ConstantStepResistiveDevice(
            ConstantStepResistiveDeviceParameters(w_max_dtod=0,
                                                  w_min_dtod=0))

        return AnalogTile(out_size, in_size, resistive_device, **kwargs)

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

"""Tests for the RPU array bindings."""

from unittest import SkipTest

from numpy import array, std, dot
from numpy.random import uniform
from numpy.testing import assert_array_equal, assert_array_almost_equal

from torch import Tensor, from_numpy

from aihwkit.simulator.rpu_base import tiles, cuda

from aihwkit.simulator.configs import FloatingPointRPUConfig, SingleRPUConfig
from aihwkit.simulator.configs.devices import FloatingPointDevice, ConstantStepDevice, IdealDevice
from aihwkit.simulator.configs.utils import IOParameters

from .helpers.decorators import parametrize_over_tiles
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import (FloatingPoint, ConstantStep,
                            FloatingPointCuda, ConstantStepCuda)


@parametrize_over_tiles([
    FloatingPoint,
    ConstantStep,
    FloatingPointCuda,
    ConstantStepCuda
])
class BindingsTilesTest(ParametrizedTestCase):
    """Tests the basic functionality of FloatingPoint and Analog tiles."""

    @staticmethod
    def set_init_weights(python_tile):
        """Generate and set the weight init."""
        init_weights = from_numpy(
            uniform(-0.5, 0.5, size=(python_tile.out_size, python_tile.in_size)))
        python_tile.set_weights(init_weights)

    def get_noisefree_tile(self, out_size, in_size):
        """Return a tile of the specified dimensions with noisiness turned off."""
        rpu_config = None

        if 'FloatingPoint' not in self.parameter:
            rpu_config = SingleRPUConfig(
                forward=IOParameters(is_perfect=True),
                backward=IOParameters(is_perfect=True),
                device=IdealDevice()
            )

        python_tile = self.get_tile(out_size, in_size, rpu_config)
        self.set_init_weights(python_tile)

        return python_tile

    def get_custom_tile(self, out_size, in_size, **parameters):
        """Return a tile with custom parameters for the resistive device."""
        if 'FloatingPoint' in self.parameter:
            rpu_config = FloatingPointRPUConfig(
                device=FloatingPointDevice(**parameters))
        else:
            rpu_config = SingleRPUConfig(
                device=ConstantStepDevice(**parameters))

        python_tile = self.get_tile(out_size, in_size, rpu_config)
        self.set_init_weights(python_tile)

        return python_tile

    def test_instantiate(self):
        """Check that the binding is instantiable."""
        python_tile = self.get_tile(2, 3)
        cpp_tile = python_tile.tile

        # Take advantage of inheritance for this assertion.
        self.assertIsInstance(cpp_tile, tiles.FloatingPointTile)

    def test_setters_learning_rate(self):
        """Check setting and getting the learning rate."""
        python_tile = self.get_tile(2, 3)
        cpp_tile = python_tile.tile

        cpp_tile.set_learning_rate(1.23)
        self.assertAlmostEqual(cpp_tile.get_learning_rate(), 1.23)

    def test_decay_weights(self):
        """Check decaying the weights."""
        decay_rate = 0.1

        # Use custom parameters for the tile.
        python_tile = self.get_custom_tile(2, 3, lifetime=1.0/decay_rate)
        cpp_tile = python_tile.tile

        init_weights = cpp_tile.get_weights().copy()
        cpp_tile.decay_weights(1.0)
        weights = cpp_tile.get_weights()

        assert_array_almost_equal(weights, init_weights*(1.0 - decay_rate))

    def test_diffuse_weights(self):
        """Check diffusing the weights."""
        diffusion_rate = 0.1

        # Use custom parameters for the tile.
        python_tile = self.get_custom_tile(100, 122, diffusion=diffusion_rate)
        cpp_tile = python_tile.tile

        init_weights = cpp_tile.get_weights().copy()
        cpp_tile.diffuse_weights()
        weights = cpp_tile.get_weights()

        deviation_std = std((weights - init_weights).flatten())
        self.assertLess(deviation_std, 1.1*diffusion_rate)
        self.assertGreater(deviation_std, 0.9*diffusion_rate)

    def test_mimic_rpu_mac(self):
        """Check using the update, forward and backward functions."""
        n_rows = 6  # out_size aka d_size
        n_cols = 5  # in_size aka x_size
        m_batch = 4
        lr = 0.1

        python_tile = self.get_noisefree_tile(n_rows, n_cols)
        cpp_tile = python_tile.tile

        cpp_tile.set_learning_rate(lr)

        x_t = from_numpy(uniform(-1.2, 1.2, size=(m_batch, n_cols)).astype('float32'))
        d_t = from_numpy(uniform(-0.1, 0.1, size=(m_batch, n_rows)).astype('float32'))

        init_weights = cpp_tile.get_weights().copy()

        if python_tile.is_cuda:
            x_t = x_t.cuda()
            d_t = d_t.cuda()

        # Perform forward.
        y = cpp_tile.forward(x_t).cpu()
        self.assertIsInstance(y, Tensor)
        assert_array_almost_equal(y.numpy(), dot(x_t.cpu(), init_weights.T))

        # Perform backward.
        z = cpp_tile.backward(d_t).cpu()
        self.assertIsInstance(y, Tensor)
        assert_array_almost_equal(z.numpy(), dot(d_t.cpu(), init_weights))

        # Perform update.
        cpp_tile.update(x_t, d_t, bias=False)
        post_rank_weights = cpp_tile.get_weights()
        ref_weights = init_weights - lr*dot(d_t.cpu().T, x_t.cpu())

        assert_array_almost_equal(post_rank_weights, ref_weights)

    def test_cuda_instantiation(self):
        """Test whether cuda weights are copied correctly."""
        if not self.use_cuda and not cuda.is_compiled():
            raise SkipTest('not compiled with CUDA support')

        python_tile = self.get_tile(10, 12)
        init_weights = python_tile.tile.get_weights()

        cuda_python_tile = python_tile.cuda()
        init_weights_cuda = cuda_python_tile.tile.get_weights()
        assert_array_almost_equal(init_weights, init_weights_cuda)


@parametrize_over_tiles([
    FloatingPoint, FloatingPointCuda
])
class FloatingPointTileTest(ParametrizedTestCase):
    """Test `rpu_base.FloatingPointTile` functionality."""

    def test_setters_weights(self):
        """Check setting and getting the weights."""
        python_tile = self.get_tile(2, 3)
        cpp_tile = python_tile.tile

        # Set weights using Lists.
        input_weights = [[1, 2, 3], [4, 5, 6]]
        cpp_tile.set_weights(input_weights)
        assert_array_equal(cpp_tile.get_weights(), input_weights)

        # Set weights using numpy arrays.
        input_weights = array([[6, 5, 4], [3, 2, 1]])
        cpp_tile.set_weights(input_weights)
        assert_array_equal(cpp_tile.get_weights(), input_weights)

    def test_setters_weights_realistic(self):
        """Check setting and getting the weights."""
        python_tile = self.get_tile(2, 3)
        cpp_tile = python_tile.tile

        # Set weights using Lists.
        input_weights = [[1, 2, 3], [4, 5, 6]]
        cpp_tile.set_weights_realistic(input_weights, 1)
        assert_array_equal(cpp_tile.get_weights_realistic(), input_weights)

        # Set weights using numpy arrays.
        input_weights = array([[6, 5, 4], [3, 2, 1]])
        cpp_tile.set_weights_realistic(input_weights, 10)
        assert_array_equal(cpp_tile.get_weights_realistic(), input_weights)


@parametrize_over_tiles([
    ConstantStep, ConstantStepCuda
])
class AnalogTileTest(ParametrizedTestCase):
    """Test `rpu_base.AnalogTile` functionality."""

    def test_setters_weights(self):
        """Check setting and getting the weights."""
        python_tile = self.get_tile(2, 3)
        cpp_tile = python_tile.tile

        # Set weights using Lists.
        input_weights = [[1, 2, 3], [4, 5, 6]]
        cpp_tile.set_weights(input_weights)
        self.assertEqual(cpp_tile.get_weights().shape, (2, 3))

        # Set weights using numpy arrays.
        input_weights = array([[6, 5, 4], [3, 2, 1]])
        cpp_tile.set_weights(input_weights)
        self.assertEqual(cpp_tile.get_weights().shape, (2, 3))

    def test_setters_weights_realistic(self):
        """Check setting and getting the weights."""
        python_tile = self.get_tile(2, 3)
        cpp_tile = python_tile.tile

        # Set weights using Lists.
        input_weights = [[1, 2, 3], [4, 5, 6]]
        cpp_tile.set_weights_realistic(input_weights, 10)
        self.assertEqual(cpp_tile.get_weights().shape, (2, 3))

        # Set weights using numpy arrays.
        input_weights = array([[6, 5, 4], [3, 2, 1]])
        cpp_tile.set_weights_realistic(input_weights, 10)
        self.assertEqual(cpp_tile.get_weights().shape, (2, 3))

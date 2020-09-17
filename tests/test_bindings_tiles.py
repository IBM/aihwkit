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

from unittest import TestCase, skipIf

from numpy import array, std, dot
from numpy.random import uniform
from numpy.testing import assert_array_equal, assert_array_almost_equal

from torch import Tensor, from_numpy

from aihwkit.simulator.tiles import FloatingPointTile, AnalogTile
from aihwkit.simulator.rpu_base import tiles, cuda

from aihwkit.simulator.parameters import (
    AnalogTileBackwardInputOutputParameters, FloatingPointTileParameters,
    ConstantStepResistiveDeviceParameters,
    AnalogTileInputOutputParameters,
    AnalogTileUpdateParameters,
    PulseType
)

from aihwkit.simulator.devices import (FloatingPointResistiveDevice,
                                       ConstantStepResistiveDevice)


class TileTestMixin:
    """Mixin for tests over analog tiles."""

    def set_init_weights(self, python_tile):
        """Generate and set the weight init."""
        init_weights = from_numpy(
            uniform(-0.5, 0.5, size=(python_tile.out_size, python_tile.in_size)))
        python_tile.set_weights(init_weights)

    def get_device_params(self, **kwargs):
        """ Return the default device params"""
        raise NotImplementedError

    def get_tile(self, out_size, in_size, device_params=None):
        """Return an floating point tile of the specified dimensions."""
        raise NotImplementedError

    def get_noisefree_tile(self, out_size, in_size):
        """Return a tile of the specified dimensions with noisiness turned off."""
        raise NotImplementedError

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
        params = self.get_device_params(lifetime=1.0/decay_rate)
        python_tile = self.get_tile(2, 3, params)
        cpp_tile = python_tile.tile

        init_weights = cpp_tile.get_weights().copy()
        cpp_tile.decay_weights(1.0)
        weights = cpp_tile.get_weights()

        assert_array_almost_equal(weights, init_weights*(1.0 - decay_rate))

    def test_diffuse_weights(self):
        """Check diffusing the weights."""
        diffusion_rate = 0.1

        params = self.get_device_params(diffusion=diffusion_rate)
        python_tile = self.get_tile(100, 122, params)
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


class FloatingPointTileTest(TestCase, TileTestMixin):
    """Test `rpu_base.FloatingPointTile` functionality."""

    def get_device_params(self, **kwargs):
        """ Return the default device params"""
        return FloatingPointTileParameters(**kwargs)

    def get_tile(self, out_size, in_size, device_params=None):
        """Return an floating point tile of the specified dimensions."""
        device_params = device_params or FloatingPointTileParameters()
        resistive_device = FloatingPointResistiveDevice(device_params)
        python_tile = FloatingPointTile(out_size, in_size, resistive_device)
        self.set_init_weights(python_tile)
        return python_tile

    def get_noisefree_tile(self, out_size, in_size):
        """Return a tile of the specified dimensions with noisiness turned off."""
        return self.get_tile(out_size, in_size)

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


class AnalogTileTest(TestCase, TileTestMixin):
    """Test `rpu_base.AnalogTile` functionality."""

    def get_device_params(self, **kwargs):
        """Return the default device params"""
        return ConstantStepResistiveDeviceParameters(**kwargs)

    def get_tile(self, out_size, in_size, device_params=None):
        """Return an analog tile of the specified dimensions."""
        device_params = device_params or ConstantStepResistiveDeviceParameters()
        resistive_device = ConstantStepResistiveDevice(params_devices=device_params)
        python_tile = AnalogTile(out_size, in_size, resistive_device)
        self.set_init_weights(python_tile)
        return python_tile

    def get_noisefree_tile(self, out_size, in_size):
        """Return a tile of the specified dimensions with noisiness turned off."""
        params = ConstantStepResistiveDeviceParameters()
        io_params = AnalogTileInputOutputParameters(is_perfect=True)
        io_params_backward = AnalogTileBackwardInputOutputParameters(is_perfect=True)
        up_params = AnalogTileUpdateParameters(pulse_type=PulseType('None'))
        resistive_device = ConstantStepResistiveDevice(params_devices=params,
                                                       params_forward=io_params,
                                                       params_backward=io_params_backward,
                                                       params_update=up_params)
        python_tile = AnalogTile(out_size, in_size, resistive_device)
        return python_tile

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


@skipIf(not cuda.is_compiled(), 'not compiled with CUDA support')
class CudaFloatingPointTileTest(FloatingPointTileTest):
    """Test `rpu_base.CudaFloatingPointTile` functionality."""

    def get_tile(self, out_size, in_size, device_params=None):
        """Return a CUDA floating point tile of the specified dimensions."""
        return super().get_tile(out_size, in_size, device_params).cuda()


@skipIf(not cuda.is_compiled(), 'not compiled with CUDA support')
class CudaAnalogTileTest(AnalogTileTest):
    """Test `rpu_base.CudaAnalogTile` functionality."""

    def get_tile(self, out_size, in_size, device_params=None):
        """Return a CUDA analog tile of the specified dimensions."""
        return super().get_tile(out_size, in_size, device_params).cuda()

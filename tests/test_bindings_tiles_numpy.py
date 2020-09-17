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

from numpy import dot
from numpy.random import uniform
from numpy.testing import assert_array_almost_equal, assert_array_equal

from torch import from_numpy

from aihwkit.simulator.tiles_numpy import NumpyFloatingPointTile, NumpyAnalogTile

from aihwkit.simulator.parameters import (
    AnalogTileBackwardInputOutputParameters, ConstantStepResistiveDeviceParameters,
    AnalogTileInputOutputParameters,
    AnalogTileUpdateParameters,
    PulseType
)

from aihwkit.simulator.devices import (FloatingPointResistiveDevice,
                                       ConstantStepResistiveDevice)


class NumpyFloatingPointTileTest(TestCase):
    """Test NumpyFloatingPointTile."""

    def get_tile(self, out_size, in_size):
        """Return a tile of the specified dimensions with noisiness turned off."""
        resistive_device = FloatingPointResistiveDevice()
        python_tile = NumpyFloatingPointTile(out_size, in_size, resistive_device)

        # Set weights.
        init_weights = uniform(-0.5, 0.5, size=(python_tile.out_size, python_tile.in_size))
        python_tile.set_weights(from_numpy(init_weights))

        return python_tile

    def test_mimic_rpu_mac(self):
        """Check using the update, forward and backward functions."""
        n_rows = 6  # out_size aka d_size
        n_cols = 5  # in_size aka x_size
        m_batch = 4
        lr = 0.1

        python_tile = self.get_tile(n_rows, n_cols)
        cpp_tile = python_tile.tile

        cpp_tile.set_learning_rate(lr)

        x = uniform(-1.2, 1.2, size=(m_batch, n_cols))
        d = uniform(-0.1, 0.1, size=(m_batch, n_rows))

        init_weights = cpp_tile.get_weights().copy()

        # Perform forward.
        y = cpp_tile.forward_numpy(x)
        assert_array_almost_equal(y, dot(x, init_weights.T))

        # Perform backward.
        z = cpp_tile.backward_numpy(d)
        assert_array_almost_equal(z, dot(d, init_weights))

        # Perform update.
        cpp_tile.update_numpy(x, d, bias=False)
        post_rank_weights = cpp_tile.get_weights()
        ref_weights = init_weights - lr*dot(d.T, x)

        assert_array_almost_equal(post_rank_weights, ref_weights)

    def test_mimic_rpu_mac_trans(self):
        """Check using the update, forward and backward functions in
        transposed mode."""
        n_rows = 6
        n_cols = 5
        m_batch = 4
        lr = 0.1

        python_tile = self.get_tile(n_rows, n_cols)
        python_tile.in_trans = True
        python_tile.out_trans = True

        cpp_tile = python_tile.tile
        cpp_tile.set_learning_rate(lr)

        x = uniform(-1.2, 1.2, size=(m_batch, n_cols))
        d = uniform(-0.1, 0.1, size=(m_batch, n_rows))

        init_weights = cpp_tile.get_weights().copy()

        # Perform forward.
        # Caution: x.T just a view! need to copy
        y = cpp_tile.forward_numpy(x.T.copy(), x_trans=True, d_trans=True)
        assert_array_almost_equal(y.T, dot(x, init_weights.T))

        # Perform backward.
        z = cpp_tile.backward_numpy(d.T.copy(), x_trans=True, d_trans=True).T
        assert_array_almost_equal(z, dot(d, init_weights))

        # Perform update.
        cpp_tile.update_numpy(x.T.copy(), d.T.copy(), bias=False,
                              x_trans=True, d_trans=True)
        post_rank_weights = cpp_tile.get_weights()
        ref_weights = init_weights - lr*dot(d.T, x)

        assert_array_almost_equal(post_rank_weights, ref_weights)

    @skipIf(True, "Currently not supported")
    def test_move_weights_new_variable(self):
        """Test moving the weights to a new shared variable."""
        python_tile = self.get_tile(2, 3)
        cpp_tile = python_tile.tile

        def to_shape(mat):
            if python_tile.is_cuda:
                shape = mat.shape
                return mat.T.copy().flatten().reshape(*shape)
            return mat

        # Read the weights.
        initial_weights = cpp_tile.get_weights()

        # Move the weights to a new shared variable.
        weights_holder = python_tile.move_weights()
        weights_clone = weights_holder.clone().cpu().numpy()
        assert_array_equal(weights_clone, to_shape(initial_weights))

        # Modify the shared variable manually.
        weights_holder.add_(0.123)  # in place
        current_weights = cpp_tile.get_weights()
        weights_clone = weights_holder.clone().cpu().numpy()
        assert_array_equal(weights_clone, to_shape(current_weights))

        # Modify the weights manually.
        cpp_tile.set_weights([[.2, .2, .2], [.3, .3, .3]])
        current_weights = cpp_tile.get_weights()
        weights_clone = weights_holder.clone().cpu().numpy()
        assert_array_equal(weights_clone, to_shape(current_weights))


class NumpyAnalogTileTest(NumpyFloatingPointTileTest):
    """Test NumpyAnalogTile."""

    def get_tile(self, out_size, in_size):
        """Return a tile of the specified dimensions with noisiness turned off."""
        params = ConstantStepResistiveDeviceParameters()
        io_params = AnalogTileInputOutputParameters(is_perfect=True)
        io_params_backward = AnalogTileBackwardInputOutputParameters(is_perfect=True)
        up_params = AnalogTileUpdateParameters(pulse_type=PulseType('None'))
        resistive_device = ConstantStepResistiveDevice(params_devices=params,
                                                       params_forward=io_params,
                                                       params_backward=io_params_backward,
                                                       params_update=up_params)
        python_tile = NumpyAnalogTile(out_size, in_size, resistive_device)

        # Set weights.
        init_weights = uniform(-0.5, 0.5, size=(python_tile.out_size, python_tile.in_size))
        python_tile.set_weights(from_numpy(init_weights))

        return python_tile

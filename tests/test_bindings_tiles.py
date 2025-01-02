# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for the RPU array bindings."""

from unittest import SkipTest

from numpy import array, std, dot, reshape
from numpy.random import uniform
from numpy.testing import assert_array_equal, assert_array_almost_equal

from torch import Tensor, from_numpy
from torch.cuda import init

from aihwkit.simulator.rpu_base import tiles

from aihwkit.simulator.configs import FloatingPointRPUConfig, SingleRPUConfig
from aihwkit.simulator.configs.devices import FloatingPointDevice, ConstantStepDevice, IdealDevice
from aihwkit.simulator.parameters import IOParameters, DriftParameter

from .helpers.decorators import parametrize_over_tiles
from .helpers.testcases import ParametrizedTestCase, SKIP_CUDA_TESTS
from .helpers.tiles import FloatingPoint, ConstantStep, FloatingPointCuda, ConstantStepCuda

if not SKIP_CUDA_TESTS:
    init()


@parametrize_over_tiles([FloatingPoint, ConstantStep, FloatingPointCuda, ConstantStepCuda])
class BindingsTilesTest(ParametrizedTestCase):
    """Tests the basic functionality of FloatingPoint and Analog tiles."""

    @staticmethod
    def set_init_weights(python_tile):
        """Generate and set the weight init."""
        init_weights = from_numpy(
            uniform(-0.5, 0.5, size=(python_tile.out_size, python_tile.in_size))
        )
        python_tile.set_weights(init_weights)

    def get_noisefree_tile(self, out_size, in_size):
        """Return a tile of the specified dimensions with noisiness turned off."""
        rpu_config = None

        if "FloatingPoint" not in self.parameter:
            rpu_config = SingleRPUConfig(
                forward=IOParameters(is_perfect=True),
                backward=IOParameters(is_perfect=True),
                device=IdealDevice(),
            )

        python_tile = self.get_tile(out_size, in_size, rpu_config)
        self.set_init_weights(python_tile)

        return python_tile

    def get_custom_tile(self, out_size, in_size, **parameters):
        """Return a tile with custom parameters for the resistive device."""
        if "FloatingPoint" in self.parameter:
            rpu_config = FloatingPointRPUConfig(device=FloatingPointDevice(**parameters))
        else:
            rpu_config = SingleRPUConfig(device=ConstantStepDevice(**parameters))

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
        python_tile = self.get_custom_tile(2, 3, lifetime=1.0 / decay_rate)
        cpp_tile = python_tile.tile

        init_weights = cpp_tile.get_weights().numpy()
        cpp_tile.decay_weights(1.0)
        weights = cpp_tile.get_weights().numpy()

        assert_array_almost_equal(weights, init_weights * (1.0 - decay_rate))

    def test_diffuse_weights(self):
        """Check diffusing the weights."""
        diffusion_rate = 0.1

        # Use custom parameters for the tile.
        python_tile = self.get_custom_tile(100, 122, diffusion=diffusion_rate)
        cpp_tile = python_tile.tile

        init_weights = cpp_tile.get_weights().numpy()
        cpp_tile.diffuse_weights()
        weights = cpp_tile.get_weights().numpy()

        deviation_std = std((weights - init_weights).flatten())
        self.assertLess(deviation_std, 1.1 * diffusion_rate)
        self.assertGreater(deviation_std, 0.9 * diffusion_rate)

    def test_drift_weights(self):
        """Check drifting the weights."""
        nu = 0.1
        drift_params = DriftParameter(nu=nu, t_0=1.0, nu_dtod=0.0, nu_std=0.0, w_noise_std=0.0)
        delta_t = 2.0
        # Use custom parameters for the tile.
        python_tile = self.get_custom_tile(100, 122, drift=drift_params)
        cpp_tile = python_tile.tile

        init_weights = cpp_tile.get_weights()
        cpp_tile.drift_weights(delta_t)
        weights = cpp_tile.get_weights()

        assert_array_almost_equal(weights, init_weights * (delta_t) ** (-nu))

    def test_mimic_rpu_mac(self):
        """Check using the update, forward and backward functions."""
        n_rows = 6  # out_size aka d_size
        n_cols = 5  # in_size aka x_size
        m_batch = 4
        lr = 0.1

        python_tile = self.get_noisefree_tile(n_rows, n_cols)
        cpp_tile = python_tile.tile

        cpp_tile.set_learning_rate(lr)

        x_t = from_numpy(uniform(-1.2, 1.2, size=(m_batch, n_cols)).astype("float32"))
        d_t = from_numpy(uniform(-0.1, 0.1, size=(m_batch, n_rows)).astype("float32"))

        init_weights = cpp_tile.get_weights()

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
        ref_weights = init_weights - lr * dot(d_t.cpu().T, x_t.cpu())

        assert_array_almost_equal(post_rank_weights, ref_weights)

    def test_cuda_instantiation(self):
        """Test whether cuda weights are copied correctly."""
        if not self.use_cuda or SKIP_CUDA_TESTS:
            raise SkipTest("not compiled with CUDA support")

        python_tile = self.get_tile(10, 12)
        init_weights = python_tile.tile.get_weights()

        cuda_python_tile = python_tile.cuda()
        init_weights_cuda = cuda_python_tile.tile.get_weights()
        assert_array_almost_equal(init_weights, init_weights_cuda)


@parametrize_over_tiles([FloatingPoint, FloatingPointCuda])
class FloatingPointTileTest(ParametrizedTestCase):
    """Test `rpu_base.FloatingPointTile` functionality."""

    def test_setters_weights(self):
        """Check setting and getting the weights."""
        python_tile = self.get_tile(2, 3)
        cpp_tile = python_tile.tile

        # Set weights using Tensors.
        input_weights = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        cpp_tile.set_weights(input_weights)
        assert_array_equal(cpp_tile.get_weights(), input_weights)

        # Set weights using numpy (via python tile).
        input_weights = array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        python_tile.set_weights(input_weights)
        assert_array_equal(cpp_tile.get_weights(), input_weights)

        # Set weights using Lists (via python tile).
        input_weights = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        python_tile.set_weights(input_weights)
        assert_array_equal(cpp_tile.get_weights(), input_weights)

    def test_setters_weights_realistic(self):
        """Check setting and getting the weights."""
        python_tile = self.get_tile(2, 3)
        cpp_tile = python_tile.tile

        # Set weights using Tensors.
        input_weights = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        cpp_tile.set_weights(input_weights)
        assert_array_equal(cpp_tile.get_weights(), input_weights)

    def test_n_dim_forward(self):
        """Tests whether forward n-dim inputs work as expected"""
        in_size = 6
        out_size = 5
        add_shape = [2, 3, 4]

        python_tile = self.get_tile(out_size, in_size)
        init_weights = python_tile.get_weights()[0]
        x_t = from_numpy(uniform(-0.1, 0.1, size=add_shape + [in_size]).astype("float32"))

        if python_tile.is_cuda:
            x_t = x_t.cuda()

        y_t = python_tile.forward(x_t).detach().cpu().numpy()
        x_t = x_t.detach().cpu().numpy()
        self.assertEqual(x_t.ndim, y_t.ndim)
        self.assertEqual(out_size, y_t.shape[-1])

        x_t = reshape(x_t, [-1, in_size])
        y_t = reshape(y_t, [-1, out_size])
        assert_array_almost_equal(dot(x_t, init_weights.T), y_t)

    def test_n_dim_backward(self):
        """Tests whether backward n-dim inputs work as expected"""
        in_size = 6
        out_size = 5
        add_shape = [4, 2]

        python_tile = self.get_tile(out_size, in_size)
        init_weights = python_tile.get_weights()[0].cpu().numpy()
        d_t = from_numpy(uniform(-0.1, 0.1, size=add_shape + [out_size]).astype("float32"))

        if python_tile.is_cuda:
            d_t = d_t.cuda()

        y_t = python_tile.backward(d_t).detach().cpu().numpy()
        d_t = d_t.detach().cpu().numpy()
        self.assertEqual(d_t.ndim, y_t.ndim)
        self.assertEqual(in_size, y_t.shape[-1])

        d_t = reshape(d_t, [-1, out_size])
        y_t = reshape(y_t, [-1, in_size])
        assert_array_almost_equal(dot(d_t, init_weights), y_t)

    def test_n_dim_update(self):
        """Tests whether update n-dim inputs work as expected"""
        in_size = 6
        out_size = 5
        lr = 0.3
        add_shape = []

        python_tile = self.get_tile(out_size, in_size)
        init_weights = python_tile.get_weights()[0].numpy()
        python_tile.set_learning_rate(lr)

        x_t = from_numpy(uniform(-0.1, 0.1, size=add_shape + [in_size]).astype("float32"))
        d_t = from_numpy(uniform(-0.1, 0.1, size=add_shape + [out_size]).astype("float32"))

        if python_tile.is_cuda:
            x_t = x_t.cuda()
            d_t = d_t.cuda()

        python_tile.update(x_t, d_t)
        updated_weights = python_tile.get_weights()[0]

        x_t = x_t.detach().cpu().numpy()
        x_t = reshape(x_t, [-1, in_size])

        d_t = d_t.detach().cpu().numpy()
        d_t = reshape(d_t, [-1, out_size])

        ref_weights = init_weights - lr * dot(d_t.T, x_t)
        assert_array_almost_equal(updated_weights, ref_weights)


@parametrize_over_tiles([ConstantStep, ConstantStepCuda])
class AnalogTileTest(ParametrizedTestCase):
    """Test `rpu_base.AnalogTile` functionality."""

    def test_setters_weights(self):
        """Check setting and getting the weights."""
        python_tile = self.get_tile(2, 3)
        cpp_tile = python_tile.tile

        # Set weights using Tensors.
        input_weights = Tensor([[6, 5, 4], [3, 2, 1]])
        cpp_tile.set_weights(input_weights)
        self.assertEqual(cpp_tile.get_weights().shape, (2, 3))

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

# pylint: disable=too-many-locals

"""Tests for the high level simulator devices functionality."""

from unittest import SkipTest

from torch import Tensor, zeros, ones

from aihwkit.simulator.configs.configs import UnitCellRPUConfig
from aihwkit.simulator.configs.compounds import (
    VectorUnitCell, ReferenceUnitCell
)
from aihwkit.simulator.configs.enums import (
    VectorUnitCellUpdatePolicy, NoiseManagementType, BoundManagementType
)
from aihwkit.simulator.configs.utils import PrePostProcessingRPU

from .helpers.decorators import parametrize_over_tiles
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import (
    FloatingPoint, Ideal, ConstantStep, LinearStep, SoftBounds,
    ExpStep, Vector, OneSided, Transfer, BufferedTransfer, MixedPrecision,
    PiecewiseStep, PiecewiseStepCuda,
    Inference, Reference, FloatingPointCuda, IdealCuda, ConstantStepCuda,
    LinearStepCuda, SoftBoundsCuda, ExpStepCuda, VectorCuda, OneSidedCuda,
    TransferCuda, BufferedTransferCuda, InferenceCuda, ReferenceCuda,
    MixedPrecisionCuda, PowStep, PowStepCuda,
    PowStepReference, PowStepReferenceCuda,
    SoftBoundsReference, SoftBoundsReferenceCuda,
)
from .helpers.testcases import SKIP_CUDA_TESTS

TOL = 1e-6


@parametrize_over_tiles([
    FloatingPoint,
    Ideal,
    ConstantStep,
    LinearStep,
    ExpStep,
    SoftBounds,
    SoftBoundsReference,
    Vector,
    OneSided,
    Transfer,
    BufferedTransfer,
    MixedPrecision,
    Inference,
    Reference,
    PowStep,
    PowStepReference,
    PiecewiseStep,
    PiecewiseStepCuda,
    FloatingPointCuda,
    IdealCuda,
    ConstantStepCuda,
    LinearStepCuda,
    ExpStepCuda,
    PowStepCuda,
    PowStepReferenceCuda,
    PiecewiseStepCuda,
    SoftBoundsCuda,
    SoftBoundsReferenceCuda,
    VectorCuda,
    OneSidedCuda,
    TransferCuda,
    BufferedTransferCuda,
    MixedPrecisionCuda,
    InferenceCuda,
    ReferenceCuda,
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
            self.assertTrue(all(abs(val - 0.6) < TOL for val in
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

    def test_post_update_step_diffuse(self):
        """Tests whether post update diffusion is performed"""
        rpu_config = self.get_rpu_config()

        if not hasattr(rpu_config.device, 'diffusion'):
            if hasattr(rpu_config.device, 'unit_cell_devices') \
               and hasattr(rpu_config.device.unit_cell_devices[-1], 'diffusion'):
                rpu_config.device.unit_cell_devices[-1].diffusion = 0.323
            else:
                raise SkipTest('This device does not support diffusion')
        else:
            rpu_config.device.diffusion = 0.323

        analog_tile = self.get_tile(2, 3, rpu_config=rpu_config, bias=True)

        weights = Tensor([[0.1, 0.2, 0.3], [0.4, -0.5, -0.6]])
        biases = Tensor([-0.1, 0.2])

        analog_tile.set_learning_rate(0.123)
        analog_tile.set_weights(weights, biases)

        analog_tile.post_update_step()

        tile_weights, tile_biases = analog_tile.get_weights()

        self.assertNotAlmostEqualTensor(tile_weights, weights)
        self.assertNotAlmostEqualTensor(tile_biases, biases)

    def test_post_update_step_lifetime(self):
        """Tests whether post update decay is performed"""
        rpu_config = self.get_rpu_config()

        if not hasattr(rpu_config.device, 'lifetime'):
            if hasattr(rpu_config.device, 'unit_cell_devices') \
               and hasattr(rpu_config.device.unit_cell_devices[-1], 'lifetime'):
                for idx, _ in enumerate(rpu_config.device.unit_cell_devices):
                    rpu_config.device.unit_cell_devices[idx].lifetime = 100.
            else:
                raise SkipTest('This device does not support lifetime')
        else:
            rpu_config.device.lifetime = 100.

        analog_tile = self.get_tile(2, 3, rpu_config=rpu_config, bias=True)

        weights = Tensor([[0.1, 0.2, 0.3], [0.4, -0.5, -0.6]])
        biases = Tensor([-0.1, 0.2])

        analog_tile.set_learning_rate(0.123)
        analog_tile.set_weights(weights, biases)

        analog_tile.post_update_step()

        tile_weights, tile_biases = analog_tile.get_weights()

        self.assertNotAlmostEqualTensor(tile_weights, weights)
        self.assertNotAlmostEqualTensor(tile_biases, biases)

    def test_set_hidden_update_index(self):
        """Tests hidden update index"""
        rpu_config = self.get_rpu_config()

        if not isinstance(rpu_config, UnitCellRPUConfig) \
           or not isinstance(rpu_config.device, (VectorUnitCell, ReferenceUnitCell)):
            analog_tile = self.get_tile(2, 3, rpu_config=rpu_config, bias=False)
            index = analog_tile.get_hidden_update_index()
            self.assertEqual(index, 0)
            analog_tile.set_hidden_update_index(1)
            index = analog_tile.get_hidden_update_index()
            self.assertEqual(index, 0)
        else:
            rpu_config.device.first_update_idx = 0
            rpu_config.device.update_policy = VectorUnitCellUpdatePolicy.SINGLE_FIXED
            analog_tile = self.get_tile(2, 3, rpu_config=rpu_config, bias=False)
            analog_tile.set_learning_rate(0.123)

            # update index
            index = analog_tile.get_hidden_update_index()
            self.assertEqual(index, 0)
            analog_tile.set_hidden_update_index(1)
            index = analog_tile.get_hidden_update_index()
            self.assertEqual(index, 1)

            # set weights index 0
            analog_tile.set_hidden_update_index(0)
            weights_0 = Tensor([[0.1, 0.2, 0.3], [0.4, -0.5, -0.6]])
            analog_tile.tile.set_weights(weights_0)
            hidden_par = analog_tile.get_hidden_parameters()
            self.assertTensorAlmostEqual(hidden_par['hidden_weights_0'],
                                         weights_0)
            self.assertTensorAlmostEqual(hidden_par['hidden_weights_1'],
                                         zeros((2, 3)))

            # set weights index 1
            analog_tile.set_hidden_update_index(1)
            weights_1 = Tensor([[0.4, 0.1, 0.2], [0.5, -0.2, -0.1]])
            analog_tile.tile.set_weights(weights_1)

            hidden_par = analog_tile.get_hidden_parameters()

            self.assertTensorAlmostEqual(hidden_par['hidden_weights_0'],
                                         weights_0)
            self.assertTensorAlmostEqual(hidden_par['hidden_weights_1'],
                                         weights_1)

            # update
            analog_tile.set_hidden_update_index(1)
            x = Tensor([[0.1, 0.2, 0.3], [0.4, -0.5, -0.6]])
            d = Tensor([[0.5, 0.1], [0.4, -0.5]])
            if analog_tile.is_cuda:
                x = x.cuda()
                d = d.cuda()
            analog_tile.update(x, d)

            hidden_par_after = analog_tile.get_hidden_parameters()

            self.assertTensorAlmostEqual(hidden_par['hidden_weights_0'],
                                         hidden_par_after['hidden_weights_0'])
            self.assertNotAlmostEqualTensor(hidden_par['hidden_weights_1'],
                                            hidden_par_after['hidden_weights_1'])

    def test_input_range(self):
        """Tests whether input range is applied"""
        rpu_config = self.get_rpu_config()

        if not isinstance(rpu_config, PrePostProcessingRPU):
            raise SkipTest('This device does not support input range learning')

        if hasattr(rpu_config, 'forward'):
            rpu_config.forward.noise_management = NoiseManagementType.NONE
            rpu_config.forward.bound_management = BoundManagementType.NONE
        rpu_config.pre_post.input_range.enable = True
        rpu_config.pre_post.input_range.init_from_data = 0
        rpu_config.pre_post.input_range.init_value = 3.0
        rpu_config.pre_post.input_range.manage_output_clipping = False

        analog_tile = self.get_tile(3, 3, rpu_config=rpu_config, bias=True)

        inputs = 5 * ones((3, 3))
        outputs = analog_tile.pre_forward(inputs, 1, False)
        outputs = analog_tile.post_forward(outputs, 1, False)
        self.assertEqual(outputs.max().item(), 3.0)

        inputs = -5 * ones((3, 3))
        outputs = analog_tile.pre_forward(inputs, 1, False)
        outputs = analog_tile.post_forward(outputs, 1, False)
        self.assertEqual(outputs.max().item(), -3.0)

    def test_cuda_to_cpu(self):
        """ test the copy to CPU from cuda """

        if not self.use_cuda or SKIP_CUDA_TESTS:
            raise SkipTest('CUDA tile needed')

        out_size = 2
        in_size = 3

        cuda_analog_tile = self.get_tile(out_size, in_size, bias=True)
        self.assertIsInstance(cuda_analog_tile.tile, self.simulator_tile_class)

        learning_rate = 0.123
        # Use [out_size, in_size] weights.
        weights = Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        biases = Tensor([-0.1, -0.2])

        # Set some properties in the simulators.Tile.
        cuda_analog_tile.set_learning_rate(0.123)
        cuda_analog_tile.set_weights(weights, biases)

        hidden_parameters = cuda_analog_tile.get_hidden_parameters()
        field = self.first_hidden_field
        # Update one of the values of the hidden parameters.
        if field:
            # set higher as default otherwise hidden weight might change
            hidden_parameters[field][1][1] = 0.8
        cuda_analog_tile.set_hidden_parameters(hidden_parameters)

        self.assertEqual(cuda_analog_tile.is_cuda, True)
        self.assertEqual(cuda_analog_tile.tile.__class__, self.simulator_tile_class)
        analog_tile = cuda_analog_tile.cpu()
        self.assertEqual(analog_tile.is_cuda, False)
        self.assertNotEqual(analog_tile.tile.__class__, self.simulator_tile_class)
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

        # Check that the change was propagated to the tile.
        new_hidden_parameters = cuda_analog_tile.get_hidden_parameters()

        # Compare old and new hidden parameters tensors.
        for (_, old), (_, new) in zip(hidden_parameters.items(),
                                      new_hidden_parameters.items()):
            self.assertTrue(old.allclose(new))

        if field:
            self.assertEqual(new_hidden_parameters[field][1][1], 0.8)

    def test_cpu_to_cuda(self):
        """ test the copy to CUDA from CPU"""

        if self.use_cuda or SKIP_CUDA_TESTS:
            raise SkipTest('CUDA tile needed')

        out_size = 2
        in_size = 3

        cpu_analog_tile = self.get_tile(out_size, in_size, bias=True)
        self.assertIsInstance(cpu_analog_tile.tile, self.simulator_tile_class)

        learning_rate = 0.123
        # Use [out_size, in_size] weights.
        weights = Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        biases = Tensor([-0.1, -0.2])

        # Set some properties in the simulators.Tile.
        cpu_analog_tile.set_learning_rate(0.123)
        cpu_analog_tile.set_weights(weights, biases)

        hidden_parameters = cpu_analog_tile.get_hidden_parameters()
        field = self.first_hidden_field
        # Update one of the values of the hidden parameters.
        if field:
            # set higher as default otherwise hidden weight might change
            hidden_parameters[field][1][1] = 0.8
        cpu_analog_tile.set_hidden_parameters(hidden_parameters)

        self.assertEqual(cpu_analog_tile.is_cuda, False)
        self.assertEqual(cpu_analog_tile.tile.__class__, self.simulator_tile_class)
        analog_tile = cpu_analog_tile.cuda()
        self.assertEqual(analog_tile.is_cuda, True)
        self.assertNotEqual(analog_tile.tile.__class__, self.simulator_tile_class)

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

        # Check that the change was propagated to the tile.
        new_hidden_parameters = cpu_analog_tile.get_hidden_parameters()

        # Compare old and new hidden parameters tensors.
        for (_, old), (_, new) in zip(hidden_parameters.items(),
                                      new_hidden_parameters.items()):
            self.assertTrue(old.allclose(new))

        if field:
            self.assertEqual(new_hidden_parameters[field][1][1], 0.8)

    def test_program_weights(self):
        """Tests whether weight programming is performed"""
        rpu_config = self.get_rpu_config()
        if hasattr(rpu_config, 'forward'):
            rpu_config.forward.is_perfect = True  # avoid additional reading noise
        analog_tile = self.get_tile(2, 3, rpu_config=rpu_config, bias=True)

        weights = Tensor([[0.1, 0.2, 0.3], [0.4, -0.5, -0.6]])
        biases = Tensor([-0.1, 0.2])
        w_amax = 0.6

        analog_tile.set_learning_rate(0.123)
        analog_tile.set_weights(weights, biases)

        tolerance = 0.05
        analog_tile.program_weights()
        tile_weights, tile_biases = analog_tile.get_weights()

        self.assertNotAlmostEqualTensor(tile_weights, weights)
        if analog_tile.bias:
            self.assertNotAlmostEqualTensor(tile_biases, biases)

        # but should be close
        if analog_tile.bias:
            deviation = (tile_biases - biases).abs().sum() + (tile_weights - weights).abs().sum()
            deviation /= weights.numel() + biases.numel()
            self.assertTrue(deviation / w_amax < tolerance)
        else:
            self.assertTrue((tile_weights - weights).abs().mean() / w_amax < tolerance)

    def test_read_weights(self):
        """Tests whether weight reading is performed"""
        rpu_config = self.get_rpu_config()
        analog_tile = self.get_tile(2, 3, rpu_config=rpu_config, bias=True)

        weights = Tensor([[0.1, 0.2, 0.3], [0.4, -0.5, -0.6]])
        biases = Tensor([-0.1, 0.2])
        w_amax = 0.6

        analog_tile.set_weights(weights, biases)

        tile_weights, tile_biases = analog_tile.read_weights()

        tolerance = 0.1
        self.assertTrue((tile_weights - weights).abs().mean() / w_amax < tolerance)
        if analog_tile.bias:
            self.assertTrue((tile_biases - biases).abs().mean() / w_amax < tolerance)

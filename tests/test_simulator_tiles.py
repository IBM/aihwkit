# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=too-many-locals

"""Tests for the high level simulator devices functionality."""

from unittest import SkipTest

from pytest import mark
from torch import Tensor, zeros, ones, full, manual_seed
from torch.testing import assert_close

from aihwkit.exceptions import ArgumentError, TileModuleError
from aihwkit.simulator.configs.configs import SingleRPUConfig, UnitCellRPUConfig
from aihwkit.simulator.configs.compounds import VectorUnitCell, ReferenceUnitCell
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.simulator.parameters.enums import (
    AnalogMVType,
    VectorUnitCellUpdatePolicy,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.configs.configs import PrePostProcessingRPU, FloatingPointRPUConfig
from aihwkit.simulator.rpu_base import tiles
from aihwkit.simulator.tiles.analog import AnalogTile
from aihwkit.simulator.tiles.transfer import TransferSimulatorTile

from .helpers.decorators import parametrize_over_tiles
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import (
    FloatingPoint,
    Ideal,
    ConstantStep,
    LinearStep,
    SoftBounds,
    ExpStep,
    Vector,
    OneSided,
    Transfer,
    BufferedTransfer,
    TorchTransfer,
    MixedPrecision,
    PiecewiseStep,
    PiecewiseStepCuda,
    Inference,
    Reference,
    FloatingPointCuda,
    IdealCuda,
    ConstantStepCuda,
    LinearStepCuda,
    SoftBoundsCuda,
    ExpStepCuda,
    VectorCuda,
    OneSidedCuda,
    TransferCuda,
    BufferedTransferCuda,
    TorchTransferCuda,
    InferenceCuda,
    ReferenceCuda,
    MixedPrecisionCuda,
    PowStep,
    PowStepCuda,
    PowStepReference,
    PowStepReferenceCuda,
    SoftBoundsReference,
    SoftBoundsReferenceCuda,
    TorchInference,
    TorchInferenceCuda,
    QuantizedTorchInference,
)
from .helpers.testcases import SKIP_CUDA_TESTS

TOL = 1e-6


@parametrize_over_tiles(
    [
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
        TorchTransfer,
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
        TorchTransferCuda,
        MixedPrecisionCuda,
        InferenceCuda,
        ReferenceCuda,
    ]
)
class TileTest(ParametrizedTestCase):
    """Test floating point tile."""

    def test_bias(self) -> None:
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
        self.assertAlmostEqual(
            analog_tile.get_learning_rate(), analog_tile.tile.get_learning_rate()
        )

        # Assert over weights and biases.
        tile_weights, tile_biases = analog_tile.get_weights()
        self.assertEqual(tile_weights.shape, (out_size, in_size))
        self.assertEqual(tile_biases.shape, (out_size,))
        self.assertTensorAlmostEqual(tile_weights, weights)
        self.assertTensorAlmostEqual(tile_biases, biases)

    def test_no_bias(self) -> None:
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
        self.assertAlmostEqual(
            analog_tile.get_learning_rate(), analog_tile.tile.get_learning_rate()
        )

        # Assert over weights and biases.
        tile_weights, tile_biases = analog_tile.get_weights()
        self.assertEqual(tuple(tile_weights.shape), (out_size, in_size))
        self.assertEqual(tile_biases, None)
        self.assertTensorAlmostEqual(tile_weights, weights)

    def test_get_hidden_parameters(self) -> None:
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
            self.assertEqual(hidden_parameters[field].shape, (4, 5))
            self.assertTrue(all(abs(val - 0.6) < TOL for val in hidden_parameters[field].flatten()))

    def test_set_hidden_parameters(self) -> None:
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
        for (_, old), (_, new) in zip(hidden_parameters.items(), new_hidden_parameters.items()):
            self.assertTrue(old.allclose(new))

        if field:
            self.assertEqual(new_hidden_parameters[field][1][1], 0.8)

    def test_post_update_step_diffuse(self) -> None:
        """Tests whether post update diffusion is performed"""
        rpu_config = self.get_rpu_config()

        if not hasattr(rpu_config.device, "diffusion"):
            if hasattr(rpu_config.device, "unit_cell_devices") and hasattr(
                rpu_config.device.unit_cell_devices[-1], "diffusion"
            ):
                rpu_config.device.unit_cell_devices[-1].diffusion = 0.323
            else:
                raise SkipTest("This device does not support diffusion")
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
        if analog_tile.analog_bias:
            self.assertNotAlmostEqualTensor(tile_biases, biases)
        else:
            self.assertTensorAlmostEqual(tile_biases, biases)

    def test_post_update_step_lifetime(self) -> None:
        """Tests whether post update decay is performed"""
        rpu_config = self.get_rpu_config()

        if not hasattr(rpu_config.device, "lifetime"):
            if hasattr(rpu_config.device, "unit_cell_devices") and hasattr(
                rpu_config.device.unit_cell_devices[-1], "lifetime"
            ):
                for idx, _ in enumerate(rpu_config.device.unit_cell_devices):
                    rpu_config.device.unit_cell_devices[idx].lifetime = 100.0
            else:
                raise SkipTest("This device does not support lifetime")
        else:
            rpu_config.device.lifetime = 100.0

        analog_tile = self.get_tile(2, 3, rpu_config=rpu_config, bias=True)

        weights = Tensor([[0.1, 0.2, 0.3], [0.4, -0.5, -0.6]])
        biases = Tensor([-0.1, 0.2])

        analog_tile.set_learning_rate(0.123)
        analog_tile.set_weights(weights, biases)

        analog_tile.post_update_step()

        tile_weights, tile_biases = analog_tile.get_weights()

        self.assertNotAlmostEqualTensor(tile_weights, weights)
        if analog_tile.analog_bias:
            self.assertNotAlmostEqualTensor(tile_biases, biases)
        else:
            self.assertTensorAlmostEqual(tile_biases, biases)

    def test_set_hidden_update_index(self) -> None:
        """Tests hidden update index"""
        rpu_config = self.get_rpu_config()

        if rpu_config.tile_class != AnalogTile:
            raise SkipTest("Not an AnalogTile")

        if not isinstance(rpu_config, UnitCellRPUConfig) or not isinstance(
            rpu_config.device, (VectorUnitCell, ReferenceUnitCell)
        ):
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
            self.assertTensorAlmostEqual(hidden_par["hidden_weights_0"], weights_0)
            self.assertTensorAlmostEqual(hidden_par["hidden_weights_1"], zeros((2, 3)))

            # set weights index 1
            analog_tile.set_hidden_update_index(1)
            weights_1 = Tensor([[0.4, 0.1, 0.2], [0.5, -0.2, -0.1]])
            analog_tile.tile.set_weights(weights_1)

            hidden_par = analog_tile.get_hidden_parameters()

            self.assertTensorAlmostEqual(hidden_par["hidden_weights_0"], weights_0)
            self.assertTensorAlmostEqual(hidden_par["hidden_weights_1"], weights_1)

            # update
            analog_tile.set_hidden_update_index(1)
            x = Tensor([[0.1, 0.2, 0.3], [0.4, -0.5, -0.6]])
            d = Tensor([[0.5, 0.1], [0.4, -0.5]])
            if analog_tile.is_cuda:
                x = x.cuda()
                d = d.cuda()
            analog_tile.update(x, d)

            hidden_par_after = analog_tile.get_hidden_parameters()

            self.assertTensorAlmostEqual(
                hidden_par["hidden_weights_0"], hidden_par_after["hidden_weights_0"]
            )
            self.assertNotAlmostEqualTensor(
                hidden_par["hidden_weights_1"], hidden_par_after["hidden_weights_1"]
            )

    def test_input_range(self) -> None:
        """Tests whether input range is applied"""
        rpu_config = self.get_rpu_config()

        if not isinstance(rpu_config, PrePostProcessingRPU):
            raise SkipTest("This device does not support input range learning")

        if hasattr(rpu_config, "forward"):
            rpu_config.forward.noise_management = NoiseManagementType.NONE
            rpu_config.forward.bound_management = BoundManagementType.NONE
        rpu_config.pre_post.input_range.enable = True
        rpu_config.pre_post.input_range.init_from_data = 0
        rpu_config.pre_post.input_range.init_value = 3.0
        rpu_config.pre_post.input_range.manage_output_clipping = False

        analog_tile = self.get_tile(3, 3, rpu_config=rpu_config, bias=True)

        inputs = 5 * ones((3, 3))
        if self.use_cuda:
            inputs = inputs.cuda()
        outputs = analog_tile.pre_forward(inputs, 1, False)
        outputs = analog_tile.post_forward(outputs, 1, False)
        self.assertEqual(outputs.max().item(), 3.0)

        inputs = -5 * ones((3, 3))
        if self.use_cuda:
            inputs = inputs.cuda()

        outputs = analog_tile.pre_forward(inputs, 1, False)
        outputs = analog_tile.post_forward(outputs, 1, False)
        self.assertEqual(outputs.max().item(), -3.0)

    def test_cuda_to_cpu(self) -> None:
        """test the copy to CPU from cuda"""

        if not self.use_cuda or SKIP_CUDA_TESTS:
            raise SkipTest("CUDA tile needed")

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
        if analog_tile.tile.__class__ in tiles.__dict__.values():
            self.assertNotEqual(analog_tile.tile.__class__, self.simulator_tile_class)
        # Assert over learning rate.
        self.assertAlmostEqual(analog_tile.get_learning_rate(), learning_rate)
        self.assertAlmostEqual(
            analog_tile.get_learning_rate(), analog_tile.tile.get_learning_rate()
        )

        # Assert over weights and biases.
        tile_weights, tile_biases = analog_tile.get_weights()
        self.assertEqual(tile_weights.shape, (out_size, in_size))
        self.assertEqual(tile_biases.shape, (out_size,))
        self.assertTensorAlmostEqual(tile_weights, weights)
        self.assertTensorAlmostEqual(tile_biases, biases)

        # Check that the change was propagated to the tile.
        new_hidden_parameters = cuda_analog_tile.get_hidden_parameters()

        # Compare old and new hidden parameters tensors.
        for (_, old), (_, new) in zip(hidden_parameters.items(), new_hidden_parameters.items()):
            self.assertTrue(old.allclose(new))

        if field:
            self.assertEqual(new_hidden_parameters[field][1][1], 0.8)

    def test_cpu_to_cuda(self) -> None:
        """test the copy to CUDA from CPU"""

        if self.use_cuda or SKIP_CUDA_TESTS:
            raise SkipTest("CUDA tile needed")

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
        if analog_tile.tile.__class__ in tiles.__dict__.values():
            self.assertNotEqual(analog_tile.tile.__class__, self.simulator_tile_class)

        # Assert over learning rate.
        self.assertAlmostEqual(analog_tile.get_learning_rate(), learning_rate)
        self.assertAlmostEqual(
            analog_tile.get_learning_rate(), analog_tile.tile.get_learning_rate()
        )

        # Assert over weights and biases.
        tile_weights, tile_biases = analog_tile.get_weights()
        self.assertEqual(tile_weights.shape, (out_size, in_size))
        self.assertEqual(tile_biases.shape, (out_size,))
        self.assertTensorAlmostEqual(tile_weights, weights)
        self.assertTensorAlmostEqual(tile_biases, biases)

        # Check that the change was propagated to the tile.
        new_hidden_parameters = cpu_analog_tile.get_hidden_parameters()

        # Compare old and new hidden parameters tensors.
        for (_, old), (_, new) in zip(hidden_parameters.items(), new_hidden_parameters.items()):
            self.assertTrue(old.allclose(new))

        if field:
            self.assertEqual(new_hidden_parameters[field][1][1], 0.8)

    def test_program_weights(self) -> None:
        """Tests whether weight programming is performed"""
        rpu_config = self.get_rpu_config()
        if hasattr(rpu_config, "forward"):
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
        if analog_tile.analog_bias:
            self.assertNotAlmostEqualTensor(tile_biases, biases)

        # but should be close
        if analog_tile.analog_bias:
            deviation = (tile_biases - biases).abs().sum() + (tile_weights - weights).abs().sum()
            deviation /= weights.numel() + biases.numel()
            self.assertTrue(deviation / w_amax < tolerance)
        else:
            self.assertTrue((tile_weights - weights).abs().mean() / w_amax < tolerance)

    def test_read_weights(self) -> None:
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
        if analog_tile.analog_bias:
            self.assertTrue((tile_biases - biases).abs().mean() / w_amax < tolerance)

    def test_dump_extra(self) -> None:
        """Tests whether weight reading is performed"""

        rpu_config = self.get_rpu_config()
        analog_tile = self.get_tile(2, 3, rpu_config=rpu_config, bias=True)

        state = analog_tile.tile.dump_extra()

        self.assertTrue(len(state) > 0)
        # try loading again (will fail if keys are not found)
        analog_tile.tile.load_extra(state, True)

        non_empty_keys = [key for key in state.keys() if len(state[key]) > 0]
        if len(non_empty_keys) == 0:
            raise SkipTest("No non empty keys")
        del state[non_empty_keys[-1]]
        with self.assertRaises(RuntimeError):
            analog_tile.tile.load_extra(state, True)

    def test_replace_rpu_config(self) -> None:
        """Tests whether it is possible to replace the RPUConfig"""
        rpu_config = self.get_rpu_config()
        if not hasattr(rpu_config, "forward") or self.simulator_tile_class == TransferSimulatorTile:
            raise SkipTest("No forward")

        rpu_config.forward.is_perfect = False
        rpu_config.forward.out_noise = 0.123
        analog_tile = self.get_tile(2, 3, rpu_config=rpu_config, bias=True)

        # check C++ tile
        meta_pars = analog_tile.tile.get_meta_parameters()
        self.assertAlmostEqual(meta_pars.forward_io.out_noise, 0.123)
        with self.assertRaises(TileModuleError):
            analog_tile.replace_with(FloatingPointRPUConfig)

        # dummy check
        rpu_config.forward.out_noise = 0.234
        self.assertNotAlmostEqual(analog_tile.rpu_config.forward.out_noise, 0.234)

        # replace
        analog_tile.replace_with(rpu_config)
        meta_pars = analog_tile.tile.get_meta_parameters()
        self.assertAlmostEqual(meta_pars.forward_io.out_noise, 0.234)
        self.assertAlmostEqual(analog_tile.rpu_config.forward.out_noise, 0.234)

    def test_replace_rpu_config_to(self) -> None:
        """Tests whether it is possible to replace the RPUConfig with to"""
        rpu_config = self.get_rpu_config()
        if not hasattr(rpu_config, "forward") or self.simulator_tile_class == TransferSimulatorTile:
            raise SkipTest("No forward")

        rpu_config.forward.out_noise = 0.123
        analog_tile = self.get_tile(2, 3, rpu_config=rpu_config, bias=True)

        # check C++ tile
        meta_pars = analog_tile.tile.get_meta_parameters()
        self.assertAlmostEqual(meta_pars.forward_io.out_noise, 0.123)
        with self.assertRaises(TileModuleError):
            analog_tile.replace_with(FloatingPointRPUConfig)

        # dummy check
        rpu_config.forward.out_noise = 0.234
        self.assertNotAlmostEqual(analog_tile.rpu_config.forward.out_noise, 0.234)

        # replace
        analog_tile.to(rpu_config)
        meta_pars = analog_tile.tile.get_meta_parameters()
        self.assertAlmostEqual(meta_pars.forward_io.out_noise, 0.234)
        self.assertAlmostEqual(analog_tile.rpu_config.forward.out_noise, 0.234)

        rpu_config.forward.out_noise = 0.54
        analog_tile.to(rpu_config=rpu_config)
        meta_pars = analog_tile.tile.get_meta_parameters()
        self.assertAlmostEqual(meta_pars.forward_io.out_noise, 0.54)
        self.assertAlmostEqual(analog_tile.rpu_config.forward.out_noise, 0.54)


@parametrize_over_tiles(
    [
        ConstantStep,
        ConstantStepCuda,
        Inference,
        InferenceCuda,
        TorchInference,
        TorchInferenceCuda,
        QuantizedTorchInference,
    ]
)
class TileForwardBackwardTest(ParametrizedTestCase):
    """Test some forward aspects."""

    def test_set_forward_out_noise_std(self) -> None:
        """Test setting forward parameters."""
        manual_seed(123)

        rpu_config = self.get_rpu_config()
        rpu_config.forward.is_perfect = False
        rpu_config.forward.out_noise_std = 1.0
        rpu_config.forward.out_noise = 0.1
        rpu_config.forward.w_noise = 0.0

        analog_tile = self.get_tile(2, 2, rpu_config=rpu_config)

        weights = Tensor([[0.1, 0.2], [0.4, -0.5]])
        biases = Tensor([-0.1, 0.2])

        analog_tile.set_learning_rate(0.123)
        analog_tile.set_weights(weights, biases)

        forward_parameters = analog_tile.get_forward_parameters()

        field = "out_noise_values"
        forward_parameters[field][1] = 0.0
        forward_parameters[field][0] = 0.123

        analog_tile.set_forward_parameters(forward_parameters)

        # Check that the change was propagated to the tile.
        new_forward_parameters = analog_tile.get_forward_parameters()
        # Compare old and new hidden parameters tensors.
        for (_, old), (_, new) in zip(forward_parameters.items(), new_forward_parameters.items()):
            self.assertTrue(old.allclose(new))

        self.assertEqual(new_forward_parameters[field][0], 0.123)

        inputs = Tensor([[-0.1, 0.4], [-0.5, 0.1]])
        if analog_tile.is_cuda:
            inputs = inputs.cuda()
        y_1_lst = []
        for _ in range(10):
            y_1_lst.append(analog_tile(inputs))
        y_2_lst = []
        for _ in range(10):
            y_2_lst.append(analog_tile(inputs))
        self.assertTensorAlmostEqual(y_1_lst[-1][:, 1], y_2_lst[-1][:, 1])
        self.assertNotAlmostEqualTensor(y_1_lst[-1][:, 0], y_2_lst[-1][:, 0])

        with self.assertRaises(ArgumentError):
            forward_parameters["_not_existent"] = Tensor([1.0])
            analog_tile.set_forward_parameters(forward_parameters)


def _forward_bm_tile(
    weights: Tensor,
    inp_res: float = 0.0,
    test_negative_bound: bool = False,
    mv_type: AnalogMVType = AnalogMVType.ONE_PASS,
    use_cuda: bool = True,
) -> AnalogTile:
    """Build a deterministic native tile with iterative forward BM."""
    forward = IOParameters(
        bound_management=BoundManagementType.ITERATIVE,
        noise_management=NoiseManagementType.NONE,
        inp_res=inp_res,
        out_res=0.0,
        out_noise=0.0,
        inp_bound=1.0,
        out_bound=1.0,
        bm_test_negative_bound=test_negative_bound,
        mv_type=mv_type,
    )
    config = SingleRPUConfig(
        device=ConstantStepDevice(w_min=-1.0, w_max=1.0, w_min_dtod=0.0, w_max_dtod=0.0),
        forward=forward,
    )
    tile = AnalogTile(weights.shape[0], weights.shape[1], config)
    tile.set_weights(weights)
    return tile.cuda() if use_cuda else tile


def _forward_bm_output(tile: AnalogTile, inputs: Tensor) -> Tensor:
    """Run native forward on CUDA and return its result on CPU."""
    return tile.tile.forward(inputs.to(tile.device)).cpu()


@mark.skipif(SKIP_CUDA_TESTS, reason="CUDA unavailable")
def test_cuda_batch_uses_incremental_bm_factor_without_nm() -> None:
    """Identical vectors must not depend on selecting the single or batch CUDA kernel."""
    weights = full((1, 8), 0.75)
    tile = _forward_bm_tile(weights, inp_res=1 / 16)
    vector = full((1, 8), 0.625)

    single = _forward_bm_output(tile, vector)
    batch = _forward_bm_output(tile, vector.repeat(4, 1))

    # DAC step delta = 2 * inp_bound * inp_res = 1 / 8. For BM scale s,
    # y_raw(s) = 8 * 0.75 * Q_delta(0.625 / s). The first two attempts saturate:
    #   s=1: y_raw = 8 * 0.75 * 0.625 = 3.75 > 1,
    #   s=2: y_raw = 8 * 0.75 * 0.375 = 2.25 > 1.
    # The third attempt succeeds and restores its scale in the final output:
    #   s=4: y_raw = 8 * 0.75 * 0.125 = 0.75; y = s * y_raw = 3.0.
    assert_close(single, full((1, 1), 3.0), atol=0, rtol=0)
    assert_close(batch, single.repeat(4, 1), atol=0, rtol=0)


@mark.skipif(SKIP_CUDA_TESTS, reason="CUDA unavailable")
@mark.parametrize("test_negative_bound,expected", [(True, -2.25), (False, -1.0)])
def test_cuda_single_vector_tests_negative_bound(
    test_negative_bound: bool, expected: float
) -> None:
    """Negative-bound handling must agree between single-vector and batch CUDA kernels."""
    weights = full((8, 3), -0.75)
    tile = _forward_bm_tile(weights, test_negative_bound=test_negative_bound)
    vector = ones(1, 3)

    single = _forward_bm_output(tile, vector)
    batch = _forward_bm_output(tile, vector.repeat(4, 1))

    # Before the ADC bound, each result is y = 3 * 1 * (-0.75) = -2.25.
    # When enabled, negative-bound testing retries and restores -2.25;
    # otherwise the first pass is accepted with its ADC-clipped value of -1.0.
    assert_close(batch, full((4, 8), expected), atol=1e-6, rtol=0)
    assert_close(single, batch[:1], atol=1e-6, rtol=0)


def test_cpu_ignored_negative_bound_does_not_hide_positive_clipping() -> None:
    """An ignored negative saturation must not clear an earlier positive failure."""
    weights = full((2, 8), 0.75)
    weights[1] *= -1
    tile = _forward_bm_tile(weights, use_cuda=False)
    inputs = ones(1, 8)

    actual = tile.tile.forward(inputs)

    # The first pass clips [6, -6] to [1, -1]. Although the negative saturation
    # is ignored, the preceding positive saturation must still make BM retry.
    assert_close(actual, Tensor([[6.0, -6.0]]), atol=1e-6, rtol=0)


@mark.skipif(SKIP_CUDA_TESTS, reason="CUDA unavailable")
@mark.parametrize(
    "mv_type", [AnalogMVType.POS_NEG_SEPARATE, AnalogMVType.POS_NEG_SEPARATE_DIGITAL_SUM]
)
def test_cuda_split_mvm_retries_restore_input_buffer(mv_type: AnalogMVType) -> None:
    """Each split-MVM retry must rebuild its input from the original buffer."""
    weights = full((3, 8), 0.75)
    tile = _forward_bm_tile(weights, mv_type=mv_type)
    inputs = ones(2, 8)
    inputs[:, -1] = -1.0

    actual = _forward_bm_output(tile, inputs)

    # Each output is (7 * 1 + 1 * -1) * 0.75 = 4.5. The first pass clips,
    # so this also verifies that split MVMs preserve both signs on every retry.
    assert_close(actual, full((2, 3), 4.5), atol=1e-6, rtol=0)

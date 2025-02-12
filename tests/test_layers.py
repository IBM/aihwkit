# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for general functionality of layers."""

# pylint: disable=too-few-public-methods
from tempfile import TemporaryFile
from unittest import SkipTest

from torch import Tensor, device, load, save, zeros
from torch.cuda import current_device, device_count

from aihwkit.nn import AnalogSequential
from aihwkit.simulator.rpu_base import tiles
from aihwkit.simulator.tiles.transfer import TransferSimulatorTile
from aihwkit.simulator.tiles.torch_tile import TorchSimulatorTile

from .helpers.decorators import parametrize_over_layers
from .helpers.layers import (
    Linear,
    Conv1d,
    Conv2d,
    Conv3d,
    LinearCuda,
    Conv1dCuda,
    Conv2dCuda,
    Conv3dCuda,
)
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import ConstantStep, Inference, Custom, TorchTransfer, TorchInference
from .helpers.testcases import SKIP_CUDA_TESTS

CUDA_TILE_CLASS = tiles.CudaAnalogTile if hasattr(tiles, "CudaAnalogTile") else None

TILE_CLASSES = {
    tiles.AnalogTile: CUDA_TILE_CLASS,
    CUDA_TILE_CLASS: CUDA_TILE_CLASS,
    TransferSimulatorTile: TransferSimulatorTile,
    TorchSimulatorTile: TorchSimulatorTile,
}


@parametrize_over_layers(
    layers=[Linear, Conv1d, Conv2d, Conv3d, LinearCuda, Conv1dCuda, Conv2dCuda, Conv3dCuda],
    tiles=[ConstantStep, Inference, TorchTransfer, TorchInference],
    biases=["digital"],
)
class AnalogLayerMoveTest(ParametrizedTestCase):
    """Analog layers abstraction tests."""

    def test_sequential_move_to_cuda(self):
        """Test moving AnalogSequential to cuda (from CPU)."""
        if SKIP_CUDA_TESTS:
            raise SkipTest("not compiled with CUDA support")

        # Map the original tile classes to the expected ones after `cuda()`.
        layer = self.get_layer()
        analog_tile = next(layer.analog_tiles())
        expected_class = TILE_CLASSES[analog_tile.tile.__class__]
        expected_device = device("cuda", current_device())

        # Create a container and move to cuda.
        model = AnalogSequential(layer)
        model.cuda()

        self.assertEqual(analog_tile.device, expected_device)
        self.assertEqual(analog_tile.get_analog_ctx().data.device, expected_device)
        if analog_tile.shared_weights is not None:
            self.assertEqual(analog_tile.shared_weights.data.device, expected_device)
            self.assertEqual(
                analog_tile.shared_weights.data.size()[0], analog_tile.tile.get_x_size()
            )
            self.assertEqual(
                analog_tile.shared_weights.data.size()[1], analog_tile.tile.get_d_size()
            )

        # Assert the tile has been moved to cuda.
        self.assertIsInstance(analog_tile.tile, expected_class)

    def test_sequential_move_to_cuda_via_to(self):
        """Test moving AnalogSequential to cuda (from CPU), using ``.to()``."""
        if SKIP_CUDA_TESTS:
            raise SkipTest("not compiled with CUDA support")

        # Map the original tile classes to the expected ones after `cuda()`.
        layer = self.get_layer()
        analog_tile = next(layer.analog_tiles())
        expected_class = TILE_CLASSES[analog_tile.tile.__class__]
        expected_device = device("cuda", current_device())

        # Create a container and move to cuda.
        model = AnalogSequential(layer)
        model.to(device("cuda"))

        self.assertEqual(analog_tile.device, expected_device)
        self.assertEqual(analog_tile.get_analog_ctx().data.device, expected_device)
        if analog_tile.shared_weights is not None:
            self.assertEqual(analog_tile.shared_weights.data.device, expected_device)
            self.assertEqual(
                analog_tile.shared_weights.data.size()[0], analog_tile.tile.get_x_size()
            )
            self.assertEqual(
                analog_tile.shared_weights.data.size()[1], analog_tile.tile.get_d_size()
            )

        # Assert the tile has been moved to cuda.
        self.assertIsInstance(analog_tile.tile, expected_class)

    def test_sequential_move_to_cuda_via_to_multiple_gpus(self):
        """Test moving AnalogSequential to cuda (from CPU), using ``.to()``."""
        if SKIP_CUDA_TESTS:
            raise SkipTest("not compiled with CUDA support")
        if device_count() < 2:
            raise SkipTest("Need at least two devices for this test")

        # Test whether it can move to GPU with index 1
        expected_device_num = 1

        layer = self.get_layer()
        analog_tile = next(layer.analog_tiles())
        if isinstance(analog_tile.tile, (tiles.CudaAnalogTile, tiles.CudaFloatingPointTile)):
            raise SkipTest("Layer is already on CUDA")

        expected_class = TILE_CLASSES[analog_tile.tile.__class__]
        expected_device = device("cuda", expected_device_num)

        # Create a container and move to cuda.
        model = AnalogSequential(layer)
        model.to(device("cuda", expected_device_num))

        self.assertEqual(analog_tile.device, expected_device)
        self.assertEqual(analog_tile.get_analog_ctx().data.device, expected_device)
        if analog_tile.shared_weights is not None:
            self.assertEqual(analog_tile.shared_weights.data.device, expected_device)
            self.assertEqual(
                analog_tile.shared_weights.data.size()[0], analog_tile.tile.get_x_size()
            )
            self.assertEqual(
                analog_tile.shared_weights.data.size()[1], analog_tile.tile.get_d_size()
            )

        # Assert the tile has been moved to cuda.
        self.assertIsInstance(analog_tile.tile, expected_class)

    def test_sequential_move_to_cuda_multiple_gpus(self):
        """Test moving AnalogSequential to cuda (from CPU), using ``.to()``."""
        if SKIP_CUDA_TESTS:
            raise SkipTest("not compiled with CUDA support")
        if device_count() < 2:
            raise SkipTest("Need at least two devices for this test")

        # Test whether it can move to GPU with index 1
        expected_device_num = 1

        layer = self.get_layer()
        analog_tile = next(layer.analog_tiles())

        expected_class = TILE_CLASSES[analog_tile.tile.__class__]
        expected_device = device("cuda", expected_device_num)

        # Create a container and move to cuda.
        model = AnalogSequential(layer)
        model.cuda(device("cuda", expected_device_num))

        self.assertEqual(analog_tile.device, expected_device)
        self.assertEqual(analog_tile.get_analog_ctx().data.device, expected_device)
        if analog_tile.shared_weights is not None:
            self.assertEqual(analog_tile.shared_weights.data.device, expected_device)
            self.assertEqual(
                analog_tile.shared_weights.data.size()[0], analog_tile.tile.get_x_size()
            )
            self.assertEqual(
                analog_tile.shared_weights.data.size()[1], analog_tile.tile.get_d_size()
            )

        # Assert the tile has been moved to cuda.
        self.assertIsInstance(analog_tile.tile, expected_class)

    def test_save_with_cuda(self):
        """Whether model is correctly reconstructed after saving"""
        if SKIP_CUDA_TESTS:
            raise SkipTest("not compiled with CUDA support")

        # Map the original tile classes to the expected ones after `cuda()`.
        layer = self.get_layer()

        model = AnalogSequential(layer)
        model.cuda()
        analog_tile = next(model.analog_tiles())
        expected_class = TILE_CLASSES[analog_tile.tile.__class__]

        with TemporaryFile() as file:
            save(model.state_dict(), file)
            # Create a new model and load its state dict.
            file.seek(0)
            checkpoint = load(file, weights_only=False)
        model.load_state_dict(checkpoint)

        expected_device = device("cuda", current_device())
        expected_class = TILE_CLASSES[analog_tile.tile.__class__]
        analog_tile = next(model.analog_tiles())

        self.assertEqual(analog_tile.device, expected_device)
        self.assertEqual(analog_tile.get_analog_ctx().data.device, expected_device)
        if analog_tile.shared_weights is not None:
            self.assertEqual(analog_tile.shared_weights.data.device, expected_device)
            self.assertEqual(
                analog_tile.shared_weights.data.size()[0], analog_tile.tile.get_x_size()
            )
            self.assertEqual(
                analog_tile.shared_weights.data.size()[1], analog_tile.tile.get_d_size()
            )

        # Assert the tile has been moved to cuda.
        self.assertIsInstance(analog_tile.tile, expected_class)


@parametrize_over_layers(
    layers=[Linear, Conv1d, Conv2d, Conv3d],
    tiles=[ConstantStep, Inference, TorchTransfer, TorchInference],
    biases=["analog", "digital", None],
)
class CpuAnalogLayerTest(ParametrizedTestCase):
    """Analog layers tests using CPU tiles as the source."""

    def test_sequential_move_to_cpu(self):
        """Test moving AnalogSequential to CPU (from CPU)."""
        layer = self.get_layer()

        # Create a container and move to cuda.
        model = AnalogSequential(layer)
        model.cpu()

        analog_tile = next(layer.analog_tiles())
        self.assertEqual(analog_tile.device, device("cpu"))
        self.assertEqual(analog_tile.get_analog_ctx().data.device, device("cpu"))

        if analog_tile.shared_weights is not None:
            self.assertEqual(analog_tile.shared_weights.data.device, device("cpu"))
            self.assertEqual(
                analog_tile.shared_weights.data.size()[0], analog_tile.tile.get_d_size()
            )
            self.assertEqual(
                analog_tile.shared_weights.data.size()[1], analog_tile.tile.get_x_size()
            )

    def test_sequential_move_to_cpu_via_to(self):
        """Test moving AnalogSequential to CPU (from CPU), using ``.to()``."""
        layer = self.get_layer()

        expected_device = device("cpu")
        # Create a container and move to cuda.
        model = AnalogSequential(layer)
        model.to(device("cpu"))

        analog_tile = next(layer.analog_tiles())
        self.assertEqual(analog_tile.device, expected_device)
        self.assertEqual(analog_tile.get_analog_ctx().data.device, expected_device)

        if analog_tile.shared_weights is not None:
            self.assertEqual(analog_tile.shared_weights.data.device, expected_device)
            self.assertEqual(
                analog_tile.shared_weights.data.size()[0], analog_tile.tile.get_d_size()
            )
            self.assertEqual(
                analog_tile.shared_weights.data.size()[1], analog_tile.tile.get_x_size()
            )


@parametrize_over_layers(
    layers=[Linear, Conv2d, LinearCuda, Conv2dCuda], tiles=[ConstantStep], biases=[None]
)
class AnalogLayerTest(ParametrizedTestCase):
    """Analog layers abstraction tests."""

    # pylint: disable=no-member

    def test_realistic_weights(self):
        """Test using realistic weights."""
        layer = self.get_layer()

        # 1. Set the layer weights and biases. (Exact writing / reading)
        user_weights = Tensor(layer.out_features, layer.in_features).uniform_(-0.5, 0.5)
        user_biases = Tensor(layer.out_features).uniform_(-0.5, 0.5)
        layer.set_weights(user_weights, user_biases, realistic=False)
        tile_weights, tile_biases = layer.get_weights()

        # Check that the tile weights are equal
        self.assertTensorAlmostEqual(user_weights, tile_weights)
        if self.analog_bias:
            self.assertTensorAlmostEqual(user_biases, tile_biases)

        # 2. Realistic writing
        layer.set_weights(user_weights, user_biases, realistic=True)
        tile_weights, tile_biases = layer.get_weights()

        # Check that the tile weights are not equal
        self.assertNotAlmostEqualTensor(user_weights, tile_weights)
        # but approximately correct
        self.assertTensorAlmostEqual((user_weights - tile_weights).mean(), zeros(1), decimal=1)
        if self.analog_bias:
            self.assertNotAlmostEqualTensor(user_biases, tile_biases)

        # 2. Realistic reading
        layer.set_weights(user_weights, user_biases, realistic=False)
        tile_weights, tile_biases = layer.get_weights(realistic=True)

        # Check that the tile weights are not equal
        self.assertNotAlmostEqualTensor(user_weights, tile_weights)
        if self.analog_bias:
            self.assertNotAlmostEqualTensor(user_biases, tile_biases)


@parametrize_over_layers(
    layers=[Linear, Conv2d, LinearCuda, Conv2dCuda],
    tiles=[Custom, TorchTransfer, TorchInference],
    biases=["digital", None],
)
class CustomTileTest(ParametrizedTestCase):
    """Test for analog layers using custom tiles."""

    def test_custom_tile(self):
        """Test using a custom tile with analog layers."""
        # Create the layer, which uses the custom RPUConfig
        rpu_config = self.get_rpu_config()
        layer = self.get_layer(rpu_config=rpu_config)

        # Assert that the internal analog tile is of the correct type
        analog_tile = next(layer.analog_tiles())
        self.assertIsInstance(analog_tile, rpu_config.tile_class)

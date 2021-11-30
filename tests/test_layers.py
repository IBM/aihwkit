# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for general functionality of layers."""

# pylint: disable=too-few-public-methods
from tempfile import TemporaryFile
from unittest import SkipTest

from torch import Tensor, device, load, save
from torch.cuda import current_device, device_count

from aihwkit.nn import AnalogSequential
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.rpu_base import cuda, tiles
from aihwkit.simulator.tiles import AnalogTile

from .helpers.decorators import parametrize_over_layers
from .helpers.layers import (
    Linear, Conv1d, Conv2d, Conv3d,
    LinearCuda, Conv1dCuda, Conv2dCuda, Conv3dCuda
)
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import ConstantStep, Inference


@parametrize_over_layers(
    layers=[Linear, Conv1d, Conv2d, Conv3d,
            LinearCuda, Conv1dCuda, Conv2dCuda, Conv3dCuda],
    tiles=[ConstantStep],
    biases=['analog', None]
)
class AnalogLayerTest(ParametrizedTestCase):
    """Analog layers abstraction tests."""
    # pylint: disable=no-member

    def test_realistic_weights(self):
        """Test using realistic weights."""
        layer = self.get_layer(realistic_read_write=True)

        shape = layer.weight.shape
        # Check that the tile weights are equal from the layer weights, as
        # the weights are synced after being set.
        tile_weights, tile_biases = layer.analog_tile.get_weights()
        self.assertTensorAlmostEqual(layer.weight, tile_weights.reshape(shape))
        if self.analog_bias:
            self.assertTensorAlmostEqual(layer.bias, tile_biases)

        # 1. Set the layer weights and biases.
        user_weights = Tensor(layer.out_features, layer.in_features).uniform_(-0.5, 0.5)
        user_biases = Tensor(layer.out_features).uniform_(-0.5, 0.5)
        layer.set_weights(user_weights, user_biases)

        # Check that the tile weights are equal from the layer weights, as
        # the weights are synced after being set.
        tile_weights, tile_biases = layer.analog_tile.get_weights()
        self.assertTensorAlmostEqual(layer.weight, tile_weights.reshape(shape))
        if self.analog_bias:
            self.assertTensorAlmostEqual(layer.bias, tile_biases)

        # Check that the tile weights are different than the user-specified
        # weights, as it is realistic.
        self.assertNotAlmostEqualTensor(user_weights, tile_weights.reshape(shape))
        if self.analog_bias:
            self.assertNotAlmostEqualTensor(user_biases, tile_biases)

        # 2. Get the layer weights and biases.
        gotten_weights, gotten_biases = layer.get_weights()

        # Check that the tile weights are different than the gotten
        # weights, as it is realistic.
        self.assertNotAlmostEqualTensor(gotten_weights, tile_weights.reshape(shape))
        if self.analog_bias:
            self.assertNotAlmostEqualTensor(gotten_biases, tile_biases)

    def test_not_realistic_weights(self):
        """Test using non realistic weights."""
        layer = self.get_layer(realistic_read_write=False)

        shape = layer.weight.shape
        # Check that the tile weights are equal from the layer weights, as
        # the weights are synced after being set.
        tile_weights, tile_biases = layer.analog_tile.get_weights()
        self.assertTensorAlmostEqual(layer.weight, tile_weights.reshape(shape))
        if self.analog_bias:
            self.assertTensorAlmostEqual(layer.bias, tile_biases)

        # 1. Set the layer weights and biases.
        user_weights = Tensor(layer.out_features, layer.in_features).uniform_(-0.5, 0.5)
        user_biases = Tensor(layer.out_features).uniform_(-0.5, 0.5)
        layer.set_weights(user_weights, user_biases)

        # Check that the tile weights are equal from the layer weights, as
        # the weights are synced after being set.
        tile_weights, tile_biases = layer.analog_tile.get_weights()
        self.assertTensorAlmostEqual(layer.weight, tile_weights.reshape(shape))
        if self.analog_bias:
            self.assertTensorAlmostEqual(layer.bias, tile_biases)

        # Check that the tile weights are equal to the user-specified
        # weights, as it is not realistic.
        self.assertTensorAlmostEqual(user_weights, tile_weights)
        if self.analog_bias:
            self.assertTensorAlmostEqual(user_biases, tile_biases)

        # 2. Get the layer weights and biases.
        gotten_weights, gotten_biases = layer.get_weights()

        # Check that the tile weights are equal than the gotten
        # weights, as it is not realistic.
        self.assertTensorAlmostEqual(gotten_weights, tile_weights)
        if self.analog_bias:
            self.assertTensorAlmostEqual(gotten_biases, tile_biases)


@parametrize_over_layers(
    layers=[Linear, Conv1d, Conv2d, Conv3d,
            LinearCuda, Conv1dCuda, Conv2dCuda, Conv3dCuda],
    tiles=[ConstantStep, Inference],
    biases=['analog', None]
)
class AnalogLayerMoveTest(ParametrizedTestCase):
    """Analog layers abstraction tests."""

    def test_sequential_move_to_cuda(self):
        """Test moving AnalogSequential to cuda (from CPU)."""
        if not cuda.is_compiled():
            raise SkipTest('not compiled with CUDA support')

        # Map the original tile classes to the expected ones after `cuda()`.
        tile_classes = {
            tiles.AnalogTile: tiles.CudaAnalogTile,
            tiles.CudaAnalogTile: tiles.CudaAnalogTile
        }

        layer = self.get_layer()
        expected_class = tile_classes[layer.analog_tile.tile.__class__]
        expected_device = device('cuda', current_device())

        # Create a container and move to cuda.
        model = AnalogSequential(layer)
        model.cuda()

        analog_tile = layer.analog_tile
        self.assertEqual(analog_tile.device, expected_device)
        self.assertEqual(analog_tile.get_analog_ctx().data.device, expected_device)
        if analog_tile.shared_weights is not None:
            self.assertEqual(analog_tile.shared_weights.data.device, expected_device)
            self.assertEqual(analog_tile.shared_weights.data.size()[0],
                             analog_tile.tile.get_x_size())
            self.assertEqual(analog_tile.shared_weights.data.size()[1],
                             analog_tile.tile.get_d_size())

        # Assert the tile has been moved to cuda.
        self.assertIsInstance(layer.analog_tile.tile, expected_class)

    def test_sequential_move_to_cuda_via_to(self):
        """Test moving AnalogSequential to cuda (from CPU), using ``.to()``."""
        if not cuda.is_compiled():
            raise SkipTest('not compiled with CUDA support')

        # Map the original tile classes to the expected ones after `cuda()`.
        tile_classes = {
            tiles.AnalogTile: tiles.CudaAnalogTile,
            tiles.CudaAnalogTile: tiles.CudaAnalogTile
        }

        layer = self.get_layer()
        expected_class = tile_classes[layer.analog_tile.tile.__class__]
        expected_device = device('cuda', current_device())

        # Create a container and move to cuda.
        model = AnalogSequential(layer)
        model.to(device('cuda'))

        analog_tile = layer.analog_tile
        self.assertEqual(analog_tile.device, expected_device)
        self.assertEqual(analog_tile.get_analog_ctx().data.device, expected_device)
        if analog_tile.shared_weights is not None:
            self.assertEqual(analog_tile.shared_weights.data.device, expected_device)
            self.assertEqual(analog_tile.shared_weights.data.size()[0],
                             analog_tile.tile.get_x_size())
            self.assertEqual(analog_tile.shared_weights.data.size()[1],
                             analog_tile.tile.get_d_size())

        # Assert the tile has been moved to cuda.
        self.assertIsInstance(layer.analog_tile.tile, expected_class)

    def test_sequential_move_to_cuda_via_to_multiple_gpus(self):
        """Test moving AnalogSequential to cuda (from CPU), using ``.to()``."""
        if not cuda.is_compiled():
            raise SkipTest('not compiled with CUDA support')
        if device_count() < 2:
            raise SkipTest('Need at least two devices for this test')

        # Map the original tile classes to the expected ones after `cuda()`.
        tile_classes = {
            tiles.AnalogTile: tiles.CudaAnalogTile,
            tiles.CudaAnalogTile: tiles.CudaAnalogTile
        }

        # Test whether it can move to GPU with index 1
        expected_device_num = 1

        layer = self.get_layer()
        if isinstance(layer.analog_tile.tile.__class__, (tiles.CudaAnalogTile,
                                                         tiles.CudaFloatingPointTile)):
            raise SkipTest('Layer is already on CUDA')

        expected_class = tile_classes[layer.analog_tile.tile.__class__]
        expected_device = device('cuda', expected_device_num)

        # Create a container and move to cuda.
        model = AnalogSequential(layer)
        model.to(device('cuda', expected_device_num))

        analog_tile = layer.analog_tile
        self.assertEqual(analog_tile.device, expected_device)
        self.assertEqual(analog_tile.get_analog_ctx().data.device, expected_device)
        if analog_tile.shared_weights is not None:
            self.assertEqual(analog_tile.shared_weights.data.device, expected_device)
            self.assertEqual(analog_tile.shared_weights.data.size()[0],
                             analog_tile.tile.get_x_size())
            self.assertEqual(analog_tile.shared_weights.data.size()[1],
                             analog_tile.tile.get_d_size())

        # Assert the tile has been moved to cuda.
        self.assertIsInstance(layer.analog_tile.tile, expected_class)

    def test_sequential_move_to_cuda_multiple_gpus(self):
        """Test moving AnalogSequential to cuda (from CPU), using ``.to()``."""
        if not cuda.is_compiled():
            raise SkipTest('not compiled with CUDA support')
        if device_count() < 2:
            raise SkipTest('Need at least two devices for this test')

        # Map the original tile classes to the expected ones after `cuda()`.
        tile_classes = {
            tiles.AnalogTile: tiles.CudaAnalogTile,
            tiles.CudaAnalogTile: tiles.CudaAnalogTile
        }

        # Test whether it can move to GPU with index 1
        expected_device_num = 1

        layer = self.get_layer()
        if isinstance(layer.analog_tile.tile.__class__, (tiles.CudaAnalogTile,
                                                         tiles.CudaFloatingPointTile)):
            raise SkipTest('Layer is already on CUDA')

        expected_class = tile_classes[layer.analog_tile.tile.__class__]
        expected_device = device('cuda', expected_device_num)

        # Create a container and move to cuda.
        model = AnalogSequential(layer)
        model.cuda(device('cuda', expected_device_num))

        analog_tile = layer.analog_tile
        self.assertEqual(analog_tile.device, expected_device)
        self.assertEqual(analog_tile.get_analog_ctx().data.device, expected_device)
        if analog_tile.shared_weights is not None:
            self.assertEqual(analog_tile.shared_weights.data.device, expected_device)
            self.assertEqual(analog_tile.shared_weights.data.size()[0],
                             analog_tile.tile.get_x_size())
            self.assertEqual(analog_tile.shared_weights.data.size()[1],
                             analog_tile.tile.get_d_size())

        # Assert the tile has been moved to cuda.
        self.assertIsInstance(layer.analog_tile.tile, expected_class)

    def test_save_with_cuda(self):
        """Whether model is correctly reconstructed after saving"""
        if not cuda.is_compiled():
            raise SkipTest('not compiled with CUDA support')

        # Map the original tile classes to the expected ones after `cuda()`.
        tile_classes = {
            tiles.AnalogTile: tiles.CudaAnalogTile,
            tiles.CudaAnalogTile: tiles.CudaAnalogTile
        }

        layer = self.get_layer()
        model = AnalogSequential(layer)
        model.cuda()
        with TemporaryFile() as file:
            save(model.state_dict(), file)
            # Create a new model and load its state dict.
            file.seek(0)
            checkpoint = load(file)
        model.load_state_dict(checkpoint)

        expected_device = device('cuda', current_device())
        expected_class = tile_classes[layer.analog_tile.tile.__class__]

        analog_tile = model[0].analog_tile
        self.assertEqual(analog_tile.device, expected_device)
        self.assertEqual(analog_tile.get_analog_ctx().data.device, expected_device)
        if analog_tile.shared_weights is not None:
            self.assertEqual(analog_tile.shared_weights.data.device, expected_device)
            self.assertEqual(analog_tile.shared_weights.data.size()[0],
                             analog_tile.tile.get_x_size())
            self.assertEqual(analog_tile.shared_weights.data.size()[1],
                             analog_tile.tile.get_d_size())

        # Assert the tile has been moved to cuda.
        self.assertIsInstance(layer.analog_tile.tile, expected_class)


@parametrize_over_layers(
    layers=[Linear, Conv1d, Conv2d, Conv3d],
    tiles=[ConstantStep, Inference],
    biases=['analog', 'digital', None]
)
class CpuAnalogLayerTest(ParametrizedTestCase):
    """Analog layers tests using CPU tiles as the source."""

    def test_sequential_move_to_cpu(self):
        """Test moving AnalogSequential to CPU (from CPU)."""
        layer = self.get_layer()

        # Create a container and move to cuda.
        model = AnalogSequential(layer)
        model.cpu()

        analog_tile = layer.analog_tile
        self.assertEqual(analog_tile.device, device('cpu'))
        self.assertEqual(analog_tile.get_analog_ctx().data.device, device('cpu'))

        if analog_tile.shared_weights is not None:
            self.assertEqual(analog_tile.shared_weights.data.device, device('cpu'))
            self.assertEqual(analog_tile.shared_weights.data.size()[0],
                             analog_tile.tile.get_d_size())
            self.assertEqual(analog_tile.shared_weights.data.size()[1],
                             analog_tile.tile.get_x_size())

        # Assert the tile is still on CPU.
        self.assertIsInstance(layer.analog_tile.tile, tiles.AnalogTile)

    def test_sequential_move_to_cpu_via_to(self):
        """Test moving AnalogSequential to CPU (from CPU), using ``.to()``."""
        layer = self.get_layer()

        expected_device = device('cpu')
        # Create a container and move to cuda.
        model = AnalogSequential(layer)
        model.to(device('cpu'))

        analog_tile = layer.analog_tile
        self.assertEqual(analog_tile.device, expected_device)
        self.assertEqual(analog_tile.get_analog_ctx().data.device, expected_device)

        if analog_tile.shared_weights is not None:
            self.assertEqual(analog_tile.shared_weights.data.device, expected_device)
            self.assertEqual(analog_tile.shared_weights.data.size()[0],
                             analog_tile.tile.get_d_size())
            self.assertEqual(analog_tile.shared_weights.data.size()[1],
                             analog_tile.tile.get_x_size())

        # Assert the tile is still on CPU.
        self.assertIsInstance(layer.analog_tile.tile, tiles.AnalogTile)


class CustomAnalogTile(AnalogTile):
    """Helper tile for ``CustomTileTest``."""


class CustomRPUConfig(SingleRPUConfig):
    """Helper rpu config for ``CustomTileTest``."""
    tile_class = CustomAnalogTile


class CustomTileTestHelper:
    """Helper tile for parametrizing during ``CustomTileTest``."""

    def get_rpu_config(self):
        """Return a RPU Config."""
        return CustomRPUConfig()


@parametrize_over_layers(
    layers=[Linear, Conv1d, Conv2d, Conv3d],
    tiles=[CustomTileTestHelper],
    biases=['analog', 'digital', None]
)
class CustomTileTest(ParametrizedTestCase):
    """Test for analog layers using custom tiles."""

    def test_custom_tile(self):
        """Test using a custom tile with analog layers."""
        # Create the layer, which uses `CustomRPUConfig`.
        layer = self.get_layer()

        # Assert that the internal analog tile is `CustomAnalogTile`.
        self.assertIsInstance(layer.analog_tile, CustomAnalogTile)

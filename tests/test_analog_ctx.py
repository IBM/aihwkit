# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=too-many-locals, no-member
"""Tests for AnalogContext data attribution (PR #717).

Verifies that analog_ctx.data reflects the actual weight matrix stored in the
tile, rather than being an empty scalar tensor.
"""

from unittest import SkipTest

from torch import zeros, randn, Tensor, Size, manual_seed
from torch.nn import Parameter
from torch.nn import Linear as TorchLinear, Sequential, Conv2d as TorchConv2d

from aihwkit.nn import AnalogLinear, AnalogConv2d
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim.context import AnalogContext
from aihwkit.simulator.configs import FloatingPointRPUConfig

from .helpers.decorators import parametrize_over_layers
from .helpers.layers import Linear, LinearCuda, LinearMapped, LinearMappedCuda
from .helpers.testcases import ParametrizedTestCase, SKIP_CUDA_TESTS
from .helpers.tiles import FloatingPoint, TorchInference


@parametrize_over_layers(
    layers=[Linear, LinearMapped],
    tiles=[FloatingPoint, TorchInference],
    biases=["analog", "digital", None],
)
class AnalogCtxDataAttributionTest(ParametrizedTestCase):
    """Tests that analog_ctx.data has the correct shape and values."""

    def _get_analog_tile(self, model):
        """Return the first analog tile from a model."""
        return next(model.analog_tiles())

    def test_ctx_data_shape_matches_weights(self):
        """analog_ctx.size() must return the tile weight shape, not torch.Size([])."""
        model = self.get_layer(in_features=4, out_features=6)
        tile = self._get_analog_tile(model)
        ctx = tile.analog_ctx

        # The old implementation returned torch.Size([]) — a scalar.
        # The new one must return the tile weight matrix shape.
        self.assertNotEqual(ctx.size(), Size([]))
        self.assertEqual(len(ctx.size()), 2)

        expected_rows = tile.out_size
        in_size = tile.in_size + (1 if tile.analog_bias else 0)
        self.assertEqual(ctx.size(), Size([expected_rows, in_size]))

    def test_ctx_data_values_match_tile_weights(self):
        """analog_ctx.data must reflect the actual tile weights."""
        model = self.get_layer(in_features=4, out_features=6)
        tile = self._get_analog_tile(model)

        weights_from_tile = tile.tile.get_weights()
        ctx_data = tile.analog_ctx.data

        self.assertEqual(ctx_data.shape, weights_from_tile.shape)

    def test_ctx_norm_is_meaningful(self):
        """analog_ctx.norm() should reflect the weight magnitude, not 1.0."""
        manual_seed(42)
        model = self.get_layer(in_features=4, out_features=6)
        tile = self._get_analog_tile(model)

        # With randomly initialized weights, the norm should be > 0
        # and should NOT be exactly 1.0 (which the old scalar ones(()) returned).
        norm_val = tile.analog_ctx.norm().item()
        self.assertGreater(norm_val, 0.0)

    def test_ctx_nonzero_works(self):
        """analog_ctx.nonzero() should return indices of nonzero weights."""
        model = self.get_layer(in_features=4, out_features=6)
        tile = self._get_analog_tile(model)

        # With random initialization, most weights are nonzero.
        nz = tile.analog_ctx.nonzero()
        self.assertGreater(len(nz), 0)

    def test_ctx_comparison_ops(self):
        """Comparison operators on analog_ctx should work on actual weights."""
        model = self.get_layer(in_features=4, out_features=6)
        tile = self._get_analog_tile(model)

        # Weights are initialized near zero with std ~1, so most are < 10.
        mask = tile.analog_ctx > 10
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.shape, tile.analog_ctx.shape)

    def test_ctx_is_parameter(self):
        """analog_ctx should be a torch.nn.Parameter."""
        model = self.get_layer(in_features=4, out_features=6)
        tile = self._get_analog_tile(model)
        self.assertIsInstance(tile.analog_ctx, Parameter)
        self.assertIsInstance(tile.analog_ctx, AnalogContext)

    def test_ctx_after_set_weights(self):
        """analog_ctx.data should remain valid after set_weights."""
        model = self.get_layer(in_features=4, out_features=6)
        tile = self._get_analog_tile(model)

        # Set new weights
        new_weight = randn(tile.out_size, tile.in_size)
        new_bias = randn(tile.out_size) if tile.analog_bias else None
        tile.set_weights(new_weight, new_bias)

        # ctx should still have a valid non-scalar shape
        self.assertNotEqual(tile.analog_ctx.size(), Size([]))
        self.assertEqual(len(tile.analog_ctx.size()), 2)


@parametrize_over_layers(
    layers=[LinearCuda, LinearMappedCuda],
    tiles=[FloatingPoint, TorchInference],
    biases=["analog", "digital", None],
)
class AnalogCtxDataAttributionCudaTest(ParametrizedTestCase):
    """Tests that analog_ctx.data is correct after moving to CUDA."""

    def _get_analog_tile(self, model):
        """Return the first analog tile from a model."""
        return next(model.analog_tiles())

    def test_ctx_shape_after_cuda(self):
        """analog_ctx.data should retain correct shape after .cuda()."""
        model = self.get_layer(in_features=4, out_features=6)
        tile = self._get_analog_tile(model)

        self.assertTrue(tile.analog_ctx.is_cuda)
        self.assertNotEqual(tile.analog_ctx.size(), Size([]))
        self.assertEqual(len(tile.analog_ctx.size()), 2)

    def test_ctx_device_after_cuda(self):
        """analog_ctx.device should be CUDA after .cuda()."""
        model = self.get_layer(in_features=4, out_features=6)
        tile = self._get_analog_tile(model)

        self.assertEqual(tile.analog_ctx.device.type, "cuda")
        self.assertEqual(tile.device.type, "cuda")


class AnalogCtxBackwardCompatibilityTest(ParametrizedTestCase):
    """Tests for backward compatibility with old checkpoints."""

    use_cuda = False

    def test_old_checkpoint_empty_ctx_loads(self):
        """Checkpoints with empty-size analog_ctx should load without error."""
        model = AnalogLinear(4, 6, bias=True, rpu_config=FloatingPointRPUConfig())

        # Simulate an old checkpoint where analog_ctx was torch.Size([])
        state = model.state_dict()
        for key in list(state.keys()):
            if "analog_ctx" in key:
                state[key] = zeros(())  # Simulate old empty scalar

        # Should load without error (non-strict mode for old ctx)
        model.load_state_dict(state, strict=False, load_rpu_config=False)

    def test_new_checkpoint_loads(self):
        """Checkpoints with properly-shaped analog_ctx should load correctly."""
        model = AnalogLinear(4, 6, bias=True, rpu_config=FloatingPointRPUConfig())
        state = model.state_dict()

        model2 = AnalogLinear(4, 6, bias=True, rpu_config=FloatingPointRPUConfig())
        model2.load_state_dict(state, strict=True, load_rpu_config=False)


class AnalogCtxConversionTest(ParametrizedTestCase):
    """Tests that convert_to_analog produces valid analog_ctx."""

    use_cuda = False

    def test_conversion_ctx_shape(self) -> None:
        """Converted model should have correct analog_ctx shape."""
        digital_model = Sequential(TorchLinear(8, 4), TorchLinear(4, 2))
        analog_model = convert_to_analog(
            digital_model, FloatingPointRPUConfig(), ensure_analog_root=False
        )

        for tile in analog_model.analog_tiles():
            ctx = tile.analog_ctx
            self.assertNotEqual(ctx.size(), Size([]))
            self.assertEqual(len(ctx.size()), 2)

    def test_conversion_conv2d_ctx_shape(self) -> None:
        """Converted Conv2d should have correct analog_ctx shape."""
        digital_conv = TorchConv2d(3, 16, kernel_size=3, padding=1, bias=True)
        analog_conv = AnalogConv2d.from_digital(digital_conv, FloatingPointRPUConfig())

        for tile in analog_conv.analog_tiles():
            ctx = tile.analog_ctx
            self.assertNotEqual(ctx.size(), Size([]))
            self.assertEqual(len(ctx.size()), 2)


class AnalogCtxDevicePropertyTest(ParametrizedTestCase):
    """Tests that tile.device and tile.is_cuda are computed from analog_ctx."""

    use_cuda = False

    def test_device_property_cpu(self):
        """tile.device should return CPU device for CPU tiles."""
        model = AnalogLinear(4, 6, rpu_config=FloatingPointRPUConfig())
        tile = next(model.analog_tiles())

        self.assertEqual(tile.device.type, "cpu")
        self.assertFalse(tile.is_cuda)

    def test_device_property_cuda(self):
        """tile.device should return CUDA device after .cuda()."""
        if SKIP_CUDA_TESTS:
            raise SkipTest("not compiled with CUDA support")

        model = AnalogLinear(4, 6, rpu_config=FloatingPointRPUConfig()).cuda()
        tile = next(model.analog_tiles())

        self.assertEqual(tile.device.type, "cuda")
        self.assertTrue(tile.is_cuda)

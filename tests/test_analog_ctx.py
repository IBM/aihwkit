# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=too-many-locals, no-member
"""Tests for AnalogContext data attribution (PR #717).

Verifies the public AnalogContext data modes: metadata-only placeholder,
read-only logical data view, and independent digital buffer.
"""

from unittest import SkipTest

import torch
from torch import zeros, randn, rand_like, allclose, Tensor, Size, device, manual_seed
from torch.cuda import device_count, device as cuda_device
from torch.nn import Parameter
from torch.nn import Linear as TorchLinear, Sequential, Conv2d as TorchConv2d

from aihwkit.nn import AnalogLinear, AnalogConv2d
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim.context import AnalogContext
from aihwkit.optim.weight_view import ReadOnlyWeightView
from aihwkit.simulator.parameters.enums import AnalogContextDataViewMode
from aihwkit.simulator.configs import (
    FloatingPointRPUConfig,
    InferenceRPUConfig,
    SingleRPUConfig,
    TorchInferenceRPUConfig,
)
from aihwkit.simulator.configs.devices import ConstantStepDevice, SoftBoundsDevice

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
        self.assertEqual(ctx.size(), Size([expected_rows, tile.in_size]))

    def test_ctx_data_values_match_tile_weights(self):
        """data_view mode must reflect the logical tile weights."""
        model = self.get_layer(in_features=4, out_features=6)
        tile = self._get_analog_tile(model)
        tile.analog_ctx.enable_data_view()

        weights_from_tile, _ = tile.get_weights()
        ctx_data = tile.analog_ctx.data.detach().cpu()

        self.assertEqual(ctx_data.shape, weights_from_tile.shape)
        self.assertTrue(allclose(ctx_data, weights_from_tile))

    def test_ctx_norm_is_meaningful(self):
        """analog_ctx.norm() should reflect the weight magnitude, not 1.0."""
        manual_seed(42)
        model = self.get_layer(in_features=4, out_features=6)
        tile = self._get_analog_tile(model)

        # With randomly initialized weights, the norm should be > 0
        # and should NOT be exactly 1.0 (which the old scalar ones(()) returned).
        tile.analog_ctx.enable_data_view()
        norm_val = tile.analog_ctx.norm().item()
        self.assertGreater(norm_val, 0.0)

    def test_ctx_nonzero_works(self):
        """analog_ctx.nonzero() should return indices of nonzero weights."""
        model = self.get_layer(in_features=4, out_features=6)
        tile = self._get_analog_tile(model)

        # With random initialization, most weights are nonzero.
        tile.analog_ctx.enable_data_view()
        nz = tile.analog_ctx.nonzero()
        self.assertGreater(len(nz), 0)

    def test_ctx_comparison_ops(self):
        """Comparison operators on analog_ctx should work on actual weights."""
        model = self.get_layer(in_features=4, out_features=6)
        tile = self._get_analog_tile(model)

        # Weights are initialized near zero with std ~1, so most are < 10.
        tile.analog_ctx.enable_data_view()
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

        # ctx should still have a valid logical non-scalar shape and value.
        self.assertNotEqual(tile.analog_ctx.size(), Size([]))
        self.assertEqual(tile.analog_ctx.size(), Size([tile.out_size, tile.in_size]))
        tile.analog_ctx.enable_data_view()
        self.assertTrue(allclose(tile.analog_ctx.data.detach().cpu(), tile.get_weights()[0]))


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
        self.assertEqual(tile.analog_ctx.size(), Size([tile.out_size, tile.in_size]))

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

    def test_missing_analog_state_non_strict_skips_ctx_copy(self):
        """Non-strict load should not copy into the read-only analog context."""
        model = AnalogLinear(4, 6, bias=True, rpu_config=FloatingPointRPUConfig())
        state = model.state_dict()
        for key in list(state.keys()):
            if key.endswith("analog_tile_state"):
                del state[key]

        model.load_state_dict(state, strict=False, load_rpu_config=False)


class AnalogCtxStateDictLoadTest(ParametrizedTestCase):
    """Tests for analog state restoration with read-only contexts."""

    use_cuda = False

    def test_inference_tile_load_restores_raw_weights(self):
        """Loading analog_tile_state should update shared raw tile backing."""
        source = AnalogLinear(4, 2, bias=False, rpu_config=InferenceRPUConfig())
        target = AnalogLinear(4, 2, bias=False, rpu_config=InferenceRPUConfig())

        source.set_weights(randn(2, 4), None)
        saved_raw = next(source.analog_tiles()).tile.get_weights().clone()

        target.load_state_dict(source.state_dict(), load_rpu_config=False)
        loaded_raw = next(target.analog_tiles()).tile.get_weights()

        self.assertTrue(allclose(loaded_raw, saved_raw))

    def test_square_tile_load_does_not_transpose_cpu_weights(self):
        """Square CPU tile state should not be mistaken for CUDA layout."""
        source = AnalogLinear(3, 3, bias=False, rpu_config=FloatingPointRPUConfig())
        target = AnalogLinear(3, 3, bias=False, rpu_config=FloatingPointRPUConfig())

        source.set_weights(randn(3, 3), None)
        saved_raw = next(source.analog_tiles()).tile.get_weights().clone()

        target.load_state_dict(source.state_dict(), load_rpu_config=False)
        loaded_raw = next(target.analog_tiles()).tile.get_weights()

        self.assertTrue(allclose(loaded_raw, saved_raw))


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

    def test_ctx_device_properties_do_not_read_logical_weights(self):
        """Device metadata should use raw backing without calling get_scales()."""
        model = AnalogLinear(4, 6, rpu_config=FloatingPointRPUConfig())
        tile = next(model.analog_tiles())

        def fail_get_scales():
            raise AssertionError("get_scales should not be called for device metadata")

        tile.get_scales = fail_get_scales
        self.assertEqual(tile.analog_ctx.device.type, "cpu")
        self.assertFalse(tile.analog_ctx.is_cuda)
        self.assertEqual(tile.analog_ctx.dtype, tile.analog_ctx._raw_data().dtype)

    def test_ctx_is_leaf_uses_raw_backing_with_learned_out_scaling(self):
        """Optimizer construction should not see ctx as a logical non-leaf tensor."""
        from aihwkit.optim import AnalogSGD

        rpu_config = InferenceRPUConfig()
        rpu_config.mapping.learn_out_scaling = True
        model = AnalogLinear(4, 6, bias=False, rpu_config=rpu_config)
        ctx = next(model.analog_tiles()).analog_ctx

        self.assertTrue(ctx.is_leaf)
        AnalogSGD(model.parameters(), lr=0.1)


class AnalogCtxSyncAfterSetWeightsTest(ParametrizedTestCase):
    """Reviewer concern #1: analog_ctx.data must stay in sync after set_weights."""

    use_cuda = False

    def _test_sync(self, rpu_config, use_cuda):
        """Helper: verify ctx.data matches tile weights after set_weights."""
        model = AnalogLinear(4, 6, bias=False, rpu_config=rpu_config)
        if use_cuda:
            model = model.cuda()
        tile = next(model.analog_tiles())
        tile.analog_ctx.enable_data_view()

        new_w = randn(6, 4)
        tile.set_weights(new_w, None)

        w_from_tile, _ = tile.get_weights()
        ctx_data = tile.analog_ctx.data.detach().cpu()
        self.assertTrue(allclose(ctx_data, w_from_tile),
                        "analog_ctx.data out of sync after set_weights")

    def test_sync_torch_inference_cpu(self):
        """TorchInference CPU: ctx stays in sync after set_weights."""
        self._test_sync(TorchInferenceRPUConfig(), use_cuda=False)

    def test_sync_floating_point_cpu(self):
        """FloatingPoint CPU: ctx stays in sync after set_weights."""
        self._test_sync(FloatingPointRPUConfig(), use_cuda=False)

    def test_sync_torch_inference_cuda(self):
        """TorchInference CUDA: ctx stays in sync after set_weights."""
        if SKIP_CUDA_TESTS:
            raise SkipTest("not compiled with CUDA support")
        self._test_sync(TorchInferenceRPUConfig(), use_cuda=True)

    def test_sync_floating_point_cuda(self):
        """FloatingPoint CUDA: ctx stays in sync after set_weights."""
        if SKIP_CUDA_TESTS:
            raise SkipTest("not compiled with CUDA support")
        self._test_sync(FloatingPointRPUConfig(), use_cuda=True)

    def test_sync_after_multiple_set_weights(self):
        """ctx should stay in sync after multiple consecutive set_weights."""
        model = AnalogLinear(4, 6, bias=False, rpu_config=TorchInferenceRPUConfig())
        tile = next(model.analog_tiles())
        tile.analog_ctx.enable_data_view()

        for _ in range(5):
            new_w = randn(6, 4)
            tile.set_weights(new_w, None)
            w_from_tile, _ = tile.get_weights()
            ctx_data = tile.analog_ctx.data.detach().cpu()
            self.assertTrue(allclose(ctx_data, w_from_tile))

    def test_sync_after_cuda_move(self):
        """ctx should sync after CPU->CUDA->CPU round-trip."""
        if SKIP_CUDA_TESTS:
            raise SkipTest("not compiled with CUDA support")

        model = AnalogLinear(4, 6, bias=False, rpu_config=TorchInferenceRPUConfig())
        tile = next(model.analog_tiles())

        new_w = randn(6, 4)
        tile.set_weights(new_w, None)

        # Move to CUDA
        model.cuda()
        tile = next(model.analog_tiles())
        tile.analog_ctx.enable_data_view()
        w_cuda, _ = tile.get_weights()
        ctx_cuda = tile.analog_ctx.data.detach().cpu()
        self.assertTrue(allclose(ctx_cuda, w_cuda),
                        "ctx out of sync after cuda()")


class AnalogCtxGetWeightsConventionTest(ParametrizedTestCase):
    """Reviewer concern #3: get_weights default returns detached CPU copy."""

    use_cuda = False

    def test_get_weights_returns_cpu_torch_inference(self):
        """TorchInference: get_weights() returns CPU tensor by default."""
        if SKIP_CUDA_TESTS:
            raise SkipTest("not compiled with CUDA support")
        model = AnalogLinear(
            4, 6, bias=False, rpu_config=TorchInferenceRPUConfig()
        ).cuda()
        tile = next(model.analog_tiles())
        w, _ = tile.get_weights()
        self.assertEqual(w.device.type, "cpu",
                         "get_weights() should return CPU tensor by default")

    def test_get_weights_returns_cpu_floating_point(self):
        """FloatingPoint: get_weights() returns CPU tensor by default."""
        if SKIP_CUDA_TESTS:
            raise SkipTest("not compiled with CUDA support")
        model = AnalogLinear(
            4, 6, bias=False, rpu_config=FloatingPointRPUConfig()
        ).cuda()
        tile = next(model.analog_tiles())
        w, _ = tile.get_weights()
        self.assertEqual(w.device.type, "cpu",
                         "get_weights() should return CPU tensor by default")

    def test_get_weights_is_detached(self):
        """get_weights() result should not have grad_fn (detached)."""
        model = AnalogLinear(
            4, 6, bias=False, rpu_config=TorchInferenceRPUConfig()
        )
        tile = next(model.analog_tiles())
        w, _ = tile.get_weights()
        self.assertFalse(w.requires_grad,
                         "get_weights() should return detached tensor")
        self.assertIsNone(w.grad_fn,
                          "get_weights() should return detached tensor")

    def test_get_weights_is_copy(self):
        """Modifying get_weights() result should NOT change tile weights."""
        model = AnalogLinear(
            4, 6, bias=False, rpu_config=TorchInferenceRPUConfig()
        )
        tile = next(model.analog_tiles())
        w_before, _ = tile.get_weights()
        w_copy, _ = tile.get_weights()
        w_copy.fill_(999.0)
        w_after, _ = tile.get_weights()
        self.assertTrue(allclose(w_before, w_after),
                        "get_weights() should return a copy, not a reference")


class AnalogCtxTileGetWeightsRefTest(ParametrizedTestCase):
    """Test tile.get_weights(as_ref=...) reference vs copy semantics."""

    use_cuda = False

    def _get_tile(self, rpu_config):
        model = AnalogLinear(4, 6, bias=False, rpu_config=rpu_config)
        return next(model.analog_tiles())

    def test_as_ref_true_shares_storage(self):
        """as_ref=True should return tensors sharing the same storage."""
        tile = self._get_tile(TorchInferenceRPUConfig())
        ref1 = tile.tile.get_weights(as_ref=True)
        ref2 = tile.tile.get_weights(as_ref=True)
        self.assertEqual(ref1.data_ptr(), ref2.data_ptr(),
                         "as_ref=True should return the same data pointer")

    def test_clone_does_not_share_storage(self):
        """clone() of as_ref=True should NOT share storage."""
        tile = self._get_tile(TorchInferenceRPUConfig())
        ref = tile.tile.get_weights(as_ref=True)
        clone = ref.clone()
        self.assertNotEqual(ref.data_ptr(), clone.data_ptr(),
                            "clone should have a different data pointer")

    def test_as_ref_true_modification_propagates(self):
        """Modifying as_ref=True tensor should change tile weights."""
        tile = self._get_tile(TorchInferenceRPUConfig())
        ref = tile.tile.get_weights(as_ref=True)
        original = ref[0, 0].item()
        ref[0, 0] += 999.0
        check = tile.tile.get_weights(as_ref=True)
        self.assertAlmostEqual(check[0, 0].item(), original + 999.0, places=2,
                               msg="as_ref=True modification should propagate to tile")

    def test_as_ref_false_modification_does_not_propagate(self):
        """Modifying as_ref=False tensor should NOT change tile weights."""
        tile = self._get_tile(TorchInferenceRPUConfig())
        original = tile.tile.get_weights(as_ref=True)[0, 0].item()
        copy = tile.tile.get_weights(as_ref=False)
        copy[0, 0] += 999.0
        check = tile.tile.get_weights(as_ref=True)
        self.assertAlmostEqual(check[0, 0].item(), original, places=5,
                               msg="as_ref=False modification should NOT propagate to tile")

    def test_clone_modification_does_not_propagate(self):
        """Modifying clone of as_ref=True should NOT change tile weights."""
        tile = self._get_tile(TorchInferenceRPUConfig())
        original = tile.tile.get_weights(as_ref=True)[0, 0].item()
        clone = tile.tile.get_weights(as_ref=True).clone()
        clone[0, 0] += 999.0
        check = tile.tile.get_weights(as_ref=True)
        self.assertAlmostEqual(check[0, 0].item(), original, places=5,
                               msg="clone modification should NOT propagate to tile")


class AnalogCtxSharedWeightsZeroCopyTest(ParametrizedTestCase):
    """Tests that C++ tiles use zero-copy shared weights with analog_ctx.

    After ``_bind_shared_weights``, the C++ tile's internal weight storage
    and the private ``analog_ctx._raw_data()`` share the same memory.
    ``tile.update()`` and ``tile.set_weights()`` must be visible through the shared tensor
    without any explicit sync call.
    """

    use_cuda = False

    def _get_tile(self, rpu_config):
        model = AnalogLinear(4, 6, bias=False, rpu_config=rpu_config)
        return next(model.analog_tiles())

    # -- FloatingPoint (C++ tile) tests ----------------------------------------

    def test_shared_tensor_exists_for_cpp_tile(self):
        """C++ tiles should have a non-None _shared_weight_tensor."""
        tile = self._get_tile(FloatingPointRPUConfig())
        self.assertIsNotNone(tile._shared_weight_tensor,
                             "C++ tile should have a shared weight tensor")

    def test_shared_tensor_none_for_python_tile(self):
        """Pure Python tiles should NOT use _shared_weight_tensor."""
        tile = self._get_tile(TorchInferenceRPUConfig())
        self.assertIsNone(tile._shared_weight_tensor,
                          "Python tile should not use shared weight tensor")

    def test_raw_context_shares_memory_with_cpp_tile(self):
        """analog_ctx._raw_data() and _shared_weight_tensor should share memory."""
        tile = self._get_tile(FloatingPointRPUConfig())
        ctx_ptr = tile.analog_ctx._raw_data().data_ptr()
        shared_ptr = tile._shared_weight_tensor.data_ptr()
        self.assertEqual(ctx_ptr, shared_ptr,
                         "analog_ctx raw data and shared tensor should have same data_ptr")

    def test_update_reflects_in_ctx_without_sync(self):
        """After tile.update(), analog_ctx.data should reflect changes (zero-copy)."""
        tile = self._get_tile(FloatingPointRPUConfig())
        tile.tile.set_learning_rate(0.1)

        snapshot = tile.analog_ctx._raw_data().detach().clone()

        x = randn(1, 4)
        d = randn(1, 6)
        tile.tile.update(x, d, False)

        # raw context backing should have changed without explicit sync
        self.assertFalse(
            allclose(tile.analog_ctx._raw_data().detach(), snapshot),
            "analog_ctx raw data should change after tile.update() without sync")

        # And it should match get_weights()
        tile.analog_ctx.enable_data_view()
        w_from_tile = tile.tile.get_weights()
        self.assertTrue(
            allclose(tile.analog_ctx.data.detach(), w_from_tile),
            "analog_ctx.data should match logical get_weights() after update")

    def test_set_weights_reflects_in_ctx_without_sync(self):
        """After tile.set_weights(), analog_ctx.data should reflect changes."""
        tile = self._get_tile(FloatingPointRPUConfig())
        new_w = randn(6, 4)
        tile.tile.set_weights(new_w)
        tile.analog_ctx.enable_data_view()

        self.assertTrue(
            allclose(tile.analog_ctx.data.detach(), new_w),
            "analog_ctx.data should match new weights after set_weights()")

    def test_multiple_updates_stay_in_sync(self):
        """analog_ctx.data should stay in sync across 10 consecutive updates."""
        tile = self._get_tile(FloatingPointRPUConfig())
        tile.tile.set_learning_rate(0.01)

        for _ in range(10):
            x = randn(4, 4)
            d = randn(4, 6)
            tile.tile.update(x, d, False)

            tile.analog_ctx.enable_data_view()
            w_from_tile = tile.tile.get_weights()
            self.assertTrue(
                allclose(tile.analog_ctx.data.detach(), w_from_tile),
                "analog_ctx.data drifted from logical tile weights during updates")

    def test_get_weights_ref_returns_shared_tensor(self):
        """_get_tile_weights_ref should return the shared tensor for C++ tiles."""
        tile = self._get_tile(FloatingPointRPUConfig())
        ref = tile._get_tile_weights_ref()
        self.assertEqual(ref.data_ptr(), tile._shared_weight_tensor.data_ptr(),
                         "_get_tile_weights_ref should return shared tensor")

    # -- as_ref=True after update (the key new test) ---------------------------

    def test_as_ref_true_reflects_update_python_tile(self):
        """Python tile: as_ref=True weight should reflect tile.update() changes."""
        tile = self._get_tile(TorchInferenceRPUConfig())
        ref = tile.tile.get_weights(as_ref=True)
        snapshot = ref.clone()

        # Manually modify via the ref (simulating what update would do)
        ref[0, 0] += 100.0
        check = tile.tile.get_weights(as_ref=True)
        self.assertAlmostEqual(check[0, 0].item(), snapshot[0, 0].item() + 100.0, places=2,
                               msg="Python tile: as_ref write should propagate")

    def test_as_ref_true_reflects_update_cpp_tile(self):
        """C++ tile: _get_tile_weights_ref should reflect tile.update() changes.

        This is the key test: for C++ tiles, the shared weight tensor
        (returned by _get_tile_weights_ref) must automatically reflect
        updates performed by the C++ tile.update() — zero-copy.
        """
        tile = self._get_tile(FloatingPointRPUConfig())
        tile.tile.set_learning_rate(0.1)

        ref = tile._get_tile_weights_ref()
        snapshot = ref.clone()

        x = randn(1, 4)
        d = randn(1, 6)
        tile.tile.update(x, d, False)

        # ref should have been modified in-place by the C++ update
        self.assertFalse(
            allclose(ref, snapshot),
            "C++ tile: shared weight ref should change after update (zero-copy)")

        # And the ref should match get_weights
        w_copy = tile.tile.get_weights()
        self.assertTrue(
            allclose(ref, w_copy),
            "C++ tile: shared weight ref should match get_weights() after update")


class AnalogCtxDataViewModeTest(ParametrizedTestCase):
    """Tests for AnalogContext public data view modes."""

    use_cuda = False

    def _make_model(self, bias=False, analog_bias=False):
        """Create an AnalogLinear model for context tests."""
        rpu_config = FloatingPointRPUConfig() if analog_bias else TorchInferenceRPUConfig()
        if analog_bias:
            rpu_config.mapping.digital_bias = False
        return AnalogLinear(4, 6, bias=bias, rpu_config=rpu_config)

    def _get_tile_and_ctx(self, model):
        tile = next(model.analog_tiles())
        return tile, tile.analog_ctx

    # -- placeholder mode -------------------------------------------------------

    def test_ctx_has_no_readonly_flag(self):
        """The public readonly switch was removed."""
        _, ctx = self._get_tile_and_ctx(self._make_model())
        self.assertFalse(hasattr(ctx, "readonly"))

    def test_default_mode_is_placeholder(self):
        """AnalogContext should default to metadata-only placeholder mode."""
        _, ctx = self._get_tile_and_ctx(self._make_model())

        self.assertEqual(ctx.data_view_mode, AnalogContextDataViewMode.PLACEHOLDER)
        self.assertNotIsInstance(ctx.data, ReadOnlyWeightView)

    def test_placeholder_allows_metadata_only(self):
        """Placeholder mode should expose shape-like metadata only."""
        tile, ctx = self._get_tile_and_ctx(self._make_model())
        expected_shape = Size([tile.out_size, tile.in_size])

        self.assertEqual(ctx.size(), expected_shape)
        self.assertEqual(ctx.data.size(), expected_shape)
        self.assertEqual(ctx.shape, expected_shape)
        self.assertEqual(ctx.data.shape, expected_shape)
        self.assertEqual(ctx.ndim, 2)
        self.assertEqual(ctx.data.ndim, 2)
        self.assertEqual(ctx.device, ctx._raw_data().device)
        self.assertEqual(ctx.dtype, ctx._raw_data().dtype)

    def test_placeholder_blocks_value_reads(self):
        """Placeholder mode should block operations that read weight values."""
        _, ctx = self._get_tile_and_ctx(self._make_model())

        for op in [
            lambda: ctx.norm(),
            lambda: ctx.data.norm(),
            lambda: ctx.nonzero(),
            lambda: ctx.data.nonzero(),
            lambda: ctx > 10,
            lambda: ctx.data > 10,
            lambda: ctx.get_data(),
            lambda: ctx.detach().norm(),
        ]:
            with self.subTest(op=op):
                with self.assertRaises(RuntimeError):
                    op()

    def test_placeholder_state_dict_round_trip(self):
        """Saving and loading should not require public weight reads."""
        source = self._make_model()
        target = self._make_model()

        self.assertEqual(
            next(source.analog_tiles()).analog_ctx.data_view_mode,
            AnalogContextDataViewMode.PLACEHOLDER,
        )
        target.load_state_dict(source.state_dict(), load_rpu_config=False)

    # -- data_view mode ---------------------------------------------------------

    def test_enable_data_view_uses_logical_data(self):
        """size, norm, detach, nonzero, comparisons all use logical weights."""
        tile, ctx = self._get_tile_and_ctx(self._make_model())
        ctx.enable_data_view()
        tile.set_scales(2.0)
        logical_weight, _ = tile.get_weights()

        self.assertEqual(ctx.data_view_mode, AnalogContextDataViewMode.DATA_VIEW)
        self.assertIsInstance(ctx.data, ReadOnlyWeightView)
        self.assertEqual(ctx.size(), Size([6, 4]))
        self.assertEqual(ctx.shape, Size([6, 4]))
        self.assertTrue(allclose(ctx.detach().cpu(), logical_weight))
        self.assertAlmostEqual(ctx.norm().item(), logical_weight.norm().item(), places=5)
        self.assertGreater(len(ctx.nonzero()), 0)
        mask = ctx > 10
        self.assertEqual(mask.shape, ctx.shape)

    def test_scaled_context_returns_logical_weight_in_data_view(self):
        """Scaled tiles expose scaled logical weights, not raw tile weights."""
        tile, ctx = self._get_tile_and_ctx(self._make_model())
        ctx.enable_data_view()
        tile.set_scales(2.0)

        logical_weight = tile.get_weights()[0]
        raw_weight = tile.get_weights(apply_weight_scaling=False)[0]

        self.assertIsInstance(ctx.data, ReadOnlyWeightView)
        self.assertTrue(allclose(ctx.data.detach().cpu(), logical_weight))
        self.assertFalse(allclose(ctx.data.detach().cpu(), raw_weight))

    def test_setting_scale_changes_ctx_data_in_data_view(self):
        """Manually setting scales should update logical ctx.data values."""
        tile, ctx = self._get_tile_and_ctx(self._make_model())
        ctx.enable_data_view()
        before = ctx.data.detach().clone()

        tile.set_scales(2.0)

        after = ctx.data.detach()
        w_from_tile, _ = tile.get_weights()
        self.assertFalse(allclose(after, before))
        self.assertTrue(allclose(after, before * 2.0))
        self.assertTrue(allclose(after.cpu(), w_from_tile))

    def test_analog_bias_column_is_not_exposed(self):
        """Public views should not expose the raw analog-bias column."""
        tile, ctx = self._get_tile_and_ctx(self._make_model(bias=True, analog_bias=True))
        self.assertTrue(tile.analog_bias)
        self.assertEqual(ctx.data.shape, Size([tile.out_size, tile.in_size]))
        self.assertEqual(
            tile.analog_ctx._raw_data().shape,
            Size([tile.out_size, tile.in_size + 1]),
        )

        new_weight = randn(tile.out_size, tile.in_size)
        new_bias = randn(tile.out_size)
        tile.set_weights(new_weight, new_bias)
        ctx.enable_data_view()
        self.assertTrue(allclose(ctx.data.detach().cpu(), tile.get_weights()[0]))

    # -- data_view direct writes stay blocked ----------------------------------

    def test_data_view_data_inplace_ops_blocked(self):
        """ctx.data in-place ops should raise RuntimeError in data_view mode."""
        _, ctx = self._get_tile_and_ctx(self._make_model())
        ctx.enable_data_view()
        for op in [
            lambda: ctx.data.add_(1.0),
            lambda: ctx.data.mul_(2.0),
            lambda: ctx.data.copy_(randn(6, 4)),
            lambda: ctx.data.fill_(0.0),
            lambda: ctx.data.zero_(),
        ]:
            with self.subTest(op=op):
                with self.assertRaises(RuntimeError):
                    op()

    def test_data_view_context_inplace_ops_blocked(self):
        """Direct in-place ops on ctx should raise RuntimeError."""
        _, ctx = self._get_tile_and_ctx(self._make_model())
        ctx.enable_data_view()
        ctx.requires_grad = False
        snapshot = ctx.data.detach().clone()

        with self.assertRaises(RuntimeError):
            ctx.add_(1.0)
        with self.assertRaises(RuntimeError):
            ctx.copy_(snapshot)

        self.assertTrue(allclose(ctx.data.detach(), snapshot))

    def _assert_out_writes_blocked(self, ctx):
        """Assert that torch out= cannot target ctx or ctx.data."""
        ctx.enable_data_view()
        ctx.requires_grad = False
        raw_snapshot = ctx._raw_data().detach().clone()
        logical_snapshot = ctx.data.detach().clone()
        lhs = torch.ones(ctx.shape, dtype=ctx.dtype, device=ctx.device)
        rhs = torch.ones(ctx.shape, dtype=ctx.dtype, device=ctx.device)

        for out in [ctx, ctx.data]:
            with self.subTest(out_type=type(out).__name__):
                with self.assertRaises(RuntimeError):
                    torch.add(lhs, rhs, out=out)
                self.assertTrue(allclose(ctx._raw_data().detach(), raw_snapshot))
                self.assertTrue(allclose(ctx.data.detach(), logical_snapshot))

    def test_out_kwarg_writes_blocked_without_scaling(self):
        """out= writes must not bypass read-only weights without scaling."""
        rpu_config = SingleRPUConfig(device=ConstantStepDevice())
        tile, ctx = self._get_tile_and_ctx(
            AnalogLinear(4, 6, bias=False, rpu_config=rpu_config)
        )

        self.assertIsNone(tile.get_scales())
        self._assert_out_writes_blocked(ctx)

    def test_out_kwarg_writes_blocked_with_scaling(self):
        """out= writes must fail instead of silently writing to a temporary."""
        tile, ctx = self._get_tile_and_ctx(self._make_model())
        tile.set_scales(2.0)

        self.assertIsNotNone(tile.get_scales())
        self._assert_out_writes_blocked(ctx)

    def test_copy_from_context_to_external_tensor_allowed(self):
        """copy_ may read from ctx or ctx.data in data_view mode."""
        _, ctx = self._get_tile_and_ctx(self._make_model())
        ctx.enable_data_view()
        expected = ctx.data.detach().clone()

        for source in [ctx, ctx.data]:
            with self.subTest(source_type=type(source).__name__):
                destination = torch.empty_like(expected)
                destination.copy_(source)
                self.assertTrue(allclose(destination, expected))

    def test_setitem_blocked_in_data_view(self):
        """Item assignment through ctx or ctx.data should raise RuntimeError."""
        _, ctx = self._get_tile_and_ctx(self._make_model())
        ctx.enable_data_view()
        snapshot = ctx.data.detach().clone()

        with self.assertRaises(RuntimeError):
            ctx.data[0, 0] = 999.0
        with self.assertRaises(RuntimeError):
            ctx[0, 0] = 999.0

        self.assertTrue(allclose(ctx.data.detach(), snapshot))

    def test_data_assignment_blocked(self):
        """ctx.data rebinding should raise RuntimeError in all modes."""
        _, ctx = self._get_tile_and_ctx(self._make_model())
        ctx.enable_data_view()
        snapshot = ctx.data.detach().clone()

        with self.assertRaisesRegex(RuntimeError, "Direct replacement"):
            ctx.data = rand_like(ctx.data)

        self.assertIsInstance(ctx.data, ReadOnlyWeightView)
        self.assertTrue(allclose(ctx.data.detach(), snapshot))

    # -- buffer mode -----------------------------------------------------------

    def test_buffer_mode_creates_zero_logical_tensor(self):
        """buffer mode should create a zero tensor with logical weight shape."""
        tile, ctx = self._get_tile_and_ctx(self._make_model())
        ctx.enable_buffer()

        self.assertEqual(ctx.data_view_mode, AnalogContextDataViewMode.BUFFER)
        self.assertNotIsInstance(ctx.data, ReadOnlyWeightView)
        self.assertEqual(ctx.data.shape, Size([tile.out_size, tile.in_size]))
        self.assertTrue(allclose(ctx.data, torch.zeros_like(ctx.data)))

    def test_buffer_inplace_ops_mutate_only_buffer(self):
        """buffer writes should not modify analog tile weights."""
        tile, ctx = self._get_tile_and_ctx(self._make_model())
        weight_snapshot = tile.get_weights()[0].clone()
        ctx.enable_buffer()

        ctx.data.add_(2.0)
        self.assertTrue(allclose(ctx.data, torch.full_like(ctx.data, 2.0)))
        self.assertTrue(allclose(tile.get_weights()[0], weight_snapshot))

    def test_buffer_is_independent_after_mode_switches(self):
        """Switching modes should not link the buffer to tile weights."""
        tile, ctx = self._get_tile_and_ctx(self._make_model())
        weight_snapshot = tile.get_weights()[0].clone()

        ctx.enable_buffer()
        ctx.data.fill_(3.0)
        self.assertTrue(allclose(ctx.data, torch.full_like(ctx.data, 3.0)))

        ctx.enable_data_view()
        self.assertTrue(allclose(ctx.data.detach().cpu(), weight_snapshot))

        ctx.enable_buffer()
        self.assertTrue(allclose(ctx.data, torch.zeros_like(ctx.data)))
        self.assertTrue(allclose(tile.get_weights()[0], weight_snapshot))

    def test_buffer_analog_bias_uses_logical_shape(self):
        """buffer mode should not include the raw analog-bias column."""
        tile, ctx = self._get_tile_and_ctx(self._make_model(bias=True, analog_bias=True))
        ctx.enable_buffer()

        self.assertEqual(ctx.data.shape, Size([tile.out_size, tile.in_size]))
        self.assertEqual(
            tile.analog_ctx._raw_data().shape,
            Size([tile.out_size, tile.in_size + 1]),
        )

    # -- removed APIs ----------------------------------------------------------

    def test_public_write_escape_apis_removed(self):
        """Old direct-write escape hatches should not exist."""
        _, ctx = self._get_tile_and_ctx(self._make_model())
        self.assertFalse(hasattr(ctx, "set_data"))
        self.assertFalse(hasattr(ctx, "writable"))
        self.assertFalse(hasattr(ctx, "readonly"))
        self.assertFalse(hasattr(TorchInferenceRPUConfig().mapping, "readonly_weights"))

    def test_convert_to_analog_readonly_argument_removed(self):
        """convert_to_analog(readonly=...) should no longer be accepted."""
        digital_model = Sequential(TorchLinear(8, 4), TorchLinear(4, 2))
        with self.assertRaises(TypeError):
            convert_to_analog(  # pylint: disable=unexpected-keyword-arg
                digital_model,
                TorchInferenceRPUConfig(),
                ensure_analog_root=False,
                readonly=False,
            )


class SharedWeightsCudaBindingTest(ParametrizedTestCase):
    """Tests that shared weight binding works correctly after .cuda().

    ``_bind_shared_weights()`` provides zero-copy weight access, but originally
    only worked on CPU.  After ``.cuda()``, the binding was lost because
    ``RPUCudaSimulatorTileWrapper.cuda()`` never re-bound it.

    These tests cover:

    - ``_bind_shared_weights()`` must allocate on the correct device
      with the correct layout (CUDA uses transposed ``(x_size, d_size)``).
    - ``.cuda()`` / ``.cpu()`` must re-bind shared weights for
      FloatingPoint-family tiles.
    - ``CudaAnalogTile.set_shared_weights()`` breaks ``is_perfect`` forward,
      so AnalogTile variants must NOT be bound on CUDA.
    - Training must still converge after shared weight binding.
    """

    use_cuda = False  # We manually skip inside each test

    def _skip_if_no_cuda(self):
        if SKIP_CUDA_TESTS:
            raise SkipTest("not compiled with CUDA support")

    # -- FloatingPoint: shared binding survives .cuda() -----------------------

    def test_shared_tensor_exists_after_cuda_floating_point(self):
        """FloatingPoint tile should have shared tensor after .cuda()."""
        self._skip_if_no_cuda()
        model = AnalogLinear(8, 4, bias=False,
                             rpu_config=FloatingPointRPUConfig()).cuda()
        tile = next(model.analog_tiles())
        self.assertIsNotNone(tile._shared_weight_tensor,
                             "shared tensor should exist after .cuda()")
        self.assertTrue(tile._shared_weight_tensor.is_cuda,
                        "shared tensor should be on CUDA")

    def test_shared_tensor_device_matches_tile(self):
        """Shared tensor device should match the CUDA tile device."""
        self._skip_if_no_cuda()
        model = AnalogLinear(8, 4, bias=False,
                             rpu_config=FloatingPointRPUConfig()).cuda()
        tile = next(model.analog_tiles())
        self.assertEqual(tile._shared_weight_tensor.device.type, "cuda")

    def test_shared_tensor_uses_nonzero_cuda_device(self):
        """Rebinding should use the tile device, not the current CUDA context."""
        self._skip_if_no_cuda()
        if device_count() < 2:
            raise SkipTest("Need at least two devices for this test")

        target_device = device("cuda", 1)
        model = AnalogLinear(8, 4, bias=False,
                             rpu_config=FloatingPointRPUConfig()).cuda(target_device)
        tile = next(model.analog_tiles())
        expected_weights = tile.tile.get_weights().clone()

        tile._shared_weight_tensor = None
        with cuda_device(0):
            tile._bind_shared_weights()

        self.assertEqual(tile.device, target_device)
        self.assertEqual(tile._shared_weight_tensor.device, target_device)
        self.assertTrue(allclose(tile._shared_weight_tensor.t().cpu(), expected_weights))

    def test_shared_tensor_has_correct_values_after_cuda(self):
        """Shared tensor should contain actual weights, not zeros."""
        self._skip_if_no_cuda()
        model = AnalogLinear(8, 4, bias=False,
                             rpu_config=FloatingPointRPUConfig())
        tile_cpu = next(model.analog_tiles())
        w_cpu = tile_cpu.tile.get_weights().clone()

        model = model.cuda()
        tile = next(model.analog_tiles())

        # Shared tensor values should match the original CPU weights.
        # The shared tensor is transposed on CUDA, so compare sorted values.
        shared_vals = tile._shared_weight_tensor.cpu().flatten().sort()[0]
        cpu_vals = w_cpu.flatten().sort()[0]
        self.assertTrue(allclose(shared_vals, cpu_vals, atol=1e-5),
                        "shared tensor should contain the original weights")

    def test_update_reflects_in_shared_tensor_cuda(self):
        """tile.update() should modify the shared tensor on CUDA (zero-copy)."""
        self._skip_if_no_cuda()
        model = AnalogLinear(4, 6, bias=False,
                             rpu_config=FloatingPointRPUConfig()).cuda()
        tile = next(model.analog_tiles())
        tile.tile.set_learning_rate(0.1)

        snapshot = tile._shared_weight_tensor.clone()
        x = randn(2, 4, device="cuda")
        d = randn(2, 6, device="cuda")
        tile.tile.update(x, d, False)

        self.assertFalse(allclose(tile._shared_weight_tensor, snapshot),
                         "shared tensor should change after tile.update()")

    def test_set_weights_reflects_in_shared_tensor_cuda(self):
        """tile.set_weights() should sync to the shared tensor on CUDA."""
        self._skip_if_no_cuda()
        model = AnalogLinear(4, 6, bias=False,
                             rpu_config=FloatingPointRPUConfig()).cuda()
        tile = next(model.analog_tiles())

        new_w = randn(6, 4)
        tile.tile.set_weights(new_w)

        # Compare values (shared is transposed on CUDA)
        shared_vals = tile._shared_weight_tensor.cpu().flatten().sort()[0]
        expected_vals = new_w.flatten().sort()[0]
        self.assertTrue(allclose(shared_vals, expected_vals, atol=1e-5),
                        "shared tensor should reflect set_weights()")

    # -- FloatingPoint: training convergence after binding --------------------

    def test_training_converges_floating_point_cuda(self):
        """Training with FloatingPoint on CUDA should converge after binding."""
        self._skip_if_no_cuda()
        from torch.nn.functional import mse_loss
        from aihwkit.optim import AnalogSGD

        manual_seed(42)
        model = AnalogLinear(4, 2, bias=False,
                             rpu_config=FloatingPointRPUConfig()).cuda()
        x = randn(10, 4, device="cuda")
        y = randn(10, 2, device="cuda")

        initial_loss = mse_loss(model(x), y).item()

        opt = AnalogSGD(model.parameters(), lr=0.1)
        opt.regroup_param_groups(model)
        for _ in range(50):
            opt.zero_grad()
            loss = mse_loss(model(x), y)
            loss.backward()
            opt.step()

        final_loss = mse_loss(model(x), y).item()
        self.assertLess(final_loss, initial_loss,
                        "training should reduce loss on CUDA FloatingPoint")

    # -- CPU round-trip: .cuda() then .cpu() ----------------------------------

    def test_shared_tensor_survives_cpu_round_trip(self):
        """Shared tensor should be re-bound after .cuda() -> .cpu()."""
        self._skip_if_no_cuda()
        model = AnalogLinear(8, 4, bias=False,
                             rpu_config=FloatingPointRPUConfig())
        model = model.cuda()
        model = model.cpu()
        tile = next(model.analog_tiles())

        self.assertIsNotNone(tile._shared_weight_tensor,
                             "shared tensor should exist after round-trip")
        self.assertFalse(tile._shared_weight_tensor.is_cuda,
                         "shared tensor should be on CPU after .cpu()")

    def test_shared_tensor_survives_cuda_cpu_cuda(self):
        """Shared tensor should survive CPU -> CUDA -> CPU -> CUDA."""
        self._skip_if_no_cuda()
        model = AnalogLinear(8, 4, bias=False,
                             rpu_config=FloatingPointRPUConfig())
        model = model.cuda().cpu().cuda()
        tile = next(model.analog_tiles())

        self.assertIsNotNone(tile._shared_weight_tensor,
                             "shared tensor should exist after double round-trip")
        self.assertTrue(tile._shared_weight_tensor.is_cuda,
                        "shared tensor should be on CUDA")

    # -- ConstantStep / Inference: CUDA binding --------------------------------
    #
    # CudaAnalogTile.set_shared_weights() works for normal forward, but
    # corrupts the is_perfect forward path.  Verify that:
    #   - ConstantStep and Inference (is_perfect=False) ARE bound on CUDA
    #   - Inference with is_perfect=True is NOT bound (known C++ issue)

    def test_shared_binding_for_constant_step_cuda(self):
        """ConstantStep tile on CUDA should have shared weight binding."""
        self._skip_if_no_cuda()
        rpu = SingleRPUConfig(device=ConstantStepDevice())
        model = AnalogLinear(4, 6, bias=False, rpu_config=rpu).cuda()
        tile = next(model.analog_tiles())
        self.assertIsNotNone(tile._shared_weight_tensor,
                             "ConstantStep CUDA should bind shared weights")
        self.assertTrue(tile._shared_weight_tensor.is_cuda)

    def test_shared_binding_for_inference_default_cuda(self):
        """Inference (is_perfect=False) on CUDA should have shared binding."""
        self._skip_if_no_cuda()
        model = AnalogLinear(4, 6, bias=False,
                             rpu_config=InferenceRPUConfig()).cuda()
        tile = next(model.analog_tiles())
        self.assertIsNotNone(tile._shared_weight_tensor,
                             "Inference (default) CUDA should bind shared weights")

    def test_no_shared_binding_for_is_perfect_cuda(self):
        """Inference with is_perfect=True should NOT bind shared weights."""
        self._skip_if_no_cuda()
        rpu = InferenceRPUConfig()
        rpu.forward.is_perfect = True
        model = AnalogLinear(4, 6, bias=False, rpu_config=rpu).cuda()
        tile = next(model.analog_tiles())
        self.assertIsNone(tile._shared_weight_tensor,
                          "is_perfect=True should skip shared binding")

    def test_inference_is_perfect_forward_nonzero_cuda(self):
        """Inference tile with is_perfect=True should produce non-zero output."""
        self._skip_if_no_cuda()
        rpu = InferenceRPUConfig()
        rpu.forward.is_perfect = True
        manual_seed(42)
        model = AnalogLinear(4, 2, bias=False, rpu_config=rpu).cuda()
        x = randn(1, 4, device="cuda")
        out = model(x)
        self.assertTrue(out.abs().sum() > 0,
                        "is_perfect forward should produce non-zero output")

    def test_training_converges_constant_step_cuda(self):
        """Training with ConstantStep on CUDA should converge after binding."""
        self._skip_if_no_cuda()
        from torch.nn.functional import mse_loss
        from aihwkit.optim import AnalogSGD

        rpu = SingleRPUConfig(device=ConstantStepDevice())
        manual_seed(42)
        model = AnalogLinear(4, 2, bias=False, rpu_config=rpu).cuda()
        x = randn(10, 4, device="cuda")
        y = randn(10, 2, device="cuda")

        initial_loss = mse_loss(model(x), y).item()

        opt = AnalogSGD(model.parameters(), lr=0.1)
        opt.regroup_param_groups(model)
        for _ in range(50):
            opt.zero_grad()
            loss = mse_loss(model(x), y)
            loss.backward()
            opt.step()

        final_loss = mse_loss(model(x), y).item()
        self.assertLess(final_loss, initial_loss,
                        "training should reduce loss on CUDA ConstantStep")

    def test_training_converges_inference_is_perfect_cuda(self):
        """Training Inference+is_perfect on CUDA should converge (no binding)."""
        self._skip_if_no_cuda()
        from torch.nn.functional import mse_loss
        from aihwkit.optim import AnalogSGD

        rpu = InferenceRPUConfig()
        rpu.forward.is_perfect = True
        manual_seed(4321)
        model = AnalogLinear(4, 2, bias=False, rpu_config=rpu).cuda()
        x = randn(10, 4, device="cuda")
        y = randn(10, 2, device="cuda")

        initial_loss = mse_loss(model(x), y).item()

        opt = AnalogSGD(model.parameters(), lr=0.5)
        opt.regroup_param_groups(model)
        for _ in range(100):
            opt.zero_grad()
            loss = mse_loss(model(x), y)
            loss.backward()
            opt.step()

        final_loss = mse_loss(model(x), y).item()
        self.assertLess(final_loss, initial_loss,
                        "training should reduce loss for Inference CUDA")

    # -- Transposed layout verification ---------------------------------------

    def test_cuda_get_weights_cuda_binding(self):
        """get_weights_cuda() binding should return [x_size, d_size] CUDA tensor."""
        self._skip_if_no_cuda()
        model = AnalogLinear(8, 4, bias=False,
                             rpu_config=FloatingPointRPUConfig()).cuda()
        tile = next(model.analog_tiles())

        if not hasattr(tile.tile, "get_weights_cuda"):
            raise SkipTest("get_weights_cuda binding is not exposed in this build")
        out = tile.tile.get_weights_cuda()
        d = tile.tile.get_d_size()
        x = tile.tile.get_x_size()
        self.assertTrue(out.is_cuda, "get_weights_cuda() should return CUDA tensor")
        self.assertEqual(out.shape[0], x,
                         f"dim 0 should be x_size={x}")
        self.assertEqual(out.shape[1], d,
                         f"dim 1 should be d_size={d}")
        # Values must match get_weights() (which returns [d_size, x_size] on CPU).
        w = tile.tile.get_weights()
        self.assertTrue(allclose(out.t().cpu(), w),
                        "get_weights_cuda().t().cpu() should match get_weights()")

    def test_get_tile_weights_ref_returns_transposed_view_for_cuda(self):
        """_get_tile_weights_ref should return a CUDA tensor in standard (d_size, x_size) layout.

        CUDA C++ tiles store weights in transposed layout (x_size, d_size).
        _get_tile_weights_ref returns a standard (d_size, x_size) CUDA tensor,
        either through shared weights or the optional get_weights_cuda fast path.
        """
        self._skip_if_no_cuda()
        model = AnalogLinear(8, 4, bias=False,
                             rpu_config=FloatingPointRPUConfig()).cuda()
        tile = next(model.analog_tiles())

        ref = tile._get_tile_weights_ref()
        # Should be a CUDA tensor, not a CPU copy.
        self.assertTrue(ref.is_cuda,
                        "_get_tile_weights_ref should return CUDA tensor")
        # Shape should be standard (d_size, x_size), not transposed.
        w = tile.tile.get_weights()
        self.assertEqual(ref.shape, w.shape,
                         "ref shape should match get_weights() shape")
        # Values should match.
        self.assertTrue(allclose(ref.cpu(), w),
                        "_get_tile_weights_ref should match get_weights()")

    # -- Raw tile .to('cuda') -------------------------------------------------

    def test_raw_tile_to_cuda_shared_binding(self):
        """Raw tile (not AnalogLinear) should retain shared binding after .to('cuda').

        Reproduces the scenario where a tile is constructed directly via
        rpu.get_default_tile_module_class() and moved with .to('cuda').
        """
        self._skip_if_no_cuda()
        rpu = SingleRPUConfig(device=ConstantStepDevice())
        cls = rpu.get_default_tile_module_class(16, 8)
        tile = cls(16, 8, rpu, False)

        # CPU: shared binding from __init__
        self.assertIsNotNone(tile._shared_weight_tensor)
        r1 = tile._get_tile_weights_ref()
        r2 = tile._get_tile_weights_ref()
        self.assertEqual(r1.data_ptr(), r2.data_ptr(),
                         "CPU: _get_tile_weights_ref should return same ptr")

        # .to('cuda'): shared binding must survive
        tile = tile.to("cuda")
        self.assertIsNotNone(tile._shared_weight_tensor,
                             "shared tensor should exist after .to('cuda')")
        self.assertTrue(tile._shared_weight_tensor.is_cuda)
        r1 = tile._get_tile_weights_ref()
        r2 = tile._get_tile_weights_ref()
        self.assertEqual(r1.data_ptr(), r2.data_ptr(),
                         "CUDA: _get_tile_weights_ref should return same ptr")


class AnalogCtxOptimizerInteropTest(ParametrizedTestCase):
    """Optimizer-facing AnalogContext behaviour.

    These guard the interception of ``grad``, ``requires_grad`` and the
    BUFFER-mode in-place writes that mixed-precision / TT optimizers depend on.
    Before the fix, ``grad`` reads were redirected to a transient ``.data`` view
    (always ``None``), ``requires_grad`` reported ``False`` (so the context was
    dropped from the optimizer param groups), and BUFFER-mode in-place ops were
    blocked by the always-read-only guard.
    """

    use_cuda = False

    # (label, factory) pairs covering the Python inference tile and the C++
    # analog devices (SoftBounds). These behaviours are about the
    # AnalogContext Python wrapper, so they must hold for every backing tile.
    def _rpu_configs(self):
        """Return fresh (label, rpu_config) pairs to parametrize over."""
        return [
            ("torch_inference", TorchInferenceRPUConfig()),
            ("softbounds", SingleRPUConfig(device=SoftBoundsDevice())),
        ]

    def _ctx(self, rpu_config, bias=False):
        """Return the analog_ctx of a freshly built AnalogLinear."""
        model = AnalogLinear(4, 6, bias=bias, rpu_config=rpu_config)
        return next(model.analog_tiles()).analog_ctx

    def test_requires_grad_is_true(self):
        """analog_ctx.requires_grad must be True so optimizers keep the context."""
        for label, rpu_config in self._rpu_configs():
            with self.subTest(rpu_config=label):
                ctx = self._ctx(rpu_config)
                self.assertTrue(ctx.requires_grad)

    def test_grad_get_set_roundtrip(self):
        """Setting analog_ctx.grad and reading it back must return the same tensor."""
        for label, rpu_config in self._rpu_configs():
            with self.subTest(rpu_config=label):
                ctx = self._ctx(rpu_config)
                grad = randn(Size(ctx.shape))

                ctx.grad = grad

                self.assertIsNotNone(ctx.grad, "analog_ctx.grad read returned None after set")
                self.assertTrue(allclose(ctx.grad, grad))

    def test_grad_none_by_default(self):
        """A fresh analog_ctx has no gradient yet (read does not raise)."""
        for label, rpu_config in self._rpu_configs():
            with self.subTest(rpu_config=label):
                ctx = self._ctx(rpu_config)
                self.assertIsNone(ctx.grad)

    def test_buffer_mode_inplace_add_allowed(self):
        """BUFFER mode exposes an independent writable digital buffer.

        In-place ops on the context (e.g. AdamW's ``mul_``/``addcdiv_``) must
        write the buffer instead of raising the read-only error.
        """
        for label, rpu_config in self._rpu_configs():
            with self.subTest(rpu_config=label):
                ctx = self._ctx(rpu_config)
                ctx.enable_buffer()

                # Buffer starts zero-initialised with the logical weight shape.
                self.assertEqual(ctx.data.shape, Size([6, 4]))
                self.assertEqual(float(ctx.data.sum()), 0.0)

                ctx.add_(2.0)
                self.assertTrue(allclose(ctx.data, torch.full((6, 4), 2.0)))

                ctx.mul_(3.0)
                self.assertTrue(allclose(ctx.data, torch.full((6, 4), 6.0)))

    def test_data_view_mode_inplace_still_blocked(self):
        """Non-BUFFER modes keep the read-only guard on in-place writes."""
        for label, rpu_config in self._rpu_configs():
            with self.subTest(rpu_config=label):
                ctx = self._ctx(rpu_config)
                ctx.enable_data_view()
                with self.assertRaises(RuntimeError):
                    ctx.add_(1.0)


class AnalogCtxPlaceholderReprTest(ParametrizedTestCase):
    """Placeholder-mode metadata access must not read meaningless values."""

    use_cuda = False

    def _rpu_configs(self):
        """Return fresh (label, rpu_config) pairs to parametrize over."""
        return [
            ("torch_inference", TorchInferenceRPUConfig()),
            ("softbounds", SingleRPUConfig(device=SoftBoundsDevice())),
            ("constant_step", SingleRPUConfig(device=ConstantStepDevice())),
        ]

    def _ctx(self, rpu_config):
        model = AnalogLinear(4, 6, bias=False, rpu_config=rpu_config)
        return next(model.analog_tiles()).analog_ctx

    def test_placeholder_data_repr_is_metadata_only(self):
        """repr(analog_ctx.data) in placeholder mode returns a metadata summary.

        The old PlaceholderDataView had no __repr__, so the default tensor repr
        tried to read values and raised RuntimeError.
        """
        for label, rpu_config in self._rpu_configs():
            with self.subTest(rpu_config=label):
                ctx = self._ctx(rpu_config)
                ctx.enable_placeholder()

                text = repr(ctx.data)
                self.assertIn("PlaceholderDataView", text)
                self.assertIn("shape=", text)
                # str() goes through the same __repr__ and must not raise either.
                self.assertEqual(str(ctx.data), text)

    def test_placeholder_type_metadata_allowed(self):
        """analog_ctx.type() is metadata and must work in placeholder mode.

        ``type`` was newly added to the placeholder metadata allow-list; the old
        code raised the placeholder read error.
        """
        for label, rpu_config in self._rpu_configs():
            with self.subTest(rpu_config=label):
                ctx = self._ctx(rpu_config)
                ctx.enable_placeholder()

                self.assertIn("Tensor", ctx.type())
                self.assertIn("Tensor", ctx.data.type())


class AnalogCtxRequiresGradBackwardTest(ParametrizedTestCase):
    """Toggling analog_ctx.requires_grad must keep backward() correct.

    Freezing/unfreezing an analog layer is done via ``ctx.requires_grad``.
    The setter must work in the default PLACEHOLDER mode (its dispatch used to
    be routed through __torch_function__ as ``__set__`` and raised a placeholder
    read error), and backward() must remain runnable in both states:

    * requires_grad True  -> the analog gradient trace is recorded (the analog
      optimizer would update the tile).
    * requires_grad False -> the analog layer is frozen (no trace recorded), yet
      downstream digital parameters still receive gradients, so backward() does
      not raise.
    """

    use_cuda = False

    # Analog-training configs whose backward records the analog gradient trace
    # (i.e. ``has_gradient()`` is the freeze signal). Inference tiles use the
    # torch-update path instead and are intentionally excluded here.
    def _rpu_configs(self):
        """Return fresh (label, rpu_config) pairs to parametrize over."""
        return [
            ("floating_point", FloatingPointRPUConfig()),
            ("softbounds", SingleRPUConfig(device=SoftBoundsDevice())),
            ("constant_step", SingleRPUConfig(device=ConstantStepDevice())),
        ]

    @staticmethod
    def _build(rpu_config):
        """Build an analog layer with a learnable digital scale.

        ``learn_out_scaling`` adds a downstream digital parameter so the loss
        always has a grad path, letting backward() run even when the analog
        context is frozen.
        """
        rpu_config.mapping.learn_out_scaling = True
        model = AnalogLinear(4, 3, bias=False, rpu_config=rpu_config)
        tile = next(model.analog_tiles())
        return model, tile.analog_ctx, tile.out_scaling_alpha

    @staticmethod
    def _fwd_bwd(model, ctx, alpha):
        """Run one forward/backward with a non-grad input; return signals.

        Returns ``(out.requires_grad, ctx.has_gradient(), alpha.grad, ctx.grad)``.
        ``ctx.grad`` must stay None: AnalogFunction.backward returns None for the
        context, so autograd never populates its ``.grad`` slot; the gradient
        information is captured in the analog trace (``has_gradient()``) instead.
        """
        ctx.reset()  # clear the analog gradient trace
        alpha.grad = None
        x = randn(2, 4)  # leaf input, requires_grad=False
        out = model(x)
        requires_grad = out.requires_grad
        out.sum().backward()
        return requires_grad, ctx.has_gradient(), alpha.grad, ctx.grad

    def test_set_requires_grad_in_default_mode_does_not_raise(self):
        """Setting requires_grad in default (placeholder) mode must not raise.

        Regression: the setter dispatched through __torch_function__ as
        ``__set__`` and raised the placeholder read error.
        """
        for label, rpu_config in self._rpu_configs():
            with self.subTest(rpu_config=label):
                _, ctx, _ = self._build(rpu_config)

                ctx.requires_grad = False
                self.assertFalse(ctx.requires_grad)

                ctx.requires_grad = True
                self.assertTrue(ctx.requires_grad)

    def test_backward_true_to_false_freezes_analog_gradient(self):
        """True -> False: backward still runs, but analog trace stops recording."""
        for label, rpu_config in self._rpu_configs():
            with self.subTest(rpu_config=label):
                model, ctx, alpha = self._build(rpu_config)

                # requires_grad True: analog gradient recorded, digital param learns.
                # Both the forward signal (analog_input) and the backward signal
                # (analog_grad_output) must be attached to the trace.
                ctx.requires_grad = True
                requires_grad, has_grad, alpha_grad, ctx_grad = self._fwd_bwd(model, ctx, alpha)
                self.assertTrue(requires_grad)
                self.assertTrue(has_grad, "forward signal must be recorded when requires_grad=True")
                self.assertEqual(
                    len(ctx.analog_grad_output), 1, "backward signal must be attached"
                )
                self.assertIsNotNone(alpha_grad)
                # The analog context never uses the autograd .grad slot.
                self.assertIsNone(ctx_grad, "ctx.grad must stay None; gradient lives in the trace")

                # requires_grad False: backward must still run without error, but the
                # analog layer is frozen (neither signal recorded); the digital scale
                # keeps learning.
                ctx.requires_grad = False
                _, has_grad, alpha_grad, ctx_grad = self._fwd_bwd(model, ctx, alpha)
                self.assertFalse(has_grad, "forward signal must NOT be recorded when frozen")
                self.assertEqual(
                    len(ctx.analog_grad_output), 0, "no backward signal when frozen"
                )
                self.assertIsNotNone(alpha_grad, "downstream digital param must still get grad")
                self.assertIsNone(ctx_grad, "ctx.grad must stay None when frozen")

    def test_backward_false_to_true_unfreezes_analog_gradient(self):
        """False -> True: re-enabling restores analog gradient recording."""
        for label, rpu_config in self._rpu_configs():
            with self.subTest(rpu_config=label):
                model, ctx, alpha = self._build(rpu_config)

                # requires_grad False: frozen, neither signal recorded.
                ctx.requires_grad = False
                _, has_grad, alpha_grad, ctx_grad = self._fwd_bwd(model, ctx, alpha)
                self.assertFalse(has_grad)
                self.assertEqual(len(ctx.analog_grad_output), 0, "no backward signal when frozen")
                self.assertIsNotNone(alpha_grad)
                self.assertIsNone(ctx_grad, "ctx.grad must stay None when frozen")

                # requires_grad True: both signals recorded again.
                ctx.requires_grad = True
                requires_grad, has_grad, alpha_grad, ctx_grad = self._fwd_bwd(model, ctx, alpha)
                self.assertTrue(requires_grad)
                self.assertTrue(
                    has_grad, "forward signal must resume after re-enabling requires_grad"
                )
                self.assertEqual(
                    len(ctx.analog_grad_output), 1, "backward signal must resume after re-enabling"
                )
                self.assertIsNotNone(alpha_grad)
                # The analog context never uses the autograd .grad slot.
                self.assertIsNone(ctx_grad, "ctx.grad must stay None; gradient lives in the trace")


class AnalogCtxExactGradientTest(ParametrizedTestCase):
    """The analog backward must produce the exact gradient, including out_scaling.

    Uses a noise-free FloatingPoint tile with known weights and a known, non-unit
    out_scaling vector ``alpha``, and a quadratic loss ``L = 0.5 * sum((y - t)^2)``.
    Unlike a linear loss, ``dL/dy = (y - t)`` depends on the forward output, so the
    backward must thread the out_scaling through correctly. With ``raw = x @ W^T``
    and ``y = alpha * raw`` the chain rule gives closed forms the test checks:

    * ``analog_input``        == ``x``                       (forward activation)
    * ``analog_grad_output``  == ``(y - t) * alpha``         (grad w.r.t. tile output)
    * ``x.grad``              == ``((y - t) * alpha) @ W``   (grad w.r.t. the input)
    * ``out_scaling.grad``    == ``sum_b (y - t) * raw``     (grad w.r.t. alpha)

    The ``alpha`` factor is the point: a backward that dropped or misapplied
    out_scaling would still pass with ``alpha == 1`` but fail here.
    """

    use_cuda = False

    def test_exact_gradient_with_non_unit_out_scaling(self):
        """Quadratic-loss backward gradients match the closed form when alpha != 1."""
        rpu_config = FloatingPointRPUConfig()
        rpu_config.mapping.weight_scaling_omega = 0.0  # logical weight == set weight
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.out_scaling_columnwise = True  # per-output-channel alpha

        model = AnalogLinear(3, 2, bias=False, rpu_config=rpu_config)
        tile = next(model.analog_tiles())
        ctx = tile.analog_ctx

        # ``weight`` is the physical tile weight (used by the forward MVM);
        # ``get_weights`` returns the logical weight = physical * out_scaling.
        weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        alpha = torch.tensor([0.5, 2.0])  # non-unit out_scaling, one per output
        tile.set_weights(weight, None)
        with torch.no_grad():
            tile.out_scaling_alpha.copy_(alpha.view_as(tile.out_scaling_alpha))

        # Sanity: physical weight set cleanly (no hidden mapping scale), and the
        # logical weight folds in out_scaling per output channel.
        self.assertTrue(allclose(tile.get_weights()[0], weight * alpha.view(-1, 1)))

        ctx.reset()
        x = torch.tensor([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]], requires_grad=True)
        target = torch.tensor([[0.5, -1.0], [1.0, 2.0]])

        out = model(x)
        # Forward must be alpha * (x @ W^T).
        raw = x.detach() @ weight.t()
        self.assertTrue(allclose(out.detach(), raw * alpha), "forward out_scaling is wrong")

        # Quadratic (MSE-style) loss: dL/dy = (y - target).
        loss = 0.5 * ((out - target) ** 2).sum()
        loss.backward()

        residual = (raw * alpha) - target  # dL/dy = y - target

        # Forward signal: the activation that entered the tile.
        self.assertTrue(allclose(ctx.analog_input[0], x.detach()))
        # Backward signal: gradient w.r.t. the (pre-scaling) tile output.
        self.assertTrue(
            allclose(ctx.analog_grad_output[0], residual * alpha),
            "analog_grad_output must include the out_scaling factor",
        )
        # Gradient w.r.t. the input flows back through W and alpha.
        self.assertTrue(
            allclose(x.grad, (residual * alpha) @ weight),
            "input gradient must account for out_scaling",
        )
        # Gradient w.r.t. alpha is sum over the batch of (y - target) * raw.
        self.assertTrue(
            allclose(tile.out_scaling_alpha.grad.flatten(), (residual * raw).sum(dim=0)),
            "out_scaling gradient is wrong",
        )


# Configs covering the three backing-tile flavors that the AnalogContext Python
# wrapper sits on top of: a pure-Python inference tile, a C++ FloatingPoint tile,
# and a C++ analog-device tile. Attribute handling must hold for all of them.
def _ctx_attribute_rpu_configs():
    """Return fresh (label, rpu_config) pairs to parametrize attribute tests."""
    return [
        ("torch_inference", TorchInferenceRPUConfig()),
        ("floating_point", FloatingPointRPUConfig()),
        ("softbounds", SingleRPUConfig(device=SoftBoundsDevice())),
    ]


class AnalogCtxTabCompletionTest(ParametrizedTestCase):
    """dir(analog_ctx) must list tensor ops for interactive tab-completion.

    ``Tensor.__dir__`` dispatches through ``__torch_function__`` (this class
    defines one); in PLACEHOLDER mode that routed ``__dir__`` through the
    weight-read guard and raised, so completers (rlcompleter / IPython) caught
    the error and offered nothing. ``AnalogContext.__dir__`` now lists the class
    and instance attribute names directly, without a value-read dispatch.
    """

    use_cuda = False

    def _ctx(self, rpu_config):
        model = AnalogLinear(4, 6, bias=False, rpu_config=rpu_config)
        return next(model.analog_tiles()).analog_ctx

    def test_dir_does_not_raise_in_placeholder(self):
        """dir(ctx) must not raise in the default placeholder mode."""
        for label, rpu_config in _ctx_attribute_rpu_configs():
            with self.subTest(rpu_config=label):
                ctx = self._ctx(rpu_config)
                self.assertEqual(ctx.data_view_mode, AnalogContextDataViewMode.PLACEHOLDER)
                names = dir(ctx)  # must not raise
                self.assertIsInstance(names, list)

    def test_dir_lists_tensor_ops_and_ctx_methods(self):
        """Completion must offer both tensor ops and AnalogContext-specific names."""
        ctx = self._ctx(TorchInferenceRPUConfig())
        names = set(dir(ctx))
        for op in ["reshape", "relu", "reciprocal", "norm", "size", "real", "T"]:
            self.assertIn(op, names, f"tensor op '{op}' missing from dir()")
        for member in ["analog_tile", "enable_data_view", "enable_buffer", "reset"]:
            self.assertIn(member, names, f"AnalogContext member '{member}' missing from dir()")

    def test_dir_is_consistent_across_modes(self):
        """The attribute listing must not depend on the active data-view mode."""
        ctx = self._ctx(TorchInferenceRPUConfig())
        placeholder_names = set(dir(ctx))
        ctx.enable_data_view()
        self.assertEqual(set(dir(ctx)), placeholder_names)
        ctx.enable_buffer()
        self.assertEqual(set(dir(ctx)), placeholder_names)


class AnalogCtxMetadataQueryTest(ParametrizedTestCase):
    """Pure metadata queries must work in every mode and return correct values.

    These depend only on dtype / device / layout, never on weight values, so
    they belong in the placeholder allow-list. Before the fix, queries such as
    ``get_device``, ``is_floating_point``, ``is_complex`` and ``is_signed``
    raised the placeholder read error.
    """

    use_cuda = False

    # Metadata methods whose value must equal a plain reference tensor of the
    # same logical shape / dtype / device. ``data_ptr`` is excluded: it returns a
    # storage address, which legitimately differs from the reference tensor.
    METADATA_METHODS = [
        "dim",
        "ndimension",
        "numel",
        "nelement",
        "element_size",
        "is_contiguous",
        "stride",
        "storage_offset",
        "get_device",
        "is_floating_point",
        "is_complex",
        "is_signed",
        "is_pinned",
        "dense_dim",
        "sparse_dim",
        "has_names",
    ]

    def _ctx(self, rpu_config):
        model = AnalogLinear(4, 6, bias=False, rpu_config=rpu_config)
        return next(model.analog_tiles()).analog_ctx

    def test_metadata_queries_correct_in_all_modes(self):
        """Each metadata query must match a reference tensor in every mode."""
        for label, rpu_config in _ctx_attribute_rpu_configs():
            with self.subTest(rpu_config=label):
                ctx = self._ctx(rpu_config)
                reference = torch.empty(ctx.size(), dtype=ctx.dtype, device=ctx.device)

                for enable in (ctx.enable_placeholder, ctx.enable_data_view, ctx.enable_buffer):
                    enable()
                    with self.subTest(mode=ctx.data_view_mode):
                        for method in self.METADATA_METHODS:
                            self.assertEqual(
                                getattr(ctx, method)(),
                                getattr(reference, method)(),
                                f"{method}() mismatch in {ctx.data_view_mode}",
                            )

    def test_data_ptr_is_an_int_in_placeholder(self):
        """data_ptr must answer (an integer address) instead of raising."""
        for label, rpu_config in _ctx_attribute_rpu_configs():
            with self.subTest(rpu_config=label):
                ctx = self._ctx(rpu_config)
                self.assertIsInstance(ctx.data_ptr(), int)


class AnalogCtxValueViewPropertyTest(ParametrizedTestCase):
    """T / mT / H / mH / real / imag must honor the data-view mode.

    These tensor properties materialize weight values, so they must behave like
    their method equivalents (``t()`` / ``conj()``): blocked in PLACEHOLDER mode,
    and served from the logical view / buffer otherwise. Previously the getset
    descriptors fell through to ``__torch_function__`` as the whitelisted
    ``__get__`` and silently returned uninitialized placeholder memory.
    """

    use_cuda = False

    VALUE_VIEW_PROPERTIES = ["T", "mT", "H", "mH", "real", "imag"]
    # ``imag`` is excluded from value checks: it is undefined for real dtypes and
    # raises the same error as a plain real tensor would.
    REAL_VALUED_PROPERTIES = ["T", "mT", "H", "mH", "real"]

    def _tile_ctx(self, rpu_config):
        model = AnalogLinear(4, 6, bias=False, rpu_config=rpu_config)
        tile = next(model.analog_tiles())
        return tile, tile.analog_ctx

    def test_placeholder_blocks_value_view_properties(self):
        """Every value-bearing property must raise the placeholder read error."""
        for label, rpu_config in _ctx_attribute_rpu_configs():
            with self.subTest(rpu_config=label):
                _, ctx = self._tile_ctx(rpu_config)
                for prop in self.VALUE_VIEW_PROPERTIES:
                    with self.subTest(prop=prop):
                        with self.assertRaises(RuntimeError):
                            getattr(ctx, prop)

    def test_data_view_properties_match_logical_weight(self):
        """In data_view mode the properties must reflect the logical weights."""
        for label, rpu_config in _ctx_attribute_rpu_configs():
            with self.subTest(rpu_config=label):
                tile, ctx = self._tile_ctx(rpu_config)
                ctx.enable_data_view()
                logical, _ = tile.get_weights()

                for prop in self.REAL_VALUED_PROPERTIES:
                    with self.subTest(prop=prop):
                        ctx_view = getattr(ctx, prop).detach().cpu()
                        expected = getattr(logical, prop)
                        self.assertEqual(ctx_view.shape, expected.shape)
                        self.assertTrue(
                            allclose(ctx_view, expected), f"ctx.{prop} != logical.{prop}"
                        )

    def test_buffer_properties_reflect_buffer(self):
        """In buffer mode the properties must reflect the digital buffer."""
        for label, rpu_config in _ctx_attribute_rpu_configs():
            with self.subTest(rpu_config=label):
                tile, ctx = self._tile_ctx(rpu_config)
                ctx.enable_buffer()
                ctx.data.add_(5.0)  # buffer is now uniformly 5.0
                expected = torch.full((tile.out_size, tile.in_size), 5.0)

                self.assertTrue(allclose(ctx.T.detach().cpu(), expected.T))
                self.assertTrue(allclose(ctx.real.detach().cpu(), expected))


class AnalogCtxRequiresGradMethodTest(ParametrizedTestCase):
    """``requires_grad_()`` (method) must toggle the real autograd flag.

    Regression: ``requires_grad_`` was allow-listed but the read-only in-place
    guard rejected it first (it ends with ``_``), and the data-view redirection
    would have set the flag on a throwaway placeholder. Consequently the standard
    PyTorch freezing idiom ``module.requires_grad_(False)`` -- which iterates the
    parameters and calls ``param.requires_grad_()`` -- raised on the context.
    """

    use_cuda = False

    def _model_ctx(self, rpu_config, bias=False):
        model = AnalogLinear(4, 6, bias=bias, rpu_config=rpu_config)
        return model, next(model.analog_tiles()).analog_ctx

    def test_method_toggles_flag_and_returns_self(self):
        """ctx.requires_grad_(flag) must set the flag and return the context."""
        for label, rpu_config in _ctx_attribute_rpu_configs():
            with self.subTest(rpu_config=label):
                _, ctx = self._model_ctx(rpu_config)

                self.assertIs(ctx.requires_grad_(False), ctx)
                self.assertFalse(ctx.requires_grad)
                ctx.requires_grad_(True)
                self.assertTrue(ctx.requires_grad)

    def test_method_defaults_to_true(self):
        """ctx.requires_grad_() with no argument must enable gradients."""
        _, ctx = self._model_ctx(TorchInferenceRPUConfig())
        ctx.requires_grad = False
        ctx.requires_grad_()
        self.assertTrue(ctx.requires_grad)

    def test_module_requires_grad_freezes_analog_layer(self):
        """nn.Module.requires_grad_() must (un)freeze the context without raising."""
        for label, rpu_config in _ctx_attribute_rpu_configs():
            with self.subTest(rpu_config=label):
                model, ctx = self._model_ctx(rpu_config)

                model.requires_grad_(False)
                self.assertFalse(ctx.requires_grad)
                model.requires_grad_(True)
                self.assertTrue(ctx.requires_grad)

    def test_method_works_in_every_mode(self):
        """The flag toggle must not depend on the active data-view mode."""
        for enable in ("enable_placeholder", "enable_data_view", "enable_buffer"):
            with self.subTest(mode=enable):
                _, ctx = self._model_ctx(TorchInferenceRPUConfig())
                getattr(ctx, enable)()
                ctx.requires_grad_(False)
                self.assertFalse(ctx.requires_grad)
                ctx.requires_grad_(True)
                self.assertTrue(ctx.requires_grad)

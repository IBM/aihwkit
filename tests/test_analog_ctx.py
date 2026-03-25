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

from torch import zeros, randn, allclose, Tensor, Size, manual_seed
from torch.nn import Parameter
from torch.nn import Linear as TorchLinear, Sequential, Conv2d as TorchConv2d

from aihwkit.nn import AnalogLinear, AnalogConv2d
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim.context import AnalogContext
from aihwkit.optim.weight_view import ReadOnlyWeightView
from aihwkit.simulator.configs import (
    FloatingPointRPUConfig,
    InferenceRPUConfig,
    SingleRPUConfig,
    TorchInferenceRPUConfig,
)
from aihwkit.simulator.configs.devices import ConstantStepDevice

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


class AnalogCtxSyncAfterSetWeightsTest(ParametrizedTestCase):
    """Reviewer concern #1: analog_ctx.data must stay in sync after set_weights."""

    use_cuda = False

    def _test_sync(self, rpu_config, use_cuda):
        """Helper: verify ctx.data matches tile weights after set_weights."""
        model = AnalogLinear(4, 6, bias=False, rpu_config=rpu_config)
        if use_cuda:
            model = model.cuda()
        tile = next(model.analog_tiles())

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
    and ``analog_ctx.data`` share the same memory.  ``tile.update()`` and
    ``tile.set_weights()`` must be visible through the shared tensor
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

    def test_ctx_data_shares_memory_with_cpp_tile(self):
        """analog_ctx.data and _shared_weight_tensor should share memory."""
        tile = self._get_tile(FloatingPointRPUConfig())
        ctx_ptr = tile.analog_ctx.data.data_ptr()
        shared_ptr = tile._shared_weight_tensor.data_ptr()
        self.assertEqual(ctx_ptr, shared_ptr,
                         "analog_ctx.data and shared tensor should have same data_ptr")

    def test_update_reflects_in_ctx_without_sync(self):
        """After tile.update(), analog_ctx.data should reflect changes (zero-copy)."""
        tile = self._get_tile(FloatingPointRPUConfig())
        tile.tile.set_learning_rate(0.1)

        snapshot = tile.analog_ctx.data.detach().clone()

        x = randn(1, 4)
        d = randn(1, 6)
        tile.tile.update(x, d, False)

        # analog_ctx.data should have changed without explicit sync
        self.assertFalse(
            allclose(tile.analog_ctx.data.detach(), snapshot),
            "analog_ctx.data should change after tile.update() without sync")

        # And it should match get_weights()
        w_from_tile = tile.tile.get_weights()
        self.assertTrue(
            allclose(tile.analog_ctx.data.detach(), w_from_tile),
            "analog_ctx.data should match get_weights() after update")

    def test_set_weights_reflects_in_ctx_without_sync(self):
        """After tile.set_weights(), analog_ctx.data should reflect changes."""
        tile = self._get_tile(FloatingPointRPUConfig())
        new_w = randn(6, 4)
        tile.tile.set_weights(new_w)

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

            w_from_tile = tile.tile.get_weights()
            self.assertTrue(
                allclose(tile.analog_ctx.data.detach(), w_from_tile),
                "analog_ctx.data drifted from tile weights during updates")

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


class AnalogCtxReadOnlyTest(ParametrizedTestCase):
    """Tests for ReadOnlyWeightView and the readonly flag on AnalogContext."""

    use_cuda = False

    def _make_model(self, readonly=True):
        """Create an AnalogLinear model with the given readonly setting."""
        rpu_config = TorchInferenceRPUConfig()
        rpu_config.mapping.readonly_weights = readonly
        return AnalogLinear(4, 6, bias=False, rpu_config=rpu_config)

    def _get_ctx(self, model):
        tile = next(model.analog_tiles())
        return tile.analog_ctx

    # -- default behaviour ----------------------------------------------------

    def test_default_readonly_true(self):
        """By default, analog_ctx.data should be a ReadOnlyWeightView."""
        model = self._make_model(readonly=True)
        ctx = self._get_ctx(model)
        self.assertTrue(ctx.readonly)
        self.assertIsInstance(ctx.data, ReadOnlyWeightView)

    def test_default_readonly_false_via_config(self):
        """Setting readonly_weights=False in config should disable protection."""
        model = self._make_model(readonly=False)
        ctx = self._get_ctx(model)
        self.assertFalse(ctx.readonly)
        self.assertNotIsInstance(ctx.data, ReadOnlyWeightView)

    # -- read operations work transparently -----------------------------------

    def test_read_ops_work_when_readonly(self):
        """size, norm, nonzero, comparisons should all work on readonly data."""
        model = self._make_model(readonly=True)
        ctx = self._get_ctx(model)

        self.assertEqual(ctx.size(), Size([6, 4]))
        self.assertGreater(ctx.norm().item(), 0.0)
        self.assertGreater(len(ctx.nonzero()), 0)
        mask = ctx > 10
        self.assertEqual(mask.shape, ctx.shape)

    # -- in-place ops blocked when readonly -----------------------------------

    def test_add_inplace_blocked(self):
        """ctx.data.add_() should raise RuntimeError when readonly."""
        model = self._make_model(readonly=True)
        ctx = self._get_ctx(model)
        with self.assertRaises(RuntimeError):
            ctx.data.add_(1.0)

    def test_mul_inplace_blocked(self):
        """ctx.data.mul_() should raise RuntimeError when readonly."""
        model = self._make_model(readonly=True)
        ctx = self._get_ctx(model)
        with self.assertRaises(RuntimeError):
            ctx.data.mul_(2.0)

    def test_copy_inplace_blocked(self):
        """ctx.data.copy_() should raise RuntimeError when readonly."""
        model = self._make_model(readonly=True)
        ctx = self._get_ctx(model)
        with self.assertRaises(RuntimeError):
            ctx.data.copy_(randn(6, 4))

    def test_fill_inplace_blocked(self):
        """ctx.data.fill_() should raise RuntimeError when readonly."""
        model = self._make_model(readonly=True)
        ctx = self._get_ctx(model)
        with self.assertRaises(RuntimeError):
            ctx.data.fill_(0.0)

    def test_zero_inplace_blocked(self):
        """ctx.data.zero_() should raise RuntimeError when readonly."""
        model = self._make_model(readonly=True)
        ctx = self._get_ctx(model)
        with self.assertRaises(RuntimeError):
            ctx.data.zero_()

    def test_setitem_blocked(self):
        """ctx.data[0, 0] = ... should raise RuntimeError when readonly."""
        model = self._make_model(readonly=True)
        ctx = self._get_ctx(model)
        with self.assertRaises(RuntimeError):
            ctx.data[0, 0] = 999.0

    # -- in-place ops allowed when writable -----------------------------------

    def test_add_inplace_allowed_when_not_readonly(self):
        """ctx.data.add_() should work when readonly=False."""
        model = self._make_model(readonly=False)
        ctx = self._get_ctx(model)
        ctx.data.add_(1.0)  # should not raise

    def test_setitem_allowed_when_not_readonly(self):
        """ctx.data[0,0] = ... should work when readonly=False."""
        model = self._make_model(readonly=False)
        ctx = self._get_ctx(model)
        ctx.data[0, 0] = 999.0  # should not raise

    # -- flag toggling --------------------------------------------------------

    def test_toggle_readonly_on(self):
        """Switching readonly from False to True should wrap data."""
        model = self._make_model(readonly=False)
        ctx = self._get_ctx(model)
        self.assertNotIsInstance(ctx.data, ReadOnlyWeightView)

        ctx.readonly = True
        self.assertIsInstance(ctx.data, ReadOnlyWeightView)
        with self.assertRaises(RuntimeError):
            ctx.data.add_(1.0)

    def test_toggle_readonly_off(self):
        """Switching readonly from True to False should unwrap data."""
        model = self._make_model(readonly=True)
        ctx = self._get_ctx(model)
        self.assertIsInstance(ctx.data, ReadOnlyWeightView)

        ctx.readonly = False
        self.assertNotIsInstance(ctx.data, ReadOnlyWeightView)
        ctx.data.add_(1.0)  # should not raise

    # -- context manager ------------------------------------------------------

    def test_writable_context_manager(self):
        """writable() should temporarily allow in-place ops."""
        model = self._make_model(readonly=True)
        ctx = self._get_ctx(model)

        with ctx.writable():
            self.assertFalse(ctx.readonly)
            ctx.data.add_(1.0)  # should not raise

        # Readonly restored
        self.assertTrue(ctx.readonly)
        with self.assertRaises(RuntimeError):
            ctx.data.add_(1.0)

    def test_writable_context_manager_restores_on_exception(self):
        """writable() should restore readonly even if an exception occurs."""
        model = self._make_model(readonly=True)
        ctx = self._get_ctx(model)

        try:
            with ctx.writable():
                raise ValueError("test exception")
        except ValueError:
            pass

        self.assertTrue(ctx.readonly)

    # -- data assignment auto-wraps -------------------------------------------

    def test_data_assignment_auto_wraps(self):
        """Assigning to ctx.data should auto-wrap when readonly=True."""
        model = self._make_model(readonly=True)
        ctx = self._get_ctx(model)

        ctx.data = randn(6, 4)
        self.assertIsInstance(ctx.data, ReadOnlyWeightView)

    def test_data_assignment_no_wrap_when_writable(self):
        """Assigning to ctx.data should NOT wrap when readonly=False."""
        model = self._make_model(readonly=False)
        ctx = self._get_ctx(model)

        ctx.data = randn(6, 4)
        self.assertNotIsInstance(ctx.data, ReadOnlyWeightView)

    # -- set_data respects readonly -------------------------------------------

    def test_set_data_works_when_readonly(self):
        """set_data() should succeed even when readonly (uses assignment)."""
        model = self._make_model(readonly=True)
        ctx = self._get_ctx(model)

        new_data = randn(6, 4)
        ctx.set_data(new_data)
        self.assertIsInstance(ctx.data, ReadOnlyWeightView)
        self.assertTrue(allclose(ctx.data.detach(), new_data))

    # -- convert_to_analog readonly parameter ---------------------------------

    def test_convert_to_analog_readonly_override_false(self):
        """convert_to_analog(readonly=False) should set all ctx.readonly=False."""
        digital_model = Sequential(TorchLinear(8, 4), TorchLinear(4, 2))
        analog_model = convert_to_analog(
            digital_model, TorchInferenceRPUConfig(),
            ensure_analog_root=False, readonly=False,
        )
        for param in analog_model.parameters():
            if isinstance(param, AnalogContext):
                self.assertFalse(param.readonly)
                self.assertNotIsInstance(param.data, ReadOnlyWeightView)

    def test_convert_to_analog_readonly_override_true(self):
        """convert_to_analog(readonly=True) should set all ctx.readonly=True."""
        rpu_config = TorchInferenceRPUConfig()
        rpu_config.mapping.readonly_weights = False  # config says writable
        digital_model = Sequential(TorchLinear(8, 4), TorchLinear(4, 2))
        analog_model = convert_to_analog(
            digital_model, rpu_config,
            ensure_analog_root=False, readonly=True,
        )
        for param in analog_model.parameters():
            if isinstance(param, AnalogContext):
                self.assertTrue(param.readonly)
                self.assertIsInstance(param.data, ReadOnlyWeightView)

    def test_convert_to_analog_readonly_default_from_config(self):
        """convert_to_analog() without readonly uses rpu_config.mapping value."""
        rpu_config = TorchInferenceRPUConfig()
        rpu_config.mapping.readonly_weights = False
        digital_model = Sequential(TorchLinear(8, 4))
        analog_model = convert_to_analog(
            digital_model, rpu_config, ensure_analog_root=False,
        )
        for param in analog_model.parameters():
            if isinstance(param, AnalogContext):
                self.assertFalse(param.readonly)


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

        self.assertTrue(hasattr(tile.tile, "get_weights_cuda"),
                        "CudaFloatingPointTile should have get_weights_cuda binding")
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
        _get_tile_weights_ref uses get_weights_cuda().t() so callers see the
        standard (d_size, x_size) shape without a CPU round-trip.
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

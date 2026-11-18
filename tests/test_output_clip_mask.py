# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for the ADC output clip mask used for the output-bound gradient."""

# pylint: disable=redefined-outer-name

import os
from itertools import product

from pytest import fixture, mark, raises, skip
from torch import Tensor, allclose, equal, ones, rand, tensor, zeros
from torch import bool as torch_bool
from torch import cuda as torch_cuda

from aihwkit.nn import AnalogConv2d, AnalogLinear
from aihwkit.simulator.configs import (
    FloatingPointRPUConfig,
    InferenceRPUConfig,
    SingleRPUConfig,
    TorchInferenceRPUConfig,
)
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.simulator.parameters.enums import (
    AnalogMVType,
    BoundManagementType,
    NoiseManagementType,
    WeightNoiseType,
)
from aihwkit.simulator.rpu_base import cuda
from aihwkit.simulator.tiles import AnalogTile

SKIP_CUDA_TESTS = os.getenv("SKIP_CUDA_TESTS") or not cuda.is_compiled()
MASK_BINDING = "forward_with_output_clip_mask"


@fixture(params=["cpu", "cuda"])
def device(request):
    """Run the test on CPU and (if available) CUDA."""
    if request.param == "cuda" and (SKIP_CUDA_TESTS or not torch_cuda.is_available()):
        skip("CUDA not available")
    return request.param


def set_io_ideal(io_pars, **kwargs):
    """No noise, no quantization, unit bounds, no management."""
    io_pars.inp_noise = 0.0
    io_pars.out_noise = 0.0
    io_pars.w_noise = 0.0
    io_pars.inp_res = -1
    io_pars.out_res = -1
    io_pars.inp_bound = 1.0
    io_pars.out_bound = 1.0
    io_pars.bound_management = BoundManagementType.NONE
    io_pars.noise_management = NoiseManagementType.NONE
    for key, value in kwargs.items():
        setattr(io_pars, key, value)


def single_config(**kwargs):
    """Deterministic single device config with an ideal backward."""
    rpu_config = SingleRPUConfig(
        device=ConstantStepDevice(w_max=10.0, w_min=-10.0, w_max_dtod=0.0, w_min_dtod=0.0)
    )
    set_io_ideal(rpu_config.forward, **kwargs)
    set_io_ideal(rpu_config.backward, is_perfect=True)
    return rpu_config


def inference_config(**kwargs):
    """Deterministic inference config with an ideal backward."""
    rpu_config = InferenceRPUConfig()
    rpu_config.forward.w_noise_type = WeightNoiseType.NONE
    set_io_ideal(rpu_config.forward, **kwargs)
    set_io_ideal(rpu_config.backward, is_perfect=True)
    return rpu_config


def native_tile(weights, device, **kwargs):
    """Return the native tile (``rpu_base``) with the given weights."""
    weights = tensor(weights)
    analog_tile = AnalogTile(
        weights.shape[0], weights.shape[1], single_config(**kwargs), bias=False
    )
    analog_tile.set_weights(weights)
    if device == "cuda":
        analog_tile = analog_tile.cuda()
    return analog_tile.tile


def linear_layer(weights, device, rpu_config):
    """Return an analog linear layer without bias."""
    weights = tensor(weights)
    layer = AnalogLinear(weights.shape[1], weights.shape[0], bias=False, rpu_config=rpu_config)
    layer.set_weights(weights)
    return layer.to(device)


def forward_backward(layer, x_values):
    """Return output and input gradient for ``loss = sum(y)``."""
    x_input = tensor(x_values, device=layer.analog_module.device, requires_grad=True)
    y_output = layer(x_input)
    y_output.sum().backward()
    return y_output.detach().cpu(), x_input.grad.cpu()


def to_trans(values: Tensor, size: int) -> Tensor:
    """``[*, size]`` to the transposed native layout ``[size, *]``."""
    rest = values.shape[:-1]
    return values.reshape(-1, size).t().contiguous().reshape(size, *rest)


# Binding: forward_with_output_clip_mask


@mark.parametrize(
    "kwargs,x_values,expected_y,expected_mask",
    [
        # real clipping (positive and negative); exactly at the bound is not clipped
        ({}, [[1.5, -1.5], [1.0, -1.0]], [[1.0, -1.0], [1.0, -1.0]], [[1, 1], [0, 0]]),
        # bound management recovers the value: the ADC did not clip
        (
            {"bound_management": BoundManagementType.ITERATIVE},
            [[1.5, -1.5]],
            [[1.5, -1.5]],
            [[0, 0]],
        ),
        # perfect forward and ideal mv type do not use the ADC
        ({"is_perfect": True}, [[1.5, -1.5]], [[1.5, -1.5]], [[0, 0]]),
        ({"mv_type": AnalogMVType.IDEAL}, [[1.5, -1.5]], [[1.5, -1.5]], [[0, 0]]),
    ],
)
def test_binding_mask(device, kwargs, x_values, expected_y, expected_mask):
    """Mask marks outputs clipped by the ADC (identity weights, inp_bound 2)."""
    tile = native_tile([[1.0, 0.0], [0.0, 1.0]], device, inp_bound=2.0, **kwargs)
    x_input = tensor(x_values, device=device)
    y_output, mask = tile.forward_with_output_clip_mask(x_input)
    assert mask.dtype == torch_bool
    assert mask.device == y_output.device == x_input.device
    assert allclose(y_output.cpu(), tensor(expected_y))
    assert equal(mask.cpu(), tensor(expected_mask).bool())
    assert allclose(tile.forward(x_input).cpu(), y_output.cpu())


@mark.parametrize("batch_dims", [(), (4,), (2, 5)])
@mark.parametrize("x_trans,d_trans", list(product([False, True], repeat=2)))
@mark.parametrize("is_perfect", [False, True])
def test_binding_layout(device, batch_dims, x_trans, d_trans, is_perfect):
    """Output and mask have the native layout for all transposition flags."""
    weights = tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])  # d_size 3, x_size 2
    tile = native_tile(weights.tolist(), device, out_bound=0.4, is_perfect=is_perfect)
    x_values = rand(*batch_dims, 2) * 1.6 - 0.8
    y_values = x_values @ weights.t()

    expected_y = y_values if is_perfect else y_values.clamp(-0.4, 0.4)
    expected_mask = zeros(y_values.shape).bool() if is_perfect else y_values.abs() > 0.4
    x_input = to_trans(x_values, 2) if x_trans else x_values
    if d_trans:
        expected_y, expected_mask = to_trans(expected_y, 3), to_trans(expected_mask, 3)

    y_output, mask = tile.forward_with_output_clip_mask(x_input.to(device), False, x_trans, d_trans)
    assert y_output.shape == mask.shape == expected_y.shape
    assert allclose(y_output.cpu(), expected_y, atol=1e-6)
    assert equal(mask.cpu(), expected_mask)


def test_binding_non_blocking(device):
    """``non_blocking`` returns the same output and mask after synchronizing."""
    tile = native_tile([[1.0, 0.0], [0.0, 1.0]], device, inp_bound=2.0)
    x_input = tensor([[1.5, -0.5], [0.5, -1.5]], device=device)
    y_output, mask = tile.forward_with_output_clip_mask(x_input, False, False, False, False, True)
    if device == "cuda":
        torch_cuda.synchronize()
    assert allclose(y_output.cpu(), tensor([[1.0, -0.5], [0.5, -1.0]]))
    assert equal(mask.cpu(), tensor([[True, False], [False, True]]))


@mark.parametrize("x_trans", [False, True])
def test_binding_wrong_input_size(device, x_trans):
    """A wrong feature size (also zero) is rejected before the native call."""
    tile = native_tile([[1.0, 0.0], [0.0, 1.0]], device)
    for size in [0, 3]:
        shape = (size, 4) if x_trans else (4, size)
        with raises(RuntimeError, match="Invalid x_input dimensions"):
            tile.forward_with_output_clip_mask(
                zeros(*shape, device=device), False, x_trans, x_trans
            )


def test_binding_unsupported_mv_type(device):
    """Multi-pass forward types are rejected by the native entry point."""
    tile = native_tile([[1.0, 0.0], [0.0, 1.0]], device, mv_type=AnalogMVType.POS_NEG_SEPARATE)
    with raises(RuntimeError, match="one-pass"):
        tile.forward_with_output_clip_mask(ones(1, 2, device=device))
    # the regular forward still works afterwards
    assert allclose(tile.forward(ones(1, 2, device=device)).cpu(), ones(1, 2))


def test_binding_new_mask_per_call(device):
    """Each call returns a new, fully initialized mask."""
    tile = native_tile([[1.0, 0.0], [0.0, 1.0]], device, inp_bound=2.0)
    _, first = tile.forward_with_output_clip_mask(tensor([[1.5, -1.5]], device=device))
    _, second = tile.forward_with_output_clip_mask(zeros(1, 2, device=device))
    assert first.data_ptr() != second.data_ptr()
    assert equal(first.cpu(), tensor([[True, True]]))
    assert not second.any()


@mark.parametrize(
    "max_bm_factor,expected_y,expected_mask",
    [
        # needs several bound management rounds (factor 4): finally not clipped
        (1000, [[3.0]], [[False]]),
        # stops at max_bm_factor while still clipped (output 2 * clamped 1)
        (1, [[2.0]], [[True]]),
    ],
)
def test_binding_mask_of_final_bm_round(device, max_bm_factor, expected_y, expected_mask):
    """The mask of the last bound management round is returned."""
    weights = [[1.5, 1.5]] if max_bm_factor > 1 else [[2.0, 2.0]]
    tile = native_tile(
        weights, device, bound_management=BoundManagementType.ITERATIVE, max_bm_factor=max_bm_factor
    )
    y_output, mask = tile.forward_with_output_clip_mask(ones(1, 2, device=device))
    assert allclose(y_output.cpu(), tensor(expected_y))
    assert equal(mask.cpu(), tensor(expected_mask))


@mark.parametrize(
    "weight,max_bm_factor,expected",
    [
        # stops at max_bm_factor: rows 1 and 3 still clipped, row 2 never clipped
        (4.0, 1, ([[2.0], [0.8], [-2.0]], [[True], [False], [True]])),
        # all rows recovered by bound management
        (1.5, 1000, ([[3.0], [0.3], [-3.0]], [[False], [False], [False]])),
    ],
)
@mark.parametrize("trans", list(product([False, True], repeat=2)))
def test_binding_mask_of_final_bm_round_batch(device, weight, max_bm_factor, expected, trans):
    """Batched bound management keeps the per-row mask of the last round."""
    expected_y, expected_mask = expected
    x_trans, d_trans = trans
    tile = native_tile(
        [[weight, weight]],
        device,
        bound_management=BoundManagementType.ITERATIVE,
        max_bm_factor=max_bm_factor,
    )
    x_input = tensor([[1.0, 1.0], [0.1, 0.1], [-1.0, -1.0]])
    expected_y, expected_mask = tensor(expected_y), tensor(expected_mask)
    if x_trans:
        x_input = to_trans(x_input, 2)
    if d_trans:
        expected_y, expected_mask = to_trans(expected_y, 1), to_trans(expected_mask, 1)
    y_output, mask = tile.forward_with_output_clip_mask(x_input.to(device), False, x_trans, d_trans)
    assert allclose(y_output.cpu(), expected_y)
    assert equal(mask.cpu(), expected_mask)


@mark.parametrize(
    "out_res,x_values,expected_y,expected_mask",
    [
        # step 0.5: raw +-1.1 is rounded to +-1.0 (= bound), not clipped
        (0.25, [[1.1, -1.1]], [[1.0, -1.0]], [[False, False]]),
        # step 0.6: raw +-0.95 is rounded to +-1.2 and then clipped to +-1.0
        (0.3, [[0.95, -0.95]], [[1.0, -1.0]], [[True, True]]),
    ],
)
def test_binding_mask_after_output_quantization(
    device, out_res, x_values, expected_y, expected_mask
):
    """The mask is taken after the output quantization and before the clamp."""
    tile = native_tile([[1.0, 0.0], [0.0, 1.0]], device, inp_bound=2.0, out_res=out_res)
    y_output, mask = tile.forward_with_output_clip_mask(tensor(x_values, device=device))
    assert allclose(y_output.cpu(), tensor(expected_y))
    assert equal(mask.cpu(), tensor(expected_mask))


@mark.parametrize(
    "weights,x_value,expected_y,expected_mask",
    [
        # x=2 is scaled to 1: ADC value 0.8, restored output 1.6 > bound, not clipped
        ([[0.4, 0.4]], 2.0, [[1.6]], [[False]]),
        # x=2 is scaled to 1: ADC value 2 is clipped to 1, restored output 2
        ([[1.0, 1.0]], 2.0, [[2.0]], [[True]]),
        # x=0.5 is scaled to 1: ADC value 2 is clipped to 1, restored output 0.5 < bound
        ([[1.0, 1.0]], 0.5, [[0.5]], [[True]]),
    ],
)
def test_binding_noise_management(device, weights, x_value, expected_y, expected_mask):
    """The mask is taken before the noise management rescaling."""
    tile = native_tile(weights, device, noise_management=NoiseManagementType.ABS_MAX)
    x_input = tensor([[x_value, x_value]], device=device)
    y_output, mask = tile.forward_with_output_clip_mask(x_input)
    assert allclose(y_output.cpu(), tensor(expected_y))
    assert equal(mask.cpu(), tensor(expected_mask))


def test_binding_zero_input_noise_management(device):
    """Zero input with noise management and without output noise is not clipped."""
    tile = native_tile([[1.0, 1.0]], device, noise_management=NoiseManagementType.ABS_MAX)
    y_output, mask = tile.forward_with_output_clip_mask(zeros(8, 2, device=device))
    assert not y_output.any()
    assert not mask.any()


def test_binding_zero_input_noisy_output(device):
    """Zero input with noise management and large output noise (backend specific).

    The CPU skips the ADC for an all-zero input (noise management
    short-cut), so the mask is false. CUDA computes the output noise
    and clipping and sets the output to zero afterwards, so the mask
    reports the actual (noise induced) ADC clipping. Final outputs are
    zero on both. With ``out_noise=10`` and ``out_bound=1`` about 92%
    of the outputs are clipped on CUDA, so the mask is not all false.
    """
    tile = native_tile(
        [[1.0, 1.0]] * 4, device, noise_management=NoiseManagementType.ABS_MAX, out_noise=10.0
    )
    y_output, mask = tile.forward_with_output_clip_mask(zeros(64, 2, device=device))
    assert not y_output.any()
    if device == "cpu":
        assert not mask.any()
    else:
        assert mask.any()


# Layer: output-bound gradient


@mark.parametrize(
    "config,kwargs,expected",
    [
        (single_config, {}, True),
        (single_config, {"is_perfect": True}, True),
        (single_config, {"mv_type": AnalogMVType.IDEAL}, True),
        (inference_config, {}, True),
        (single_config, {"mv_type": AnalogMVType.POS_NEG_SEPARATE}, False),
        (single_config, {"mv_type": AnalogMVType.POS_NEG_SEPARATE_DIGITAL_SUM}, False),
        (single_config, {"mv_type": AnalogMVType.POS_NEG_SEPARATE, "is_perfect": True}, True),
    ],
)
def test_supports_output_clip_mask(device, config, kwargs, expected):
    """Native clip mask is used for perfect and one-pass forward only."""
    layer = linear_layer([[1.0, 1.0]], device, config(**kwargs))
    assert layer.analog_module.supports_output_clip_mask() == expected


@mark.parametrize(
    "config,kwargs,expected_y,expected_dx",
    [
        (single_config, {"bound_management": BoundManagementType.ITERATIVE}, 2.0, 1.0),
        (single_config, {"is_perfect": True}, 2.0, 1.0),
        (single_config, {"mv_type": AnalogMVType.IDEAL}, 2.0, 1.0),
        (single_config, {}, 1.0, 0.0),  # really clipped
        (single_config, {"out_bound": 2.0}, 2.0, 1.0),  # exactly at the bound
        (inference_config, {"bound_management": BoundManagementType.ITERATIVE}, 2.0, 1.0),
        (inference_config, {"is_perfect": True}, 2.0, 1.0),
        (inference_config, {}, 1.0, 0.0),
    ],
)
def test_layer_gradient(device, config, kwargs, expected_y, expected_dx):
    """The input gradient is only masked where the ADC clipped (W=[1,1], x=[1,1])."""
    layer = linear_layer([[1.0, 1.0]], device, config(**kwargs))
    y_output, x_grad = forward_backward(layer, [[1.0, 1.0]])
    assert allclose(y_output, tensor([[expected_y]]))
    assert allclose(x_grad, tensor([[expected_dx, expected_dx]]))


def test_layer_gradient_noise_management(device):
    """Output restored by noise management above the bound keeps the gradient."""
    rpu_config = single_config(noise_management=NoiseManagementType.ABS_MAX)
    layer = linear_layer([[0.4, 0.4]], device, rpu_config)
    y_output, x_grad = forward_backward(layer, [[2.0, 2.0]])
    assert allclose(y_output, tensor([[1.6]]))
    assert allclose(x_grad, tensor([[0.4, 0.4]]))


def test_layer_gradient_noise_management_clipped_below_bound(device):
    """Output clipped at the ADC but rescaled below the bound is masked.

    ``x=[0.5, 0.5]`` is scaled to ``[1, 1]`` by noise management, the ADC
    value 2 is clipped to 1 and restored to 0.5. The previous mask on the
    final output (``0.5 < 0.999 * bound``) did not mask this gradient.
    """
    rpu_config = single_config(noise_management=NoiseManagementType.ABS_MAX)
    layer = linear_layer([[1.0, 1.0]], device, rpu_config)
    y_output, x_grad = forward_backward(layer, [[0.5, 0.5]])
    assert allclose(y_output, tensor([[0.5]]))
    assert allclose(x_grad, zeros(1, 2))


def test_layer_gradient_analog_bias(device):
    """Analog bias column: managed output is not masked."""
    rpu_config = single_config(bound_management=BoundManagementType.ITERATIVE)
    rpu_config.mapping.digital_bias = False
    layer = AnalogLinear(2, 1, bias=True, rpu_config=rpu_config)
    layer.set_weights(tensor([[1.0, 1.0]]), tensor([1.0]))
    layer = layer.to(device)
    y_output, x_grad = forward_backward(layer, [[1.0, 1.0]])
    assert allclose(y_output, tensor([[3.0]]))
    assert allclose(x_grad, tensor([[1.0, 1.0]]))


def test_layer_gradient_independent_forwards(device):
    """Two forwards before a single backward keep their own masks."""
    layer = linear_layer([[1.0, 1.0]], device, single_config())
    x_clip = tensor([[1.0, 1.0]], device=device, requires_grad=True)
    x_free = tensor([[0.2, 0.2]], device=device, requires_grad=True)
    (layer(x_clip).sum() + layer(x_free).sum()).backward()
    assert allclose(x_clip.grad.cpu(), zeros(1, 2))
    assert allclose(x_free.grad.cpu(), ones(1, 2))


def test_layer_gradient_mapping_scale(device):
    """Digital mapping scale: the mask refers to the tile output (before scaling)."""
    rpu_config = single_config(bound_management=BoundManagementType.ITERATIVE)
    rpu_config.mapping.weight_scaling_omega = 1.0  # tile weights [1, 1], scale 2
    layer = linear_layer([[2.0, 2.0]], device, rpu_config)
    y_output, x_grad = forward_backward(layer, [[1.0, 1.0]])
    assert allclose(y_output, tensor([[4.0]]))
    assert allclose(x_grad, tensor([[2.0, 2.0]]))


def test_layer_gradient_without_output_bound_handling(device):
    """Without ``handle_output_bound`` no output gradient mask is applied."""
    layer = linear_layer([[1.0, 1.0]], device, single_config())
    layer.analog_module.handle_output_bound = False
    y_output, x_grad = forward_backward(layer, [[1.0, 1.0]])
    assert allclose(y_output, tensor([[1.0]]))  # clipped
    assert allclose(x_grad, ones(1, 2))


def test_layer_gradient_output_for_update_not_masked(device):
    """The mask only applies to the input gradient, not to the stored update gradient."""
    layer = linear_layer([[1.0, 1.0]], device, single_config())
    _, x_grad = forward_backward(layer, [[1.0, 1.0]])
    assert allclose(x_grad, zeros(1, 2))  # clipped
    grad_output = layer.analog_module.analog_ctx.analog_grad_output[-1]
    assert allclose(grad_output.cpu(), ones(1, 1))


# Layer: fallback to the previous 0.999 * bound mask


def test_fallback_multi_pass(device):
    """Multi-pass forward keeps the previous mask on the final output."""
    rpu_config = single_config(
        mv_type=AnalogMVType.POS_NEG_SEPARATE, noise_management=NoiseManagementType.ABS_MAX
    )
    layer = linear_layer([[0.4, 0.4]], device, rpu_config)
    y_output, x_grad = forward_backward(layer, [[2.0, 2.0]])
    assert allclose(y_output, tensor([[1.6]]))
    assert allclose(x_grad, zeros(1, 2))  # 1.6 >= 0.999 * bound


@mark.parametrize("replacement", ["delete", "none"])
def test_fallback_without_binding(device, monkeypatch, replacement):
    """A native tile without a callable clip mask binding uses the previous mask."""
    rpu_config = single_config(bound_management=BoundManagementType.ITERATIVE)
    layer = linear_layer([[1.0, 1.0]], device, rpu_config)
    for cls in type(layer.analog_module.tile).__mro__:
        if MASK_BINDING in cls.__dict__:
            if replacement == "delete":
                monkeypatch.delattr(cls, MASK_BINDING)
            else:
                monkeypatch.setattr(cls, MASK_BINDING, None)

    assert not layer.analog_module.supports_output_clip_mask()
    y_output, x_grad = forward_backward(layer, [[1.0, 1.0]])
    assert allclose(y_output, tensor([[2.0]]))
    assert allclose(x_grad, zeros(1, 2))  # previous behavior: 2 >= 0.999 * bound


def test_binding_error_is_not_hidden(device, monkeypatch):
    """Errors of the clip mask binding are raised, not turned into a fallback."""

    def failing_forward(*_, **__):
        raise RuntimeError("clip mask failure")

    layer = linear_layer([[1.0, 1.0]], device, single_config())
    monkeypatch.setattr(type(layer.analog_module.tile), MASK_BINDING, failing_forward)
    with raises(RuntimeError, match="clip mask failure"):
        forward_backward(layer, [[1.0, 1.0]])


@mark.parametrize("rpu_config", [FloatingPointRPUConfig(), TorchInferenceRPUConfig()])
def test_not_supported_tiles(rpu_config):
    """Floating point and torch tiles do not use the native clip mask."""
    layer = linear_layer([[1.0, 1.0]], "cpu", rpu_config)
    assert not layer.analog_module.supports_output_clip_mask()


@mark.parametrize("use_indexed", [None, False, True])
def test_convolution(device, use_indexed):
    """Unfolded convolution uses the native mask, indexed convolution the previous one.

    With ``use_indexed=None`` convolutions are indexed on CUDA and unfolded on CPU.
    """
    if device == "cpu" and use_indexed:
        skip("indexed convolution is only used on CUDA")
    rpu_config = single_config(bound_management=BoundManagementType.ITERATIVE)
    layer = AnalogConv2d(
        1, 1, kernel_size=1, bias=False, rpu_config=rpu_config, use_indexed=use_indexed
    )
    layer.set_weights(tensor([[2.0]]))
    layer = layer.to(device)
    x_input = ones(1, 1, 2, 2, device=device, requires_grad=True)
    y_output = layer(x_input)
    y_output.sum().backward()

    indexed = use_indexed or (use_indexed is None and device == "cuda")
    assert allclose(y_output.detach().cpu(), 2.0 * ones(1, 1, 2, 2))
    assert allclose(x_input.grad.cpu(), (0.0 if indexed else 2.0) * ones(1, 1, 2, 2))

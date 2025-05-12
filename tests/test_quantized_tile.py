# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for quantized tile functionality."""

from typing import Union

from pytest import mark
import torch

from aihwkit.nn import AnalogLinear
from aihwkit.simulator.digital_low_precision.range_estimators import RangeEstimators
from aihwkit.simulator.parameters.quantization import ActivationQuantConfig

from .helpers.tiles import QuantizedTorchInferenceRPUConfig, TorchInferenceRPUConfig


@mark.parametrize("n_bits", [0, 4, 8])
@mark.parametrize("symmetric", [True, False])
@mark.parametrize("range_estimator", list(RangeEstimators))
def test_output_quantization(n_bits, symmetric, range_estimator):
    """Test that output quantization works, returning the appropriate number of states"""

    def set_perfect_rpuconfig(
        rpu_config: Union[TorchInferenceRPUConfig, QuantizedTorchInferenceRPUConfig]
    ):
        rpu_config.forward.is_perfect = True
        if isinstance(rpu_config, QuantizedTorchInferenceRPUConfig):
            rpu_config.act_quant_config = ActivationQuantConfig(
                n_bits=n_bits, symmetric=symmetric, range_estimator=range_estimator
            )
        return rpu_config

    quant_rpu_config = set_perfect_rpuconfig(QuantizedTorchInferenceRPUConfig())
    rpu_config_ref = set_perfect_rpuconfig(TorchInferenceRPUConfig())

    in_d = 1
    out_d = 2 ** (n_bits + 1)
    linear_ref = AnalogLinear(
        in_features=in_d, out_features=out_d, bias=False, rpu_config=rpu_config_ref
    )
    linear_ref.analog_module.tile.weight.data = torch.arange(0, out_d, dtype=torch.float).view(
        out_d, 1
    )
    linear = AnalogLinear(
        in_features=in_d, out_features=out_d, bias=False, rpu_config=quant_rpu_config
    )
    linear.load_state_dict(linear_ref.state_dict(), strict=False, load_rpu_config=False)
    inp = torch.ones((1, in_d))
    out = linear(inp)
    out_ref = linear_ref(inp)
    if n_bits == 0:
        assert torch.allclose(out, out_ref, atol=1e-5)
    else:
        assert not torch.allclose(out, out_ref, atol=1e-5)
        assert out.unique().numel() <= 2**n_bits


@mark.parametrize("arr_rows", [512, 1024])
@mark.parametrize("arr_columns", [512, 1024])
@mark.parametrize("n_bits", [0, 4, 8])
@mark.parametrize("symmetric", [True, False])
@mark.parametrize("range_estimator", list(RangeEstimators))
def test_array_module_output_quantization(
    n_bits, symmetric, range_estimator, arr_rows, arr_columns
):
    """Test that when an array is used, output quantization is properly applied"""

    def set_perfect_rpuconfig(
        rpu_config: Union[TorchInferenceRPUConfig, QuantizedTorchInferenceRPUConfig]
    ):
        rpu_config.forward.is_perfect = True
        if isinstance(rpu_config, QuantizedTorchInferenceRPUConfig):
            rpu_config.act_quant_config = ActivationQuantConfig(
                n_bits=n_bits, symmetric=symmetric, range_estimator=range_estimator
            )
        return rpu_config

    quant_rpu_config = set_perfect_rpuconfig(QuantizedTorchInferenceRPUConfig())
    rpu_config_ref = set_perfect_rpuconfig(TorchInferenceRPUConfig())

    linear_ref = AnalogLinear(
        in_features=arr_rows, out_features=arr_columns, bias=True, rpu_config=rpu_config_ref
    )
    linear = AnalogLinear(
        in_features=arr_rows, out_features=arr_columns, bias=True, rpu_config=quant_rpu_config
    )
    linear.load_state_dict(linear_ref.state_dict(), strict=False, load_rpu_config=False)
    inp = torch.ones((1, arr_rows))
    out = linear(inp)
    out_ref = linear_ref(inp)
    if n_bits == 0:
        assert torch.allclose(out, out_ref, atol=1e-5)
    else:
        assert not torch.allclose(out, out_ref, atol=1e-5)
        assert out.unique().numel() <= 2**n_bits


@mark.parametrize("arr_rows", [512, 1024])
@mark.parametrize("arr_columns", [512, 1024])
@mark.parametrize("n_bits", [0, 4, 8])
@mark.parametrize("symmetric", [True, False])
def test_quantized_periphery(n_bits, symmetric, arr_rows, arr_columns):
    """Test that quantized periphery is properly applied"""

    def set_perfect_rpuconfig_with_periphery(
        rpu_config: Union[TorchInferenceRPUConfig, QuantizedTorchInferenceRPUConfig]
    ):
        rpu_config.forward.is_perfect = True
        rpu_config.mapping.weight_scaling_omega = 1.0
        rpu_config.mapping.weight_scaling_columnwise = True
        if isinstance(rpu_config, QuantizedTorchInferenceRPUConfig):
            rpu_config.pre_post.periph_quant.n_bits = n_bits
            rpu_config.pre_post.periph_quant.symmetric = symmetric
        return rpu_config

    quant_rpu_config = set_perfect_rpuconfig_with_periphery(QuantizedTorchInferenceRPUConfig())
    rpu_config_ref = set_perfect_rpuconfig_with_periphery(TorchInferenceRPUConfig())

    linear_ref = AnalogLinear(
        in_features=arr_rows, out_features=arr_columns, bias=True, rpu_config=rpu_config_ref
    )
    linear = AnalogLinear(
        in_features=arr_rows, out_features=arr_columns, bias=True, rpu_config=quant_rpu_config
    )
    linear.load_state_dict(linear_ref.state_dict(), strict=False, load_rpu_config=False)
    inp = torch.ones((1, arr_rows))
    out = linear(inp)
    out_ref = linear_ref(inp)
    if n_bits == 0:
        assert torch.allclose(out, out_ref, atol=1e-5)
    else:
        assert not torch.allclose(out, out_ref, atol=1e-5)

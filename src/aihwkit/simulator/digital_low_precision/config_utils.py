# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# mypy: disable-error-code=attr-defined

"""Defines configuration parameters and conversions to dict
structures for the quantized module base classes"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

from aihwkit.simulator.digital_low_precision.quantizers import QMethods
from aihwkit.simulator.digital_low_precision.range_estimators import OptMethod, RangeEstimators

if TYPE_CHECKING:
    from aihwkit.simulator.parameters.quantization import (
        ActivationQuantConfig,
        QuantizationConfig,
        WeightQuantConfig,
    )


@dataclass
class CurrentMinMaxEstimatorParams:
    """Parameters for the estimator `RangeEstimators.current_minmax`"""

    percentile: Optional[float] = None


@dataclass
class RunningMinMaxEstimatorParams:
    """Parameters for the estimator `RangeEstimators.running_minmax`"""

    momentum: float = 0.9


@dataclass
class MSEEstimatorParams:
    """Parameters for the estimator `RangeEstimators.MSE`"""

    range_opt_method: OptMethod = OptMethod.golden_section
    num_candidates: int = 100
    range_margin: float = 0.5


@dataclass
class CrossEntropyEstimatorParams(MSEEstimatorParams):
    """Parameters for the estimator `RangeEstimators.cross_entropy`.
    Alias of `MSEEstimatorParams`"""


def convert_configs_to_kwargs_dict(quant_config: "QuantizationConfig") -> Dict[str, Any]:
    """Converts the QuantizationConfig structure to a kwargs dict for the
    `QuantizedModule` base class"""
    return {
        **convert_weight_config_to_kwargs_dict(quant_config.weight_quant),
        **convert_act_config_to_kwargs_dict(quant_config.activation_quant),
    }


def convert_weight_config_to_kwargs_dict(
    weight_quant_config: "WeightQuantConfig",
) -> Dict[str, Any]:
    """Converts the WeightQuantConfig structure to a kwargs dict for the
    `QuantizedModule` base class"""
    weight_range_options = {}

    range_estim_params = weight_quant_config.range_estimator_params
    if weight_quant_config.range_estimator == RangeEstimators.running_minmax:
        weight_range_options["momentum"] = range_estim_params.momentum

    elif weight_quant_config.range_estimator in [
        RangeEstimators.MSE,
        RangeEstimators.cross_entropy,
    ]:
        weight_range_options["opt_method"] = range_estim_params.range_opt_method
        weight_range_options["num_candidates"] = range_estim_params.num_candidates
        weight_range_options["range_margin"] = range_estim_params.range_margin

    return {
        "method": QMethods[
            "symmetric_uniform" if weight_quant_config.symmetric else "asymmetric_uniform"
        ],
        "n_bits": weight_quant_config.n_bits,
        "per_channel_weights": weight_quant_config.per_channel,
        "percentile": (
            range_estim_params.percentile
            if weight_quant_config.range_estimator == RangeEstimators.current_minmax
            else None
        ),
        "weight_range_method": weight_quant_config.range_estimator,
        "weight_range_options": weight_range_options,
    }


def convert_act_config_to_kwargs_dict(act_quant_config: "ActivationQuantConfig") -> Dict[str, Any]:
    """Converts the ActivationQuantConfig structure to a kwargs dict for the
    `QuantizedModule` base class"""
    act_range_options = {}

    range_estim_params = act_quant_config.range_estimator_params
    if act_quant_config.range_estimator == RangeEstimators.current_minmax:
        act_range_options["percentile"] = range_estim_params.percentile
    elif act_quant_config.range_estimator == RangeEstimators.running_minmax:
        act_range_options["momentum"] = range_estim_params.momentum

    elif act_quant_config.range_estimator in [RangeEstimators.MSE, RangeEstimators.cross_entropy]:
        act_range_options["opt_method"] = range_estim_params.range_opt_method
        act_range_options["num_candidates"] = range_estim_params.num_candidates
        act_range_options["range_margin"] = range_estim_params.range_margin

    return {
        "act_method": QMethods[
            "symmetric_uniform" if act_quant_config.symmetric else "asymmetric_uniform"
        ],
        "n_bits_act": act_quant_config.n_bits,
        "act_range_method": act_quant_config.range_estimator,
        "act_range_options": act_range_options,
    }

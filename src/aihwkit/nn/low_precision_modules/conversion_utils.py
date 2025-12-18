# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

# mypy: disable-error-code=attr-defined

"""Utility funcitons for converting a module to its quantized counterpart"""

from copy import deepcopy
from typing import Any, Dict, Optional

from torch.nn import Conv2d, Embedding, LayerNorm, Linear, Module

from aihwkit.nn.low_precision_modules.quantized_base_modules import (
    QuantConv2d,
    QuantEmbedding,
    QuantLayerNorm,
    QuantLinear,
)
from aihwkit.simulator.digital_low_precision.quantizers import QuantizerBase
from aihwkit.simulator.digital_low_precision.range_estimators import RangeEstimatorBase
from aihwkit.simulator.parameters.quantization import QuantizationMap, QuantizedModuleConfig

DEFAULT_CONVERSIONS = {
    Linear: QuantLinear,
    LayerNorm: QuantLayerNorm,
    Embedding: QuantEmbedding,
    Conv2d: QuantConv2d,
}
LEAF_MODULES = (RangeEstimatorBase, QuantizerBase)


def append_default_conversions(quantization_map: QuantizationMap) -> None:
    """
    Appends the default conversions defined in the `DEFAULT_CONVERSIONS`
    dictionary in the `QuantizationMap` datastructure. If a conversion
    for a specific layer is already defined in the datastructure, it skips it.

    As for the conversion's `QuantizationConfig`, it utilizes the default
    one defined in the quantization_map.default_qconfig field.

    Parameters
    ----------
    quantization_map : QuantizationMap
        The QuantizationMap instance to append the default conversions
    """
    for module, q_module in DEFAULT_CONVERSIONS.items():
        if module in quantization_map.module_qconfig_map:
            continue

        quantization_map.module_qconfig_map[module] = QuantizedModuleConfig(
            quantized_module=q_module, module_qconfig=deepcopy(quantization_map.default_qconfig)
        )


def get_module_args(module: Module, activation: Optional[Module] = None) -> dict:
    """
    Get the arguments from a pytorch module to provide it to the
    initialization function of the quantized modules. The way
    to retrieve the arguments for each type of module are defined
    with functions defined inside this functions, with the convention
    `get_{module_type}_args`

    Parameters
    ----------
    module : Module
        The module to extract the arguments from
    activation : Optional[Module], optional
        The activation function for the `QuantizationHijacker` if applicable,
        by default None

    Raises
    ------
    ValueError
        If the function has not been tought how to handle
        a given module.
    """

    def get_linear_args(module: Module) -> Dict[str, Any]:
        """Quantization arguments for `QuantLinear`"""
        args = dict(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
        )
        return args

    def get_layernorm_args(module: Module) -> Dict[str, Any]:
        """Quantization arguments for `QuantLayerNorm`"""
        args = dict(normalized_shape=module.normalized_shape, eps=module.eps)
        return args

    def get_embedding_args(module: Module) -> Dict[str, Any]:
        """Quantization arguments for `QuantEmbeddings`"""
        args = dict(
            num_embeddings=module.num_embeddings,
            embedding_dim=module.embedding_dim,
            padding_idx=module.padding_idx,
            max_norm=module.max_norm,
            norm_type=module.norm_type,
            scale_grad_by_freq=module.scale_grad_by_freq,
            sparse=module.sparse,
        )
        return args

    def get_conv2d_args(module: Module) -> Dict[str, Any]:
        """Quantization arguments for `QuantConv2d`"""
        args = dict(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
        )
        return args

    if isinstance(module, Linear):
        kwargs = get_linear_args(module)
    elif isinstance(module, LayerNorm):
        kwargs = get_layernorm_args(module)
    elif isinstance(module, Embedding):
        kwargs = get_embedding_args(module)
    elif isinstance(module, Conv2d):
        kwargs = get_conv2d_args(module)
    else:
        raise ValueError

    kwargs["activation"] = activation
    return kwargs

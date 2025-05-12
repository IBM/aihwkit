# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

# mypy: disable-error-code=attr-defined

"""Basic quantized modules"""

from typing import Any, Optional
from torch import Tensor, nn
from torch.nn import functional as F

from aihwkit.simulator.digital_low_precision.base_quantized_classes import (
    FP32Acts,
    QuantizedActivation,
)
from aihwkit.simulator.digital_low_precision.base_quantized_model import QuantizedModel
from aihwkit.simulator.digital_low_precision.hijacker import QuantizationHijacker


class QuantLinear(QuantizationHijacker, nn.Linear):
    """Quantized layer of torch.nn.Linear with weight/act quantization"""

    def run_forward(
        self, x: Tensor, weight: Tensor, bias: Tensor, offsets: Optional[Any] = None
    ) -> Tensor:
        return F.linear(x.contiguous(), weight.contiguous(), bias=bias)


class QuantConv2d(QuantizationHijacker, nn.Conv2d):
    """Quantized layer of torch.nn.Conv2d with weight/act quantization"""

    def run_forward(
        self, x: Tensor, weight: Tensor, bias: Tensor, offsets: Optional[Any] = None
    ) -> Tensor:
        return F.conv2d(
            x.contiguous(),
            weight.contiguous(),
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuantLayerNorm(QuantizationHijacker, nn.LayerNorm):
    """Quantized layer of torch.nn.LayerNorm with input and weight quantization"""

    def run_forward(
        self, x: Tensor, weight: Tensor, bias: Tensor, offsets: Optional[Any] = None
    ) -> Tensor:
        return F.layer_norm(
            input=x.contiguous(),
            normalized_shape=self.normalized_shape,
            weight=weight.contiguous(),
            bias=bias.contiguous(),
            eps=self.eps,
        )


class QuantEmbedding(QuantizationHijacker, nn.Embedding):
    """Quantization of the Embedding, weight quantization.
    Note: Embedding should not quantize activations, as it is simply a lookup table,
    which is already quantized.
    """

    def __init__(self, *args: Any, activation: Optional[Any] = None, **kwargs: Any):
        super().__init__(*args, activation=activation, **kwargs)
        # NB: Embedding should not quantize activations, as it is simply a lookup table,
        # which is already quantized.
        self.activation_quantizer = FP32Acts()

    def run_forward(
        self, x: Tensor, weight: Tensor, bias: Tensor, offsets: Optional[Any] = None
    ) -> Tensor:
        return F.embedding(
            input=x.contiguous(),
            weight=weight.contiguous(),
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )


class QuantBatchNorm2d(QuantizedModel):
    """Quantization of the BatchNorm2d module. output activations are quantized."""

    def __init__(self, org_model: nn.Module, **quant_params: Any):
        super().__init__()
        self.module = org_model

        self.act_bn_quantizer = QuantizedActivation(**quant_params)

    def forward(self, x: Tensor) -> Tensor:
        """Execute BatchNorm2d and then quantize its output"""
        y = self.module(x)
        y_quant = self.act_bn_quantizer(y)
        return y_quant

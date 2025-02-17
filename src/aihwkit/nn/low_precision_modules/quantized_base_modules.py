# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

""" Basic quantized modules """

from torch import nn
from torch.nn import functional as F

from aihwkit.simulator.digital_low_precision.base_quantized_classes import (
    FP32Acts,
    QuantizedActivation,
)
from aihwkit.simulator.digital_low_precision.base_quantized_model import QuantizedModel
from aihwkit.simulator.digital_low_precision.hijacker import QuantizationHijacker


class QuantLinear(QuantizationHijacker, nn.Linear):
    """Quantized layer of torch.nn.Linear with weight/act quantization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_forward(self, x, weight, bias, offsets=None):
        return F.linear(x.contiguous(), weight.contiguous(), bias=bias)


class QuantConv2d(QuantizationHijacker, nn.Conv2d):
    """Quantized layer of torch.nn.Conv2d with weight/act quantization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_forward(self, x, weight, bias, offsets=None):
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

    def run_forward(self, x, weight, bias, offsets=None):
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

    def __init__(self, *args, activation=None, **kwargs):
        super().__init__(*args, activation=activation, **kwargs)
        # NB: Embedding should not quantize activations, as it is simply a lookup table,
        # which is already quantized.
        self.activation_quantizer = FP32Acts()

    def run_forward(self, x, weight, bias, offsets=None):
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

    def __init__(self, org_model, **quant_params):
        super().__init__()
        self.module = org_model

        self.act_bn_quantizer = QuantizedActivation(**quant_params)

    def forward(self, x):
        """Execute BatchNorm2d and then quantize its output"""
        y = self.module(x)
        y_quant = self.act_bn_quantizer(y)
        return y_quant

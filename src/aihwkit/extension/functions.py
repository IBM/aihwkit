# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=abstract-method, no-name-in-module

"""AIHWKIT Extension functions"""

from typing import Any

from torch import Tensor
from torch.autograd import Function
from torch.nn import Module
from aihwkit.extension.aihwkit_extension import ops  # pylint: disable=import-error


class FloatPrecisionCastFunction(Function):
    """Function for the float precision cast."""

    @staticmethod
    def forward(
        _: Any, input_: Tensor, exponent: int = 8, mantissa: int = 7, saturate_to_inf: bool = True
    ) -> Tensor:
        # pylint: disable=unused-argument, arguments-differ
        return ops.float_precision_cast(input_, exponent, mantissa, saturate_to_inf)

    @staticmethod
    def backward(ctx: Any, grad_in: Tensor) -> Tensor:
        # pylint: disable=unused-argument, arguments-differ
        return grad_in


class FloatPrecisionCast(Module):
    """Fake cast of FP32 numbers with variable exponent and mantissa.
    Backward pass is pass-trough

    Args:
        exponent: number of bits used for exponent
        mantissa: number of bits used for mantissa
        saturate_to_inf: whether to set it to infinity if saturated or to perform clipping

    Returns:
        Cast tensor.
    """

    def __init__(self, exponent: int = 8, mantissa: int = 7, saturate_to_inf: bool = True):
        super().__init__()
        self.exponent = exponent
        self.mantissa = mantissa
        self.saturate_to_inf = saturate_to_inf

    def forward(self, input_: Tensor) -> Tensor:
        """Fake cast with FP32 input."""
        return FloatPrecisionCastFunction.apply(
            input_, self.exponent, self.mantissa, self.saturate_to_inf
        )

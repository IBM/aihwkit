# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Low level implementation of torch-based tile."""

from typing import Union, Tuple
from numpy import ndarray

from torch import Tensor, tensor
from torch import isinf as torch_isinf

from torch.autograd.function import FunctionCtx, InplaceFunction


class UniformQuantize(InplaceFunction):
    """Quantization in-place function."""

    # pylint: disable=abstract-method, redefined-builtin, arguments-differ

    @staticmethod
    def forward(
        ctx: FunctionCtx, inp: Tensor, res: float, bound: float, stochastic: bool = False
    ) -> Tensor:
        """Quantizes the input tensor and performs straight-through estimation.

        Args:
            ctx (FunctionCtx): Context.
            inp (torch.Tensor): Input to be discretized.
            res (float): Resolution (number of states).
            bound (float): Input bounds w.r.t. which we quantize.
            stochastic (bool, optional): Stochastic rounding? Defaults to False.

        Returns:
            torch.Tensor: Quantized input.
        """
        # - Compute 1 / states if the number of states are provided
        res = 1 / res if res > 1.0 else res
        assert res > 0, "resolution is <= 0"
        # - Scale res by range
        res *= 2 * bound
        output = inp.clone()
        output = output / res
        ctx.stochastic = stochastic

        if ctx.stochastic:
            # - Stochastic rounding
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)
            output = output.round()
        else:
            # - Perform explicit rounding
            output = output.round()

        # - Scale back down
        output *= res
        return output

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tuple[Tensor, None, None, None]:
        """Straight-through estimator.

        Args:
            ctx: Context.
            grad_output: Gradient w.r.t. the inputs.

        Returns:
            Gradients w.r.t. inputs to forward.
        """
        # - Straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None


def isinf(x: Union[float, str, Tensor, ndarray]) -> Tensor:
    """Checks if the input is inf.

    Args:
        x (Union[float, str, torch.Tensor, ndarray]): Input.

    Returns:
        torch.Tensor: Boolean tensor where tensor is inf.
    """
    return torch_isinf(tensor(x))

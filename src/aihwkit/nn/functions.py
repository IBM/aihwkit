# -*- coding: utf-8 -*-

# (C) Copyright 2020 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Autograd functions for aihwkit."""

from typing import Any, Optional, Tuple

from torch import Tensor
from torch.autograd import Function

from aihwkit.simulator.tiles import FloatingPointTile


class AnalogFunction(Function):
    """Function that delegates into a `RPU` unit."""
    # pylint: disable=arguments-differ

    @staticmethod
    def forward(
            ctx: Any,
            analog_tile: FloatingPointTile,
            input_: Tensor,
            weights: Tensor,
            _: Optional[Tensor] = None) -> Tensor:
        """Execute the forward pass in the analog tile."""

        # Store in context for using during `backward()`.
        ctx.analog_tile = analog_tile
        ctx.weights = weights
        ctx.save_for_backward(input_)

        # Invoke the forward pass in the tile instance.
        return analog_tile.forward(input_)

    @staticmethod
    def backward(
            ctx: Any,
            grad_output: Tensor
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Execute the backward pass in the analog tile."""

        # Call the backward function in the tile instance.
        grad_input = ctx.analog_tile.backward(grad_output)

        # Store the parameters needed by the optimizer for `rpu.update()`.
        input_, = ctx.saved_tensors
        ctx.weights.input = input_
        ctx.weights.grad_output = grad_output

        return None, grad_input, None, None

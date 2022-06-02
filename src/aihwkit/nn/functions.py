# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
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

from torch import Tensor, empty_like
from torch.autograd import Function
from aihwkit.optim.context import AnalogContext


class AnalogFunctionBase(Function):
    """Base function for analog functions."""
    # pylint: disable=arguments-differ, protected-access, abstract-method

    @staticmethod
    def forward(
            ctx: Any,
            analog_ctx: AnalogContext,
            input_: Tensor,
            shared_weights: Optional[Tensor] = None,
            is_test: bool = False) -> Tensor:
        """Execute the forward pass in the analog tile.

        Note: Indexed versions can used when analog_ctx.use_indexed is
        set to True.
        """
        # Store in context for using during `backward()`.
        analog_tile = analog_ctx.analog_tile
        ctx.analog_ctx = analog_ctx
        ctx.shared_weights = None
        ctx.save_for_backward(input_)

        use_indexed = analog_ctx.use_indexed
        if shared_weights is not None:
            ctx.shared_weights = shared_weights
            analog_tile.ensure_shared_weights(shared_weights)
            analog_ctx.use_torch_update = True
        else:
            analog_ctx.use_torch_update = False

        # Invoke the forward pass in the tile instance.
        if use_indexed:
            return analog_tile.forward_indexed(input_, is_test)
        return analog_tile.forward(input_, is_test)

    @staticmethod
    def backward(
            ctx: Any,
            grad_output: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Execute the backward pass in the analog tile."""
        analog_ctx = ctx.analog_ctx
        analog_tile = analog_ctx.analog_tile
        input_, = ctx.saved_tensors

        shared_weights_grad = None
        use_indexed = analog_ctx.use_indexed

        if ctx.shared_weights is not None:
            analog_tile.ensure_shared_weights(ctx.shared_weights)

        # Call the backward function in the tile instance.
        if use_indexed:
            grad_input = analog_tile.backward_indexed(grad_output)
        else:
            grad_input = analog_tile.backward(grad_output)

        if analog_ctx.use_torch_update:
            # Grad computed directly (for inference training)
            shared_weights_grad = empty_like(ctx.shared_weights)
            analog_tile.set_delta_weights(shared_weights_grad)
            if use_indexed:
                analog_tile.update_indexed(input_, grad_output)
            else:
                analog_tile.update(input_, grad_output)
            analog_tile.reset_delta_weights()
        else:
            # Store activation and errors for optimizer (for analog training)
            analog_ctx.analog_input.append(input_)
            analog_ctx.analog_grad_output.append(grad_output)

        return None, grad_input, shared_weights_grad, None


class AnalogFunction(AnalogFunctionBase):
    """Function that delegates into a `RPU` unit."""
    # pylint: disable=arguments-differ, abstract-method

    @staticmethod
    def forward(
            ctx: Any,
            analog_ctx: AnalogContext,
            input_: Tensor,
            shared_weights: Optional[Tensor] = None,
            is_test: bool = False) -> Tensor:
        """Execute the forward pass in the analog tile."""
        analog_ctx.use_indexed = False
        return AnalogFunctionBase.forward(
            ctx, analog_ctx, input_, shared_weights, is_test)


class AnalogIndexedFunction(AnalogFunctionBase):
    """Function that delegates into a `RPU` unit to use the indexed forward/backward/update."""
    # pylint: disable=arguments-differ, abstract-method

    @staticmethod
    def forward(
            ctx: Any,
            analog_ctx: AnalogContext,
            input_: Tensor,
            shared_weights: Optional[Tensor] = None,
            is_test: bool = False) -> Tensor:
        """Execute the forward pass in the analog tile."""
        analog_ctx.use_indexed = True
        return AnalogFunctionBase.forward(
            ctx, analog_ctx, input_, shared_weights, is_test)

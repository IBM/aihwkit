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

""" Analog RNN layers """

from typing import Any, List, Tuple, Type, Union
from torch import Tensor, stack, jit, cat
from torch.nn import ModuleList, Module


class AnalogRNNLayer(Module):
    """Analog RNN Layer.

    Args:
        cell: RNNCell type (AnalogLSTMCell/AnalogGRUCell/AnalogVanillaRNNCell/
              AnalogLSTMCellSingleRPU)
        cell_args: arguments to RNNCell (e.g. input_size, hidden_size, rpu_configs)
    """

    # pylint: disable=abstract-method

    def __init__(self, cell: Type, *cell_args: Any):
        super().__init__()
        self.cell = cell(*cell_args)

    def get_zero_state(self, batch_size: int) -> Tensor:
        """Returns a zeroed state.

        Args:
            batch_size: batch size of the input

        Returns:
           Zeroed state tensor
        """
        return self.cell.get_zero_state(batch_size)

    def forward(
        self, input_: Tensor, state: Union[Tuple[Tensor, Tensor], Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass.

        Args:
            input_: input tensor
            state: LSTM state tensor

        Returns:
            stacked outputs and state
        """
        # pylint: disable=arguments-differ
        inputs = input_.unbind(0)
        outputs = jit.annotate(List[Tensor], [])
        for input_item in inputs:
            out, state = self.cell(input_item, state)
            outputs += [out]
        return stack(outputs), state


class AnalogReverseRNNLayer(Module):
    """Analog RNN layer for direction.

    Args:
        cell: RNNCell type (AnalogLSTMCell/AnalogGRUCell/AnalogVanillaRNNCell)
        cell_args: arguments to RNNCell (e.g. input_size, hidden_size, rpu_configs)
    """

    def __init__(self, cell: Type, *cell_args: Any):
        super().__init__()
        self.cell = cell(*cell_args)

    @staticmethod
    def reverse(lst: List[Tensor]) -> List[Tensor]:
        """Reverses the list of input tensors."""
        return lst[::-1]

    def get_zero_state(self, batch_size: int) -> Tensor:
        """Returns a zeroed state.

        Args:
            batch_size: batch size of the input

        Returns:
           Zeroed state tensor
        """
        return self.cell.get_zero_state(batch_size)

    def forward(
        self, input_: Tensor, state: Union[Tuple[Tensor, Tensor], Tensor]
    ) -> Tuple[Tensor, Union[Tuple[Tensor, Tensor], Tensor]]:
        """Forward pass.

        Args:
            input_: input tensor
            state: LSTM state tensor

        Returns:
            stacked reverse outputs and state
        """
        # pylint: disable=arguments-differ
        inputs = self.reverse(input_.unbind(0))
        outputs = jit.annotate(List[Tensor], [])
        for input_values in inputs:
            out, state = self.cell(input_values, state)
            outputs += [out]
        return stack(self.reverse(outputs)), state


class AnalogBidirRNNLayer(Module):
    """Bi-directional analog RNN layer.

    Args:
        cell: RNNCell type (AnalogLSTMCell/AnalogGRUCell/AnalogVanillaRNNCell)
        cell_args: arguments to RNNCell (e.g. input_size, hidden_size, rpu_configs)
    """

    __constants__ = ["directions"]

    def __init__(self, cell: Type, *cell_args: Any):
        super().__init__()

        self.directions = ModuleList(
            [AnalogRNNLayer(cell, *cell_args), AnalogReverseRNNLayer(cell, *cell_args)]
        )

    def get_zero_state(self, batch_size: int) -> Tensor:
        """Returns a zeroed state.

        Args:
            batch_size: batch size of the input

        Returns:
           Zeroed state tensor
        """
        return [
            self.directions[0].get_zero_state(batch_size),
            self.directions[1].get_zero_state(batch_size),
        ]

    def forward(
        self, input_: Tensor, states: List[Union[Tuple[Tensor, Tensor], Tensor]]
    ) -> Tuple[Tensor, List[Union[Tuple[Tensor, Tensor], Tensor]]]:
        """Forward pass.

        Args:
            input_: input tensor
            states: LSTM state tensor

        Returns:
            cat outputs and states
        """
        # pylint: disable=arguments-differ
        # List[RNNState]: [forward RNNState, backward RNNState]
        outputs = jit.annotate(List[Tensor], [])
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])

        for direction, state in zip(self.directions, states):
            out, out_state = direction(input_, state)
            outputs += [out]
            output_states += [out_state]

        return cat(outputs, -1), output_states

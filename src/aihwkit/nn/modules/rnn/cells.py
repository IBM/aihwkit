# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Analog cells for RNNs. """

from typing import Optional, Tuple, Type
from collections import namedtuple

from torch import Tensor, sigmoid, tanh, zeros, cat

from aihwkit.nn.modules.container import AnalogSequential
from aihwkit.nn.modules.linear import AnalogLinear

from aihwkit.simulator.configs.configs import InferenceRPUConfig
from aihwkit.simulator.parameters.base import RPUConfigBase

LSTMState = namedtuple("LSTMState", ["hx", "cx"])


class AnalogVanillaRNNCell(AnalogSequential):
    """Analog Vanilla RNN Cell.

    Args:
        input_size: in_features size for W_ih matrix
        hidden_size: in_features and out_features size for W_hh matrix
        bias: whether to use a bias row on the analog tile or not
        rpu_config: configuration for an analog resistive processing unit
        tile_module_class: Class for the analog tile module (default
            will be specified from the ``RPUConfig``).
    """

    # pylint: disable=abstract-method
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        rpu_config: Optional[RPUConfigBase] = None,
        tile_module_class: Optional[Type] = None,
    ):
        super().__init__()

        # Default to InferenceRPUConfig
        if not rpu_config:
            rpu_config = InferenceRPUConfig()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = AnalogLinear(input_size, hidden_size, bias, rpu_config, tile_module_class)
        self.weight_hh = AnalogLinear(hidden_size, hidden_size, bias, rpu_config, tile_module_class)

    def get_zero_state(self, batch_size: int) -> Tensor:
        """Returns a zeroed state.

        Args:
            batch_size: batch size of the input

        Returns:
           Zeroed state tensor
        """
        device = self.weight_ih.get_analog_tile_devices()[0]
        return zeros(batch_size, self.hidden_size, device=device)

    def forward(self, input_: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        # pylint: disable=arguments-differ
        igates = self.weight_ih(input_)
        hgates = self.weight_hh(state)

        out = tanh(igates + hgates)

        return out, out  # output will also be hidden state


class AnalogLSTMCell(AnalogSequential):
    """Analog LSTM Cell.

    Args:
        input_size: in_features size for W_ih matrix
        hidden_size: in_features and out_features size for W_hh matrix
        bias: whether to use a bias row on the analog tile or not
        rpu_config: configuration for an analog resistive processing unit
        tile_module_class: Class for the analog tile module (default
            will be specified from the ``RPUConfig``).
    """

    # pylint: disable=abstract-method

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        rpu_config: Optional[RPUConfigBase] = None,
        tile_module_class: Optional[Type] = None,
    ):
        super().__init__()

        # Default to InferenceRPUConfig
        if not rpu_config:
            rpu_config = InferenceRPUConfig()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = AnalogLinear(
            input_size, 4 * hidden_size, bias, rpu_config, tile_module_class
        )
        self.weight_hh = AnalogLinear(
            hidden_size, 4 * hidden_size, bias, rpu_config, tile_module_class
        )

    def get_zero_state(self, batch_size: int) -> Tensor:
        """Returns a zeroed state.

        Args:
            batch_size: batch size of the input

        Returns:
           Zeroed state tensor
        """
        device = self.weight_ih.get_analog_tile_devices()[0]
        return LSTMState(
            zeros(batch_size, self.hidden_size, device=device),
            zeros(batch_size, self.hidden_size, device=device),
        )

    def forward(
        self, input_: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # pylint: disable=arguments-differ
        h_x, c_x = state
        gates = self.weight_ih(input_) + self.weight_hh(h_x)
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

        in_gate = sigmoid(in_gate)
        forget_gate = sigmoid(forget_gate)
        cell_gate = tanh(cell_gate)
        out_gate = sigmoid(out_gate)

        c_y = (forget_gate * c_x) + (in_gate * cell_gate)
        h_y = out_gate * tanh(c_y)

        return h_y, (h_y, c_y)


class AnalogLSTMCellCombinedWeight(AnalogSequential):
    """Analog LSTM Cell that use a combined weight for storing gates and inputs.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: whether to use a bias row on the analog tile or not.
        rpu_config: resistive processing unit configuration.
        tile_module_class: Class for the analog tile module (default
            will be specified from the ``RPUConfig``).
    """

    # pylint: disable=abstract-method

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        rpu_config: Optional[RPUConfigBase] = None,
        tile_module_class: Optional[Type] = None,
    ):
        super().__init__()

        # Default to InferenceRPUConfig
        if not rpu_config:
            rpu_config = InferenceRPUConfig()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight = AnalogLinear(
            input_size + hidden_size, 4 * hidden_size, bias, rpu_config, tile_module_class
        )

    def get_zero_state(self, batch_size: int) -> Tensor:
        """Returns a zeroed state.

        Args:
            batch_size: batch size of the input

        Returns:
           Zeroed state tensor
        """
        device = self.weight.get_analog_tile_devices()[0]
        return LSTMState(
            zeros(batch_size, self.hidden_size, device=device),
            zeros(batch_size, self.hidden_size, device=device),
        )

    def forward(
        self, input_: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # pylint: disable=arguments-differ
        h_x, c_x = state
        x_input = cat((input_, h_x), 1)
        gates = self.weight(x_input)
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

        in_gate = sigmoid(in_gate)
        forget_gate = sigmoid(forget_gate)
        cell_gate = tanh(cell_gate)
        out_gate = sigmoid(out_gate)

        c_y = (forget_gate * c_x) + (in_gate * cell_gate)
        h_y = out_gate * tanh(c_y)

        return h_y, (h_y, c_y)


class AnalogGRUCell(AnalogSequential):
    """Analog GRU Cell.

    Args:
        input_size: in_features size for W_ih matrix
        hidden_size: in_features and out_features size for W_hh matrix
        bias: whether to use a bias row on the analog tile or not
        rpu_config: configuration for an analog resistive processing unit
        tile_module_class: Class for the analog tile module (default
            will be specified from the ``RPUConfig``).
    """

    # pylint: disable=abstract-method

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        rpu_config: Optional[RPUConfigBase] = None,
        tile_module_class: Optional[Type] = None,
    ):
        super().__init__()

        # Default to InferenceRPUConfig
        if not rpu_config:
            rpu_config = InferenceRPUConfig()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = AnalogLinear(
            input_size, 3 * hidden_size, bias, rpu_config, tile_module_class
        )
        self.weight_hh = AnalogLinear(
            hidden_size, 3 * hidden_size, bias, rpu_config, tile_module_class
        )

    def get_zero_state(self, batch_size: int) -> Tensor:
        """Returns a zeroed state.

        Args:
            batch_size: batch size of the input

        Returns:
           Zeroed state tensor
        """
        device = self.weight_ih.get_analog_tile_devices()[0]
        return zeros(batch_size, self.hidden_size, device=device)

    def forward(self, input_: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        # pylint: disable=arguments-differ

        g_i = self.weight_ih(input_)
        g_h = self.weight_hh(state)
        i_r, i_i, i_n = g_i.chunk(3, 1)
        h_r, h_i, h_n = g_h.chunk(3, 1)

        reset_gate = sigmoid(i_r + h_r)
        input_gate = sigmoid(i_i + h_i)
        new_gate = tanh(i_n + reset_gate * h_n)
        hidden_y = new_gate + input_gate * (state - new_gate)

        return hidden_y, hidden_y  # output will also be hidden state

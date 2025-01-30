# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

""" Analog RNN modules. """

import warnings
import math

from typing import Any, List, Optional, Tuple, Type, Callable
from torch import Tensor, jit
from torch.nn import Dropout, ModuleList, init, Module, Linear
from torch.autograd import no_grad

from aihwkit.nn.modules.container import AnalogContainerBase
from aihwkit.nn.modules.linear import AnalogLinear
from aihwkit.simulator.parameters.base import RPUConfigBase
from .layers import AnalogRNNLayer, AnalogBidirRNNLayer


class ModularRNN(Module):
    """Helper class to create a Modular RNN

    Args:
        num_layers: number of serially connected RNN layers
        layer: RNN layer type (e.g. AnalogLSTMLayer)
        dropout: dropout applied to output of all RNN layers except last
        first_layer_args: RNNCell type, input_size, hidden_size, rpu_config, etc.
        other_layer_args: RNNCell type, hidden_size, hidden_size, rpu_config, etc.
    """

    # pylint: disable=abstract-method

    # Necessary for iterating through self.layers and dropout support
    __constants__ = ["layers", "num_layers"]

    def __init__(
        self,
        num_layers: int,
        layer: Type,
        dropout: float,
        first_layer_args: Any,
        other_layer_args: Any,
    ):
        super().__init__()
        self.layers = self.init_stacked_analog_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )

        # Introduce a Dropout layer on the outputs of each RNN layer except
        # the last layer.
        self.num_layers = num_layers
        if num_layers == 1 and dropout > 0:
            warnings.warn(
                "dropout lstm adds dropout layers after all but last "
                "recurrent layer, it expects num_layers greater than "
                "1, but got num_layers = 1"
            )
        self.dropout_layer = Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def init_stacked_analog_lstm(
        num_layers: int, layer: Type, first_layer_args: Any, other_layer_args: Any
    ) -> ModuleList:
        """Construct a list of LSTMLayers over which to iterate.

        Args:
            num_layers: number of serially connected LSTM layers
            layer: RNN layer type (e.g. AnalogLSTMLayer)
            first_layer_args: RNNCell type, input_size, hidden_size, rpu_config, etc.
            other_layer_args: RNNCell type, hidden_size, hidden_size, rpu_config, etc.

        Returns:
            torch.nn.ModuleList, which is similar to a regular Python list,
            but where torch.nn.Module methods can be applied
        """
        layers = [layer(*first_layer_args)] + [
            layer(*other_layer_args) for _ in range(num_layers - 1)
        ]
        return ModuleList(layers)

    def get_zero_state(self, batch_size: int) -> List[Tensor]:
        """Returns a zeroed state.

        Args:
            batch_size: batch size of the input

        Returns:
           List of zeroed state tensors for each layer
        """
        return [lay.get_zero_state(batch_size) for lay in self.layers]

    def forward(  # pylint: disable=arguments-differ
        self, input: Tensor, states: List  # pylint: disable=redefined-builtin
    ) -> Tuple[Tensor, List]:
        """Forward pass.

        Args:
            input: input tensor
            states: list of LSTM state tensors

        Returns:
            outputs and states
        """
        # List[RNNState]: One state per layer.
        output_states = jit.annotate(List, [])
        output = input

        for i, rnn_layer in enumerate(self.layers):
            state = states[i]
            output, out_state = rnn_layer(output, state)
            # Apply the dropout layer except the last layer.
            if i < self.num_layers - 1 and self.dropout_layer is not None:
                output = self.dropout_layer(output)
            output_states += [out_state]

        return output, output_states


class AnalogRNN(AnalogContainerBase, Module):
    """Modular RNN that uses analog tiles.

    Args:
        cell: type of Analog RNN cell (AnalogLSTMCell/AnalogGRUCell/AnalogVanillaRNNCell)
        input_size: in_features to W_{ih} matrix of first layer
        hidden_size: in_features and out_features for W_{hh} matrices
        bias: whether to use a bias row on the analog tile or not
        rpu_config: configuration for an analog resistive processing
            unit. If not given a native torch model will be
            constructed instead.
        tile_module_class: Class for the analog tile module (default
            will be specified from the ``RPUConfig``).
        xavier: whether standard PyTorch LSTM weight
            initialization (default) or Xavier initialization
        num_layers: number of serially connected RNN layers
        bidir: if True, becomes a bidirectional RNN
        dropout: dropout applied to output of all RNN layers except last
    """

    # pylint: disable=abstract-method, too-many-arguments

    def __init__(
        self,
        cell: Type,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        rpu_config: Optional[RPUConfigBase] = None,
        tile_module_class: Optional[Type] = None,
        xavier: bool = False,
        num_layers: int = 1,
        bidir: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        if bidir:
            layer = AnalogBidirRNNLayer
            num_dirs = 2
        else:
            layer = AnalogRNNLayer
            num_dirs = 1

        self.rnn = ModularRNN(
            num_layers,
            layer,
            dropout,
            first_layer_args=[cell, input_size, hidden_size, bias, rpu_config, tile_module_class],
            other_layer_args=[
                cell,
                num_dirs * hidden_size,
                hidden_size,
                bias,
                rpu_config,
                tile_module_class,
            ],
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.reset_parameters(xavier)

    @no_grad()
    def init_layers(
        self, weight_init_fn: Callable, bias_init_fn: Optional[Callable] = None
    ) -> None:
        """Init the analog layers with custom functions.

        Args:
            weight_init_fn: in-place tensor function applied to weight of
                ``AnalogLinear`` layers
            bias_init_fn: in-place tensor function applied to bias of
                ``AnalogLinear`` layers

        Note:
            If no bias init function is provided the weight init
            function is taken for the bias as well.
        """

        def init_weight_and_bias(weight: Tensor, bias: Optional[Tensor]) -> None:
            """Init the weight and bias"""
            weight_init_fn(weight.data)
            if bias is not None:
                if bias_init_fn is None:
                    weight_init_fn(bias.data)
                else:
                    bias_init_fn(bias.data)

        for module in self.modules():
            if isinstance(module, AnalogLinear):
                weight, bias = module.get_weights()
                init_weight_and_bias(weight, bias)
                module.set_weights(weight, bias)
            elif isinstance(module, Linear):
                # init torch layers if any
                init_weight_and_bias(module.weight, module.bias)

    def reset_parameters(self, xavier: bool = False) -> None:
        """Weight and bias initialization.

        Args:
            xavier: whether standard PyTorch LSTM weight
               initialization (default) or Xavier initialization
        """
        if xavier:
            self.init_layers(init.xavier_uniform_, init.zeros_)
        else:
            stdv = 1.0 / math.sqrt(self.hidden_size)
            self.init_layers(lambda x: x.uniform_(-stdv, stdv))

    def get_zero_state(self, batch_size: int) -> List[Tensor]:
        """Returns a zeroed RNN state based on cell type and layer type

        Args:
            batch_size: batch size of the input

        Returns:
           List of zeroed state tensors for each layer

        """
        return self.rnn.get_zero_state(batch_size)

    def forward(
        self, input: Tensor, states: Optional[List] = None  # pylint: disable=redefined-builtin
    ) -> Tuple[Tensor, List]:
        """Forward pass.

        Args:
            input: input tensor
            states: list of LSTM state tensors

        Returns:
            outputs and states
        """

        if states is None:
            # TODO: batch_first.
            states = self.get_zero_state(input.shape[1])

        return self.rnn(input, states)

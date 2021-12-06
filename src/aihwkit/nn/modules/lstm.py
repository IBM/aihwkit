# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file has been adapted from:
# https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py
# Licensed under the following terms (https://github.com/pytorch/pytorch/blob/master/LICENSE):

# From PyTorch:
#
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou,
#                                                   Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# From Caffe2:
#
# Copyright (c) 2016-present, Facebook Inc. All rights reserved.
#
# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.
#
# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.
#
# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.
#
# All contributions by Kakao Brain:
# Copyright 2019-2020 Kakao Brain
#
# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.
#
# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.
#
# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is
# committed.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
#    and IDIAP Research Institute nor the names of its contributors may be
#    used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Analog LSTM layers."""

import warnings
import math
from typing import Any, List, Optional, Tuple, Type, Callable
from collections import namedtuple

from torch import Tensor, sigmoid, stack, tanh, jit, zeros
from torch.nn import Dropout, ModuleList, init

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.nn.modules.base import AnalogModuleBase, RPUConfigAlias

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])


class AnalogLSTMCell(AnalogSequential):
    """Analog LSTM Cell.

    Args:
        input_size: in_features size for W_ih matrix
        hidden_size: in_features and out_features size for W_hh matrix
        bias: whether to use a bias row on the analog tile or not
        rpu_config: configuration for an analog resistive processing unit
        realistic_read_write: whether to enable realistic read/write
            for setting initial weights and read out of weights
        weight_scaling_omega: the weight value where the max
            weight will be scaled to. If zero, no weight scaling will
            be performed
    """
    # pylint: disable=abstract-method

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool,
            rpu_config: Optional[RPUConfigAlias],
            realistic_read_write: bool = False,
            weight_scaling_omega: float = 0.0
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = AnalogLinear(input_size, 4 * hidden_size, bias=bias,
                                      rpu_config=rpu_config,
                                      realistic_read_write=realistic_read_write,
                                      weight_scaling_omega=weight_scaling_omega)
        self.weight_hh = AnalogLinear(hidden_size, 4 * hidden_size, bias=bias,
                                      rpu_config=rpu_config,
                                      realistic_read_write=realistic_read_write,
                                      weight_scaling_omega=weight_scaling_omega)

    def forward(
            self,
            input_: Tensor,
            state: Tuple[Tensor, Tensor]
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


class AnalogLSTMLayer(AnalogSequential):
    """Analog LSTM Layer.

    Args:
        cell: LSTMCell type (e.g. AnalogLSTMCell)
        cell_args: arguments to LSTMCell (e.g. input_size, hidden_size, rpu_configs)
    """
    # pylint: disable=abstract-method

    def __init__(self, cell: AnalogLSTMCell, *cell_args: Any):
        super().__init__()
        self.cell = cell(*cell_args)

    def forward(
            self, input_: Tensor,
            state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # pylint: disable=arguments-differ
        inputs = input_.unbind(0)
        outputs = jit.annotate(List[Tensor], [])
        for input_item in inputs:
            out, state = self.cell(input_item, state)
            outputs += [out]
        return stack(outputs), state


class ModularAnalogLSTMWithDropout(AnalogSequential):
    """Modular Analog LSTM with dropout.

    Args:
        num_layers: number of serially connected LSTM layers
        layer: LSTM layer type (e.g. AnalogLSTMLayer)
        dropout: dropout applied to output of all LSTM layers except last
        first_layer_args: LSTMCell type, input_size, hidden_size, rpu_config, etc.
        other_layer_args: LSTMCell type, hidden_size, hidden_size, rpu_config, etc.
    """
    # pylint: disable=abstract-method

    # Necessary for iterating through self.layers and dropout support
    __constants__ = ['layers', 'num_layers']

    def __init__(
            self,
            num_layers: int,
            layer: Type,
            dropout: float,
            first_layer_args: Any,
            other_layer_args: Any):
        super().__init__()
        self.layers = init_stacked_analog_lstm(num_layers, layer, first_layer_args,
                                               other_layer_args)

        # Introduce a Dropout layer on the outputs of each LSTM layer except
        # the last layer.
        self.num_layers = num_layers
        if num_layers == 1 and dropout > 0:
            warnings.warn('dropout lstm adds dropout layers after all but last '
                          'recurrent layer, it expects num_layers greater than '
                          '1, but got num_layers = 1')
        self.dropout_layer = Dropout(dropout) if dropout else None

    def forward(
            self,
            input_: Tensor,
            states: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # pylint: disable=arguments-differ
        # List[LSTMState]: One state per layer.
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input_

        for i, rnn_layer in enumerate(self.layers):
            state = states[i]
            output, out_state = rnn_layer(output, state)
            # Apply the dropout layer except the last layer.
            if i < self.num_layers - 1 and self.dropout_layer is not None:
                output = self.dropout_layer(output)
            output_states += [out_state]
            i += 1

        return output, output_states


class AnalogLSTM(AnalogSequential):
    """Modular LSTM that uses analog tiles.

    Args:
        input_size: in_features to W_{ih} matrix of first layer
        hidden_size: in_features and out_features for W_{hh} matrices
        num_layers: number of serially connected LSTM layers
        dropout: dropout applied to output of all LSTM layers except last
        bias: whether to use a bias row on the analog tile or not
        rpu_config: resistive processing unit configuration.
        realistic_read_write: whether to enable realistic read/write
            for setting initial weights and read out of weights
        weight_scaling_omega: the weight value where the max
            weight will be scaled to. If zero, no weight scaling will
            be performed
        xavier: whether standard PyTorch LSTM weight
            initialization (default) or Xavier initialization
    """
    # pylint: disable=abstract-method

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            dropout: float = 0.0,
            bias: bool = True,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False,
            weight_scaling_omega: float = 0.0,
            xavier: bool = False):
        super().__init__()

        # Default to SingleRPUConfig with ConstantStepDevice.
        if not rpu_config:
            rpu_config = SingleRPUConfig(device=ConstantStepDevice())

        self.lstm = ModularAnalogLSTMWithDropout(
            num_layers, AnalogLSTMLayer, dropout,
            first_layer_args=[AnalogLSTMCell, input_size, hidden_size, bias,
                              rpu_config, realistic_read_write, weight_scaling_omega],
            other_layer_args=[AnalogLSTMCell, hidden_size, hidden_size, bias,
                              rpu_config, realistic_read_write, weight_scaling_omega])
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.reset_parameters(xavier)

    def init_layers(
            self,
            weight_init_fn: Callable,
            bias_init_fn: Optional[Callable] = None
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
        def init_analog_layer(layer: AnalogModuleBase) -> None:
            """Init the weights and bias of an analog linear layer."""
            weight_init_fn(layer.weight.data)
            if layer.use_bias:
                if bias_init_fn is None:
                    weight_init_fn(layer.bias.data)
                else:
                    bias_init_fn(layer.bias.data)

            layer.set_weights(layer.weight, layer.bias)

        self._apply_to_analog(init_analog_layer)  # pylint: disable=protected-access

    def reset_parameters(self, xavier: bool = False) -> None:
        """Weight and bias initialization.

        Args:
            xavier: whether standard PyTorch LSTM weight
               initialization (default) or Xavier initialization
        """
        if xavier:
            self.init_layers(init.xavier_uniform_, init.zeros_)
        else:
            stdv = 1. / math.sqrt(self.hidden_size)
            self.init_layers(lambda x: x.uniform_(-stdv, stdv))

    def forward(
            self,
            x: Tensor,
            states: Optional[List[Tuple[Tensor, Tensor]]] = None
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # pylint: disable=arguments-differ
        if states is None:
            # TODO: batch_first.
            states = [LSTMState(zeros(x.size()[1], self.hidden_size),
                                zeros(x.size()[1], self.hidden_size))
                      for _ in range(self.num_layers)]

        return self.lstm(x, states)


def init_stacked_analog_lstm(
        num_layers: int,
        layer: Type,
        first_layer_args: Any,
        other_layer_args: Any
) -> ModuleList:
    """Construct a list of LSTMLayers over which to iterate.

    Args:
        num_layers: number of serially connected LSTM layers
        layer: LSTM layer type (e.g. AnalogLSTMLayer)
        first_layer_args: LSTMCell type, input_size, hidden_size, rpu_config, etc.
        other_layer_args: LSTMCell type, hidden_size, hidden_size, rpu_config, etc.

    Returns:
        torch.nn.ModuleList, which is similar to a regular Python list,
        but where torch.nn.Module methods can be applied
    """
    layers = [layer(*first_layer_args)] \
        + [layer(*other_layer_args) for _ in range(num_layers - 1)]
    return ModuleList(layers)

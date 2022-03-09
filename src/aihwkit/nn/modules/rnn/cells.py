import warnings
import math
from typing import Any, List, Optional, Tuple, Type, Callable, Union
from collections import namedtuple

from torch import Tensor, sigmoid, stack, relu, tanh, jit, zeros, cat
from torch.nn import Dropout, ModuleList, init

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.nn.modules.base import AnalogModuleBase, RPUConfigAlias

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])

class AnalogVanillaRNNCell(AnalogSequential):
    """Analog Vanilla RNN Cell.

    Args:
        input_size: in_features size for W_ih matrix
        hidden_size: in_features and out_features size for W_hh matrix
        bias: whether to use a bias row on the analog tile or not
        rpu_config: configuration for an analog resistive processing unit
        realistic_read_write: whether to enable realistic read/write
            for setting initial weights and read out of weights
    """
    # pylint: disable=abstract-method
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False,
    ):
        super().__init__()

        # Default to SingleRPUConfig with ConstantStepDevice.
        if not rpu_config:
            rpu_config = SingleRPUConfig(device=ConstantStepDevice())

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = AnalogLinear(input_size, hidden_size, bias=bias,
                                      rpu_config=rpu_config,
                                      realistic_read_write=realistic_read_write)
        self.weight_hh = AnalogLinear(hidden_size, hidden_size, bias=bias,
                                      rpu_config=rpu_config,
                                      realistic_read_write=realistic_read_write)

    def zero_state(self, batch_size):
        """Returns a zeroed Vanilla RNN state

        Args:
            batch_size: batch size of the input

        """
        return zeros(batch_size, self.hidden_size)

    def forward(
            self,
            input_: Tensor,
            state: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # pylint: disable=arguments-differ
        igates = self.weight_ih(input_)
        hgates = self.weight_hh(state)

        out = tanh(igates + hgates)

        return out, out #output will also be hidden state

class AnalogLSTMCell(AnalogSequential):
    """Analog LSTM Cell.

    Args:
        input_size: in_features size for W_ih matrix
        hidden_size: in_features and out_features size for W_hh matrix
        bias: whether to use a bias row on the analog tile or not
        rpu_config: configuration for an analog resistive processing unit
        realistic_read_write: whether to enable realistic read/write
            for setting initial weights and read out of weights
    """
    # pylint: disable=abstract-method

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False,
    ):
        super().__init__()

        # Default to SingleRPUConfig with ConstantStepDevice.
        if not rpu_config:
            rpu_config = SingleRPUConfig(device=ConstantStepDevice())

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = AnalogLinear(input_size, 4 * hidden_size, bias=bias,
                                      rpu_config=rpu_config,
                                      realistic_read_write=realistic_read_write)
        self.weight_hh = AnalogLinear(hidden_size, 4 * hidden_size, bias=bias,
                                      rpu_config=rpu_config,
                                      realistic_read_write=realistic_read_write)

    def zero_state(self, batch_size):
        """Returns a zeroed LSTM state

        Args:
            batch_size: batch size of the input

        """
        return LSTMState(zeros(batch_size, self.hidden_size),
                         zeros(batch_size, self.hidden_size))

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

class AnalogGRUCell(AnalogSequential):
    """Analog GRU Cell.

    Args:
        input_size: in_features size for W_ih matrix
        hidden_size: in_features and out_features size for W_hh matrix
        bias: whether to use a bias row on the analog tile or not
        rpu_config: configuration for an analog resistive processing unit
        realistic_read_write: whether to enable realistic read/write
            for setting initial weights and read out of weights
    """
    # pylint: disable=abstract-method

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False,
    ):
        super().__init__()

        # Default to SingleRPUConfig with ConstantStepDevice.
        if not rpu_config:
            rpu_config = SingleRPUConfig(device=ConstantStepDevice())

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = AnalogLinear(input_size, 3 * hidden_size, bias=bias,
                                      rpu_config=rpu_config,
                                      realistic_read_write=realistic_read_write)
        self.weight_hh = AnalogLinear(hidden_size, 3 * hidden_size, bias=bias,
                                      rpu_config=rpu_config,
                                      realistic_read_write=realistic_read_write)

    def zero_state(self, batch_size):
        """Returns a zeroed GRU RNN state

        Args:
            batch_size: batch size of the input

        """
        return zeros(batch_size, self.hidden_size)

    def forward(
            self,
            input_: Tensor,
            state: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # pylint: disable=arguments-differ
        gi = self.weight_ih(input_)
        gh = self.weight_hh(state)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = sigmoid(i_r + h_r)
        inputgate = sigmoid(i_i + h_i)
        newgate = tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (state - newgate)

        return hy, hy #output will also be hidden state
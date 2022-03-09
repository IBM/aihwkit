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

class AnalogRNNLayer(AnalogSequential):
    """Analog RNN Layer.

    Args:
        cell: RNNCell type (e.g. AnalogLSTMCell)
        cell_args: arguments to RNNCell (e.g. input_size, hidden_size, rpu_configs)
    """
    # pylint: disable=abstract-method

    def __init__(self, cell: Type, *cell_args: Any):
        super().__init__()
        self.cell = cell(*cell_args)

    def forward(
            self, input_: Tensor,
            state: Union[Tuple[Tensor, Tensor], Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # pylint: disable=arguments-differ
        inputs = input_.unbind(0)
        outputs = jit.annotate(List[Tensor], [])
        for input_item in inputs:
            out, state = self.cell(input_item, state)
            outputs += [out]
        return stack(outputs), state

    def zero_state(self, batch_size):
    	return self.cell.zero_state(batch_size)

def reverse(lst: List[Tensor]) -> List[Tensor]:
    return lst[::-1]

class AnalogReverseRNNLayer(AnalogSequential):
    def __init__(self, cell: Type, *cell_args: Any):
        super(AnalogReverseRNNLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, input: Tensor, 
                state: Union[Tuple[Tensor, Tensor], Tensor]
                ) -> Tuple[Tensor, Union[Tuple[Tensor, Tensor], Tensor]]:
        inputs = reverse(input.unbind(0))
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return stack(reverse(outputs)), state

class AnalogBidirRNNLayer(AnalogSequential):
    __constants__ = ['directions']

    def __init__(self, cell: Type, *cell_args: Any):
        super(AnalogBidirRNNLayer, self).__init__()
        self.directions = ModuleList([
            AnalogRNNLayer(cell, *cell_args),
            AnalogReverseRNNLayer(cell, *cell_args),
        ])

    def zero_state(self, batch_size):
    	return [self.directions[0].zero_state(batch_size), self.directions[0].zero_state(batch_size)] #forward and backwards zero_states

    def forward(self, input: Tensor, 
                states: List[Union[Tuple[Tensor, Tensor], Tensor]]
                ) -> Tuple[Tensor, List[Union[Tuple[Tensor, Tensor], Tensor]]]:
        # List[RNNState]: [forward RNNState, backward RNNState]
        outputs = jit.annotate(List[Tensor], [])
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        i = 0
        for direction in self.directions:
            state = states[i]
            out, out_state = direction(input, state)
            outputs += [out]
            output_states += [out_state]
            i += 1
        return cat(outputs, -1), output_states
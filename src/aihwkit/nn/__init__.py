# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Neural network modules."""

# Convenience imports for easier access to the classes.

from aihwkit.nn.modules.container import AnalogSequential, AnalogWrapper
from aihwkit.nn.modules.conv import AnalogConv1d, AnalogConv2d, AnalogConv3d
from aihwkit.nn.modules.linear import AnalogLinear
from aihwkit.nn.modules.rnn.rnn import AnalogRNN
from aihwkit.nn.modules.rnn.cells import (
    AnalogGRUCell,
    AnalogLSTMCell,
    AnalogVanillaRNNCell,
    AnalogLSTMCellCombinedWeight,
)
from aihwkit.nn.modules.linear_mapped import AnalogLinearMapped
from aihwkit.nn.modules.conv_mapped import (
    AnalogConv1dMapped,
    AnalogConv2dMapped,
    AnalogConv3dMapped,
)

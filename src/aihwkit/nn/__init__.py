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

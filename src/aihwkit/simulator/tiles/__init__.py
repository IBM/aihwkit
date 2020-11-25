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

"""High level analog tiles."""

# Convenience imports for easier access to the classes.

from aihwkit.simulator.tiles.analog import AnalogTile, CudaAnalogTile
from aihwkit.simulator.tiles.base import BaseTile
from aihwkit.simulator.tiles.floating_point import (
    CudaFloatingPointTile, FloatingPointTile
)
from aihwkit.simulator.tiles.inference import CudaInferenceTile, InferenceTile

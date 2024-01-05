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

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-lines

"""Runtime parameters for settings at simulation time."""

from dataclasses import dataclass
from .helpers import _PrintableMixin
from .enums import RPUDataType


@dataclass
class RuntimeParameter(_PrintableMixin):
    """Parameter that define the analog-matvec (forward / backward) and
    peripheral digital input-output behavior.

    Here one can enable analog-digital conversion, dynamic input
    scaling, and define the properties of the analog-matvec
    computations, such as noise and non-idealities (e.g. IR-drop).
    """

    data_type: RPUDataType = RPUDataType.FLOAT
    """Data type to use for the C++ bindings and simulations.
    """

    on_the_fly_bindings: bool = False
    """Enable on the fly generation of some parameter bindings.

    For some post update steps (e.g. for inference tile), parameter
    bindings are generated on the fly. This is only useful if these
    parameters are vary during training and is turned off by default
    for performance reason (which typically is not supported as the
    `RPUConfig` is considered fixed once the analog tile is
    instantiated).

    If indeed parameters (e.g. ``WeightClipParameter``,
    ``WeightModifierParameter``, and ``WeightRemapParameter``) vary
    during training, this options as to be turned off to re-generate
    the bindings on the fly (that is each mini-batch).

    """

    offload_input: bool = False
    """Whether to offload the stored input for the update pass on
    CPU to save GPU memory.
    """

    offload_gradient: bool = False
    """Whether to offload the stored gradient for the update pass on
    CPU to save GPU memory.

    Note:
       Only for in case tiles are simulated with RPUCuda library.
    """

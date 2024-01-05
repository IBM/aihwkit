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

"""Compound configurations presets for resistive processing units."""

# pylint: disable=too-many-instance-attributes
from typing import List
from dataclasses import dataclass, field

from aihwkit.simulator.configs.compounds import OneSidedUnitCell
from aihwkit.simulator.parameters.training import UpdateParameters
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.presets.devices import PCMPresetDevice
from aihwkit.simulator.presets.utils import PresetIOParameters, PresetUpdateParameters


@dataclass
class PCMPresetUnitCell(OneSidedUnitCell):
    """A unit cell that is comprised of two uni-directional PCM devices of
    opposite sign (see :class:`~PCMPresetDevice`).

    Check for refresh is performed after each mini-batch update. See
    :class:`~aihwkit.simulator.configs.device.OneSidedUnitCell` for
    details on the refresh implementation.
    """

    unit_cell_devices: List = field(default_factory=lambda: [PCMPresetDevice(), PCMPresetDevice()])

    refresh_every: int = 1
    units_in_mbatch: bool = True
    refresh_forward: IOParameters = field(default_factory=PresetIOParameters)
    refresh_update: UpdateParameters = field(
        default_factory=lambda: PresetUpdateParameters(desired_bl=31)
    )

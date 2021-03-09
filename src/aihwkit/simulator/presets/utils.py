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

"""Utils for configurations presets for resistive processing units."""

# pylint: disable=too-many-instance-attributes

from dataclasses import dataclass

from aihwkit.simulator.configs.utils import (
    BoundManagementType, IOParameters, NoiseManagementType, PulseType,
    UpdateParameters, WeightNoiseType
)


@dataclass
class PresetIOParameters(IOParameters):
    """Preset for the forward and backward pass parameters."""

    bound_management: BoundManagementType = BoundManagementType.ITERATIVE
    noise_management: NoiseManagementType = NoiseManagementType.ABS_MAX

    inp_res: float = 1.0 / (2**7 - 2)  # 7 bit DAC.
    inp_sto_round: bool = False

    out_bound: float = 20.0
    out_noise: float = 0.1
    out_res: float = 1.0 / (2**9 - 2)  # 9 bit ADC.

    # No read noise by default.
    w_noise: float = 0.0
    w_noise_type: WeightNoiseType = WeightNoiseType.NONE


@dataclass
class PresetUpdateParameters(UpdateParameters):
    """Preset for the general update behavior.

    Stochastic pulse trains with parallel update by default.
    """

    desired_bl: int = 31  # Less than 32 preferable (faster implementation).
    pulse_type: PulseType = PulseType.STOCHASTIC_COMPRESSED
    update_bl_management: bool = True  # Dynamically adjusts pulse train length (max 31).
    update_management: bool = True

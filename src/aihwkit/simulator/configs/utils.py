# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Legacy location of the utils module. Please import from parameter.utils / enum in future."""

# pylint: disable=unused-import

from aihwkit.simulator.parameters import (
    IOParameters,
    UpdateParameters,
    WeightModifierParameter,
    WeightClipParameter,
    WeightRemapParameter,
    SimpleDriftParameter,
    DriftParameter,
    MappingParameter,
    InputRangeParameter,
    PrePostProcessingParameter,
)

from aihwkit.simulator.parameters.enums import (
    BoundManagementType,
    NoiseManagementType,
    WeightNoiseType,
    PulseType,
    WeightModifierType,
    WeightClipType,
    WeightRemapType,
    VectorUnitCellUpdatePolicy,
    AnalogMVType,
)

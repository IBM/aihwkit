# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""RPU simulator bindings."""

# This import is required in order to load the `torch` shared libraries, which
# the simulator shared library is linked against.

from .enums import (
    RPUDataType,
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

from .training import UpdateParameters

from .io import IOParameters, IOParametersIRDropT

from .mapping import MappingParameter

from .pre_post import InputRangeParameter, PrePostProcessingParameter

from .inference import (
    WeightModifierParameter,
    WeightClipParameter,
    WeightRemapParameter,
    SimpleDriftParameter,
    DriftParameter,
)

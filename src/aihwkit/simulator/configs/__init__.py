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

"""Configurations for resistive processing units."""

from aihwkit.simulator.parameters import (
    IOParameters,
    IOParametersIRDropT,
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
    CountLRFeedbackPolicy,
)
from .devices import (
    FloatingPointDevice,
    IdealDevice,
    ConstantStepDevice,
    LinearStepDevice,
    SoftBoundsDevice,
    SoftBoundsPmaxDevice,
    SoftBoundsReferenceDevice,
    ExpStepDevice,
    PowStepDevice,
    PowStepReferenceDevice,
    PiecewiseStepDevice,
)
from .compounds import (
    VectorUnitCell,
    ReferenceUnitCell,
    OneSidedUnitCell,
    DifferenceUnitCell,
    TransferCompound,
    BufferedTransferCompound,
    ChoppedTransferCompound,
    DynamicTransferCompound,
    MixedPrecisionCompound,
)
from .configs import (
    FloatingPointRPUConfig,
    InferenceRPUConfig,
    SingleRPUConfig,
    UnitCellRPUConfig,
    DigitalRankUpdateRPUConfig,
    TorchInferenceRPUConfig,
    TorchInferenceRPUConfigIRDropT,
)

from .helpers import build_config

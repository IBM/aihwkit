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

"""RPU configurations presets used for the composer interface."""

# pylint: disable=too-many-instance-attributes

from dataclasses import dataclass, field
from aihwkit.simulator.configs.configs import InferenceRPUConfig

from aihwkit.simulator.parameters import (
    IOParameters,
    WeightClipParameter,
    WeightModifierParameter,
    MappingParameter,
    BoundManagementType,
    NoiseManagementType,
    WeightNoiseType,
    WeightModifierType,
    WeightClipType,
)
from aihwkit.inference import (
    BaseDriftCompensation,
    BaseNoiseModel,
    GlobalDriftCompensation,
    PCMLikeNoiseModel,
)


@dataclass
class WebComposerIOParameters(IOParameters):
    r"""Preset for the forward and backward pass parameters.

    The default values used for the web composer
    """

    bound_management: BoundManagementType = BoundManagementType.ITERATIVE
    noise_management: NoiseManagementType = NoiseManagementType.ABS_MAX

    inp_res: float = 1.0 / (2**7 - 2)  # 7 bit DAC.
    inp_sto_round: bool = False

    out_bound: float = 20.0
    out_noise: float = 0.1
    out_res: float = 1.0 / (2**9 - 2)  # 9 bit ADC.

    w_noise: float = 0.0
    w_noise_type: WeightNoiseType = WeightNoiseType.NONE


@dataclass
class WebComposerWeightModifierParameter(WeightModifierParameter):
    r"""Preset for the WeightModifierParameter

    The default values used for the web composer
    """

    type: WeightModifierType = WeightModifierType.ADD_NORMAL
    std_dev: float = 0.1


@dataclass
class WebComposerWeightClipParameter(WeightClipParameter):
    """Parameter that clip the weights during hardware-aware training.

    The default values used for the web composer.
    """

    fixed_value: float = 1.0
    type: WeightClipType = WeightClipType.FIXED_VALUE


@dataclass
class WebComposerMappingParameter(MappingParameter):
    """Parameter related to hardware design and the mapping of logical
    weight matrices to physical tiles.

    The default values used for the web composer.
    """

    digital_bias: bool = True

    weight_scaling_omega: float = 1.0
    weight_scaling_columnwise: bool = True
    learn_out_scaling: bool = True
    out_scaling_columnwise: bool = False

    max_input_size: int = 512
    max_output_size: int = 512


@dataclass
class WebComposerInferenceRPUConfig(InferenceRPUConfig):
    """Preset configuration used as default for the Inference Composer"""

    forward: IOParameters = field(
        default_factory=WebComposerIOParameters, metadata=dict(bindings_include=True)
    )
    """Input-output parameter setting for the forward direction."""

    noise_model: BaseNoiseModel = field(default_factory=PCMLikeNoiseModel)
    """Statistical noise model to be used during (realistic) inference."""

    drift_compensation: BaseDriftCompensation = field(default_factory=GlobalDriftCompensation)
    """For compensating the drift during inference only."""

    clip: WeightClipParameter = field(default_factory=WebComposerWeightClipParameter)
    """Parameter for weight clip."""

    modifier: WeightModifierParameter = field(default_factory=WebComposerWeightModifierParameter)
    """Parameter for weight modifier."""

    mapping: MappingParameter = field(default_factory=WebComposerMappingParameter)
    """Parameter related to mapping weights to tiles for supporting modules."""


@dataclass
class OldWebComposerMappingParameter(WebComposerMappingParameter):
    """Parameter related to hardware design and the mapping of logical
    weight matrices to physical tiles.

    The default values used for the web composer.
    """

    digital_bias: bool = False

    weight_scaling_omega: float = 1.0
    weight_scaling_columnwise: bool = True
    learn_out_scaling: bool = False
    out_scaling_columnwise: bool = False
    max_input_size: int = 0
    max_output_size: int = 0


@dataclass
class OldWebComposerInferenceRPUConfig(WebComposerInferenceRPUConfig):
    """Preset configuration used as default for the Inference Composer"""

    mapping: MappingParameter = field(default_factory=OldWebComposerMappingParameter)
    """Parameter related to mapping weights to tiles for supporting modules."""

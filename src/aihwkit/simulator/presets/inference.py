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

# pylint: disable=too-many-lines

"""RPU configurations presets for resistive processing units."""
from typing import Optional
from dataclasses import dataclass, field

from aihwkit.simulator.configs.configs import InferenceRPUConfig
from aihwkit.simulator.parameters import (
    MappingParameter,
    IOParameters,
    PrePostProcessingParameter,
    InputRangeParameter,
    WeightClipParameter,
    WeightRemapParameter,
)
from aihwkit.simulator.parameters.enums import (
    WeightClipType,
    BoundManagementType,
    NoiseManagementType,
    WeightNoiseType,
    WeightRemapType,
)
from aihwkit.inference.noise.base import BaseNoiseModel
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.inference.compensation.base import BaseDriftCompensation
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.simulator.presets.utils import PresetIOParameters


# Inference
@dataclass
class StandardHWATrainingPreset(InferenceRPUConfig):
    """Preset configuration for AIMC (Analog In-Mememory Compute)
    accuracy evaluation.

    This preset configuration can be used as a baseline for
    comparative AIMC studies. It defines a standard AIMC noisy
    inference evaluation setting for comparable benchmarking of
    hardware-aware training methods and noise robustness of different
    DNN architectures. for AIMC.

    See `Rasch et al. ArXiv 2023`_ for detailed discussions and
    attainable accuracy for state-of-the art hardware-aware training
    across many larger-scale DNNs.

    .. _`Rasch et al. ArXiv 2023`: https://arxiv.org/abs/2302.08469

    """

    mapping: MappingParameter = field(
        default_factory=lambda: MappingParameter(
            weight_scaling_omega=1.0,
            weight_scaling_columnwise=True,
            max_input_size=512,
            max_output_size=0,
            digital_bias=True,
            learn_out_scaling=True,
            out_scaling_columnwise=True,
        )
    )

    forward: IOParameters = field(
        default_factory=lambda: PresetIOParameters(
            inp_res=254.0,
            out_res=254.0,
            bound_management=BoundManagementType.NONE,
            noise_management=NoiseManagementType.CONSTANT,
            nm_thres=1.0,
            w_noise=0.0175,
            w_noise_type=WeightNoiseType.PCM_READ,
            ir_drop=1.0,
            out_noise=0.04,
            out_bound=10.0,
        )
    )

    remap: WeightRemapParameter = field(
        default_factory=lambda: WeightRemapParameter(
            remapped_wmax=1.0, type=WeightRemapType.CHANNELWISE_SYMMETRIC
        )
    )

    noise_model: BaseNoiseModel = field(default_factory=PCMLikeNoiseModel)

    drift_compensation: Optional[BaseDriftCompensation] = field(
        default_factory=GlobalDriftCompensation
    )

    pre_post: PrePostProcessingParameter = field(
        default_factory=lambda: PrePostProcessingParameter(
            input_range=InputRangeParameter(
                enable=True,
                init_value=3.0,
                init_from_data=100,
                init_std_alpha=3.0,
                decay=0.001,
                input_min_percentage=0.95,
                output_min_percentage=0.95,
                manage_output_clipping=False,
                gradient_scale=1.0,
                gradient_relative=True,
            )
        )
    )

    clip: WeightClipParameter = field(
        default_factory=lambda: WeightClipParameter(
            type=WeightClipType.FIXED_VALUE, fixed_value=1.0
        )
    )

# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""High level inference tools."""

# Convenience imports for easier access to the classes.
from aihwkit.inference.converter.base import BaseConductanceConverter
from aihwkit.inference.converter.conductance import (
    SinglePairConductanceConverter,
    SingleDeviceConductanceConverter,
    DualPairConductanceConverter,
    NPairConductanceConverter,
    CustomPairConductanceConverter
)
from aihwkit.inference.noise.base import BaseNoiseModel
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel, CustomDriftPCMLikeNoiseModel
from aihwkit.inference.noise.reram import ReRamWan2022NoiseModel, ReRamCMONoiseModel
from aihwkit.inference.noise.custom import StateIndependentNoiseModel
from aihwkit.inference.compensation.base import BaseDriftCompensation
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.inference.utils import drift_analog_weights, program_analog_weights

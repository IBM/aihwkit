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

"""High level inference tools."""

# Convenience imports for easier access to the classes.
from aihwkit.inference.converter.base import BaseConductanceConverter
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter
from aihwkit.inference.noise.base import BaseNoiseModel
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.inference.noise.reram import ReRamWan2022NoiseModel
from aihwkit.inference.noise.custom import StateIndependentNoiseModel
from aihwkit.inference.compensation.base import BaseDriftCompensation
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.inference.utils import drift_analog_weights, program_analog_weights

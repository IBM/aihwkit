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

"""Configurations presets for resistive processing units."""

from .configs import (
    # Single device configs.
    ReRamESPreset,
    ReRamSBPreset,
    CapacitorPreset,
    EcRamPreset,
    EcRamMOPreset,
    IdealizedPreset,
    GokmenVlasovPreset,
    PCMPreset,
    # 2-device configs.
    ReRamES2Preset,
    ReRamSB2Preset,
    Capacitor2Preset,
    EcRam2Preset,
    EcRamMO2Preset,
    Idealized2Preset,
    # 4-device configs.
    ReRamES4Preset,
    ReRamSB4Preset,
    Capacitor4Preset,
    EcRam4Preset,
    EcRamMO4Preset,
    Idealized4Preset,
    # Tiki-taka configs.
    TikiTakaReRamESPreset,
    TikiTakaReRamSBPreset,
    TikiTakaCapacitorPreset,
    TikiTakaEcRamPreset,
    TikiTakaEcRamMOPreset,
    TikiTakaIdealizedPreset,
    # TTv2 configs.
    TTv2ReRamESPreset,
    TTv2ReRamSBPreset,
    TTv2CapacitorPreset,
    TTv2EcRamPreset,
    TTv2EcRamMOPreset,
    TTv2IdealizedPreset,
    # c-TTv2 configs.
    ChoppedTTv2ReRamESPreset,
    ChoppedTTv2ReRamSBPreset,
    ChoppedTTv2CapacitorPreset,
    ChoppedTTv2EcRamPreset,
    ChoppedTTv2EcRamMOPreset,
    ChoppedTTv2IdealizedPreset,
    # AGAD configs.
    AGADReRamESPreset,
    AGADReRamSBPreset,
    AGADCapacitorPreset,
    AGADEcRamPreset,
    AGADEcRamMOPreset,
    AGADIdealizedPreset,
    # MixedPrecision configs.
    MixedPrecisionReRamESPreset,
    MixedPrecisionReRamSBPreset,
    MixedPrecisionCapacitorPreset,
    MixedPrecisionEcRamPreset,
    MixedPrecisionEcRamMOPreset,
    MixedPrecisionIdealizedPreset,
    MixedPrecisionGokmenVlasovPreset,
    MixedPrecisionPCMPreset,
)
from .inference import StandardHWATrainingPreset

from .devices import (
    ReRamESPresetDevice,
    ReRamSBPresetDevice,
    CapacitorPresetDevice,
    EcRamPresetDevice,
    EcRamMOPresetDevice,
    IdealizedPresetDevice,
    GokmenVlasovPresetDevice,
    PCMPresetDevice,
    ReRamArrayOMPresetDevice,
    ReRamArrayHfO2PresetDevice,
)
from .compounds import PCMPresetUnitCell
from .utils import PresetIOParameters, StandardIOParameters, PresetUpdateParameters

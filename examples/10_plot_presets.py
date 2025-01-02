# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 10: plotting of presets.

Plot the step response of different preset devices and preset configurations.
"""
# pylint: disable=invalid-name

import matplotlib.pyplot as plt

from aihwkit.utils.visualization import plot_device

from aihwkit.simulator.presets import (
    ReRamSBPresetDevice,
    ReRamESPresetDevice,
    CapacitorPresetDevice,
    EcRamPresetDevice,
    IdealizedPresetDevice,
    EcRamMOPresetDevice,
    PCMPresetUnitCell,
)


plt.ion()

# Note alternatively one can use plot_device_compact for a more compact
# plot.

# Idealized
plot_device(IdealizedPresetDevice(), n_steps=10000)

# ReRam based on ExpStep
plot_device(ReRamESPresetDevice(), n_steps=1000)

# ReRam based on SoftBounds
plot_device(ReRamSBPresetDevice(), n_steps=1000)

# Capacitor
plot_device(CapacitorPresetDevice(), n_steps=400)

# ECRAM
plot_device(EcRamPresetDevice(), n_steps=1000)

# Mo-ECRAM
plot_device(EcRamMOPresetDevice(), n_steps=8000)

# PCM
plot_device(PCMPresetUnitCell(), n_steps=80)

plt.show()

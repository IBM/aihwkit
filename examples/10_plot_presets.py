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

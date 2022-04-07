# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""aihwkit example 21: plotting of user defined device

Plot the step response of a user defined device
"""

# Imports from aihwkit.
from aihwkit.utils.visualization import plot_device_compact
from aihwkit.simulator.configs.devices import SelfDefineDevice

# Other imports.
import matplotlib.pyplot as plt
import csv
import os

# Declare variables and .csv path
n_points = 0
up_pulse = []
down_pulse = []
path = os.getcwd() + '/' + os.path.dirname(__file__) + '/csv/selfdefine.csv'

# Import .csv file to determine up_pulse, down_pulse, and n_points.
try:
    with open(path, newline = '') as csvfile:
        fieldnames = ['up_pulse', 'down_pulse']
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)
        for i, row in enumerate(reader):
            if i != 0:
                up_pulse.append(float(row['up_pulse']))
                down_pulse.append(float(row['down_pulse']))
            n_points = i
except:
    print("ERROR: Could not read .csv file")

plt.ion()
plot_device_compact(
    SelfDefineDevice(w_min=-1, w_max=1, dw_min=0.01, w_min_dtod=0.0, w_max_dtod=0.0, 
                                                                     n_points = n_points,
                                                                     up_pulse = up_pulse,
                                                                     down_pulse = down_pulse), n_steps=1000)

plt.show()
plt.savefig('my_figure.png')
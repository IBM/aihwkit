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

"""aihwkit example 22: plotting of a user defined device

Plot the step response of a piecewise linear step device
"""

# pylint: disable=redefined-outer-name, too-many-locals

import csv
import os
import numpy as np
import matplotlib.pyplot as plt

# Imports from aihwkit.
from aihwkit.utils.visualization import plot_device_compact
from aihwkit.simulator.configs.devices import PiecewiseStepDevice

# Declare variables and .csv path
DEVICE_FIT = True
if DEVICE_FIT:
    FILE_NAME = os.path.join(os.path.dirname(__file__), 'csv', 'gong_et_al.csv')
else:
    FILE_NAME = os.path.join(os.path.dirname(__file__), 'csv', 'selfdefine.csv')

def read_from_file(filename, from_pulse_response=True, n_segments=10, skip_rows=0):
    """Read the update steps from file and convert to the required device input format.

    Here the CSV file has two columns, one for the up and the second
    for the down pulses. The conductance values should be consecutive.

    Args:
        filename: CSV file name to read from

        from_pulse_response: whether to load from pulse response
            data. Otherwise the up/down pulse directly given for each
            segment

        n_segments: the number of segments

        skip_rows: initial rows to skip (to skip column names)

    Returns:
        piecewise_up: scaled vector of up pulses in the range w_min to w_max
        piecewise_down: scaled vector of down pulses in the range w_min to w_max
        dw_min: mimimal dw at zero
        up_down: bias at zero for up versus down direction
        up_data: up data read from file
        down_data: down data read from file
    """
    up_data = []
    down_data = []
    # Import .csv file to determine up_pulse, down_pulse, and n_points.
    with open(filename, newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['up_pulse', 'down_pulse']
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)
        for i, row in enumerate(reader):
            if i < skip_rows:
                continue

            up_data.append(float(row['up_pulse']))
            down_data.append(float(row['down_pulse']))

    up_data = np.array(up_data)
    down_data = np.array(down_data)

    if from_pulse_response:
        g_max = min(up_data.max(), down_data.max())
        g_min = max(up_data.min(), down_data.min())

        mean_up, _ = np.histogram(up_data, bins=n_segments, range=[g_min, g_max])
        mean_down, _ = np.histogram(down_data, bins=n_segments, range=[g_min, g_max])

        # scale to w_min = -1 and w_max = 1
        up_pulse = 1 / mean_up * (2 / n_segments)
        down_pulse = 1 / mean_down * (2 / n_segments)

    else:
        # directly given
        up_pulse = up_data
        down_pulse = down_data

    # compute the (scaled) inputs to the PiecewiseStepDevice
    center = len(up_pulse) // 2
    center_up = up_pulse[center]
    center_down = down_pulse[center]
    dw_min = max(center_up, center_down)
    up_down = center_up - center_down

    return ((up_pulse / center_up).tolist(), (down_pulse / center_down).tolist(),
            dw_min, up_down, up_data, down_data)


# define the device response from values in the CSV file
piecewise_up, piecewise_down, dw_min, up_down, up_data, down_data = read_from_file(
    FILE_NAME, from_pulse_response=DEVICE_FIT)

my_device = PiecewiseStepDevice(w_min=-1, w_max=1, w_min_dtod=0.0, w_max_dtod=0.0,
                                dw_min_std=0.0,
                                dw_min_dtod=0.0,
                                up_down_dtod=0.0,
                                dw_min=dw_min,
                                up_down=up_down,
                                piecewise_up=piecewise_up,
                                piecewise_down=piecewise_down
                                )
# plot the pulse response
plt.ion()
plot_device_compact(my_device, n_steps=1000)

plt.show()

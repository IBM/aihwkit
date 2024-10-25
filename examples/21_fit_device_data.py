# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 21: Fitting a device data to a piecewise step device.

Plots the fitted response of a piecewise step device
"""

# pylint: disable=redefined-outer-name, too-many-locals, invalid-name

import csv
import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Imports from aihwkit.
from aihwkit.utils.visualization import plot_device_compact
from aihwkit.simulator.configs import PiecewiseStepDevice
from aihwkit.simulator.rpu_base import cuda

USE_CUDA = cuda.is_compiled()

# Declare variables and .csv path
DEVICE_FIT = True
if DEVICE_FIT:
    FILE_NAME = os.path.join(os.path.dirname(__file__), "csv", "gong_et_al.csv")
else:
    FILE_NAME = os.path.join(os.path.dirname(__file__), "csv", "selfdefine.csv")


def read_from_file(
    filename: str, from_pulse_response: bool = True, n_segments: int = 10, skip_rows: int = 0
) -> Tuple[List[float], List[float], float, float, float, List[float], List[float]]:
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
        noise_std: update noise estimate
        up_data: up data read from file
        down_data: down data read from file
    """
    up_data = []
    down_data = []
    # Import .csv file to determine up_pulse, down_pulse, and n_points.
    with open(filename, newline="", encoding="utf-8-sig") as csvfile:
        fieldnames = ["up_data", "down_data"]
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)
        for i, row in enumerate(reader):
            if i < skip_rows:
                continue

            up_data.append(float(row["up_data"]))
            down_data.append(float(row["down_data"]))

    up_data = np.array(up_data)
    down_data = np.array(down_data)

    if from_pulse_response:
        g_max = min(up_data.max(), down_data.max())
        g_min = max(up_data.min(), down_data.min())

        def get_pulse_values(data):
            mean, edges = np.histogram(data, bins=n_segments, range=[g_min, g_max])
            idx = np.maximum(np.minimum(np.digitize(data, edges) - 1, n_segments - 1), 0)

            # scale to w_min = -1 and w_max = 1
            pulse_values = 1 / mean * (2 / n_segments)

            # rough estimate of the noise (ignoring the interpolation)
            noise_std = np.std(pulse_values[idx][:-1] - np.diff(up_data))

            return pulse_values, noise_std

        up_pulse, noise_std_up = get_pulse_values(up_data)
        down_pulse, noise_std_down = get_pulse_values(down_data)
        noise_std = (noise_std_up + noise_std_down) / 2

        # scale to w_min = -1 and w_max = 1
        up_data = (up_data - g_min) / (g_max - g_min) * 2 - 1
        down_data = (down_data - g_min) / (g_max - g_min) * 2 - 1

    else:
        # directly given
        up_pulse = up_data
        down_pulse = down_data
        noise_std = 0.0  # not given

    # compute the (scaled) inputs to the PiecewiseStepDevice
    center = len(up_pulse) // 2
    center_up = up_pulse[center]
    center_down = down_pulse[center]
    dw_min = max(center_up, center_down)
    up_down = center_up - center_down
    noise_std = noise_std / dw_min

    return (
        (up_pulse / center_up).tolist(),
        (down_pulse / center_down).tolist(),
        dw_min,
        up_down,
        noise_std,
        up_data,
        down_data,
    )


# define the device response from values in the CSV file
piecewise_up, piecewise_down, dw_min, up_down, noise_std, up_data, down_data = read_from_file(
    FILE_NAME, from_pulse_response=DEVICE_FIT
)

my_device = PiecewiseStepDevice(
    w_min=-1,
    w_max=1,
    w_min_dtod=0.0,
    w_max_dtod=0.0,
    dw_min_std=0.0,
    dw_min_dtod=0.0,
    up_down_dtod=0.0,
    dw_min=dw_min,
    up_down=up_down,
    write_noise_std=noise_std,
    piecewise_up=piecewise_up,
    piecewise_down=piecewise_down,
)
print(my_device)

# plot the pulse response
plt.ion()
fig = plot_device_compact(my_device, n_steps=1000, use_cuda=USE_CUDA)
if DEVICE_FIT:
    axis = fig.get_axes()[0]
    axis.plot(np.concatenate([up_data, down_data, up_data, down_data]), "k:", alpha=0.5)

plt.show()

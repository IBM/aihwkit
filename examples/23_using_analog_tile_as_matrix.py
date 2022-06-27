# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""aihwkit example 22: Simple example of how to use an analog tile as a matrix
"""
# pylint: disable=invalid-name
# pylint: disable=too-many-locals

# Imports from PyTorch.
from torch import randn
from torch import device as torch_device
from torch import cuda as torch_cuda

# Imports from aihwkit.
from aihwkit.simulator.tiles import AnalogTile
from aihwkit.simulator.presets import ReRamSBPreset

# Check GPU device
DEVICE = torch_device('cuda' if torch_cuda.is_available() else 'cpu')

# config the hardware properties
rpu_config = ReRamSBPreset()
# rpu_config.forward.out_noise = 0.1  # set some properties

# size of matrix
n = 50
m = 200
t = 10
matrix = 0.1 * randn(n, m)  # matrix A

x_values = randn(t, m)
d_values = randn(t, n)

# create analog tile (a single crossbar array)
analog_tile = AnalogTile(matrix.shape[0], matrix.shape[1], rpu_config)
analog_tile.set_weights(matrix, realistic=True)  # program weights

if DEVICE.type == 'cuda':
    analog_tile = analog_tile.cuda(DEVICE)
    x_values = x_values.cuda(DEVICE)
    d_values = d_values.cuda(DEVICE)

# compute matrix vector product (A * x)
y1 = analog_tile.forward(x_values)

# compute transposed matrix vector product (A' * d)
y2 = analog_tile.backward(d_values)

# compute rank update  A += - lr * x * d'
analog_tile.set_learning_rate(0.01)
analog_tile.update(x_values, - d_values)
current_analog_matrix = analog_tile.get_weights()  # perfect read
current_matrix = analog_tile.get_weights(realistic=True)  # actual read

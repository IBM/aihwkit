# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 22: Simple example of how to use an analog tile as a matrix
"""
# pylint: disable=invalid-name
# pylint: disable=too-many-locals

# Imports from PyTorch.
from torch import randn
from torch import device as torch_device

# Imports from aihwkit.
from aihwkit.linalg.matrix import AnalogMatrix
from aihwkit.simulator.presets import ReRamSBPreset
from aihwkit.simulator.rpu_base import cuda

# Check GPU device
DEVICE = torch_device("cuda" if cuda.is_compiled() else "cpu")

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
analog_matrix = AnalogMatrix(matrix, rpu_config, device=DEVICE)

if DEVICE.type == "cuda":
    x_values = x_values.cuda(DEVICE)
    d_values = d_values.cuda(DEVICE)

# compute matrix vector product (A * x)
y1 = analog_matrix @ x_values

# compute transposed matrix vector product (A' * d)
y2 = d_values @ analog_matrix

# compute rank update  A += - lr * x * d'
analog_matrix.ger(x_values, d_values, 0.01)

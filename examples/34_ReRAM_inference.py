# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 34: Matrix-Vector Multiplication (MVM) with ReRAM CMO.

Single matrix-vector multiplication of an analog matrix of size 64x64.
MVM accuracy is assessed for different times after programming.
"""
# pylint: disable=invalid-name

# Imports from PyTorch
from torch import nn, zeros, matmul, rand

# Imports from aihwkit
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.inference.noise.reram import ReRamCMONoiseModel
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import (
    BoundManagementType,
    WeightClipType,
    NoiseManagementType,
    WeightRemapType,
)
from aihwkit.simulator.parameters.io import IOParametersIRDropT

# Input/Output quantization
precision_i = 6
precision_o = 8

# Crossbar array size
crossbar_size = 64

# Program-verify acceptance range, either 2% or 0.2%
acceptance_range = 0.2

# Amount of vectors
batches = 100

# initial weights
matrix = rand(crossbar_size, crossbar_size) * 2 - 1

# probe vectors
input_probe = rand(crossbar_size, batches) * 2 - 1
real_MVM = matmul(input_probe.T, matrix.T)

# time of infernence ir 0s, 1s, 1h, 1d and 10y
t_inferences = [0, 1, 3600, 3600 * 24, 3600 * 24 * 365 * 10]
drifted_MVM = zeros((len(t_inferences), crossbar_size, batches))

# Wire resistance in Ohms
wire = 0.35


def inference_rpu_config():
    """Configures the RPU for inferencing with ReRAM CMO Noise model

    Args:
        None

    Returns:
        InferecneRPUConfig
    """
    rpu_config = InferenceRPUConfig()
    rpu_config.modifier.type = None

    rpu_config.mapping.digital_bias = False
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = False
    rpu_config.mapping.out_scaling_columnwise = False
    rpu_config.remap.type = WeightRemapType.LAYERWISE_SYMMETRIC

    rpu_config.clip.type = WeightClipType.NONE

    rpu_config.forward = IOParametersIRDropT()
    rpu_config.forward.is_perfect = False
    rpu_config.forward.out_noise = 0.0
    rpu_config.forward.inp_bound = 1.0
    rpu_config.forward.inp_res = 1 / (2**precision_i - 2)
    rpu_config.forward.out_bound = 15.0
    rpu_config.forward.out_res = 1 / (2**precision_o - 2)
    rpu_config.forward.ir_drop_g_ratio = 1.0 / wire / 100e-6
    rpu_config.forward.ir_drop = 1.0
    rpu_config.forward.ir_drop_rs = wire
    rpu_config.forward.bound_management = BoundManagementType.NONE
    rpu_config.forward.noise_management = NoiseManagementType.NONE

    rpu_config.noise_model = ReRamCMONoiseModel(g_max=88.19, g_min=9.0,
                                                acceptance_range=acceptance_range)
    rpu_config.drift_compensation = None
    return rpu_config


#  Initialize a squared matrix
model = nn.Linear(crossbar_size, crossbar_size, bias=False)
model.weight.data = matrix
rpu_conf = inference_rpu_config()

# Convert the FP layer to an analog tile
analog_model = convert_to_analog(model, rpu_conf)
analog_model = analog_model.eval()

# Program the weights for the configured noise model (i.e. programming noise)
analog_model.program_analog_weights(noise_model=rpu_conf.noise_model)

# Get the MVM accuracy at t=0s. Only considers programming noise
baseline_FP = matmul(input_probe.T, matrix.T)

# compute the MVM for each time of inference considering conductance relaxation
for i, t_inference in enumerate(t_inferences):
    # drift the weights for each time of inference and compute the MVM
    analog_model.drift_analog_weights(t_inference)
    for j in range(batches):
        drifted_MVM[i, :, j] = analog_model(input_probe[:, j])
drifted_MVM = drifted_MVM.detach()

# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 33: Matrix-Vector Multiplication (MVM) with ReRAM CMO.

Single matrix-vector multiplication of an analog matrix of size 64x64.
MVM accuracy is assessed for different times after programming.
"""
# pylint: disable=invalid-name

# Imports from PyTorch
from torch import Tensor, nn, zeros, matmul, sqrt, mean, random, rand

# Imports from aihwkit
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.inference.noise.reram import ReRamCMONoiseModel, ReRamWan2022NoiseModel
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import (
    WeightModifierType,
    BoundManagementType,
    WeightClipType,
    NoiseManagementType,
    WeightRemapType,
)
from aihwkit.simulator.tiles.inference import InferenceTileWithPeriphery, InferenceTile
from aihwkit.simulator.parameters.io import IOParametersIRDropT

# Input/Output quantization
precision_i = 32
precision_o =  32

# Crossbar array size
crossbar_size = 64

# Program-verify acceptance range
acceptance_range= 0.2 # either 2% or 0.2%

# Amount of vectors 
batches = 100

# Set random seed
#random.seed(0)
 
# initial weights
matrix = rand(crossbar_size, crossbar_size)*2-1

# probe vectors
input_probe = rand(crossbar_size, batches)*2-1
real_MVM = matmul(input_probe.T, matrix.T)
#rmse_base = sqrt(mean((analog_MVM - real_MVM)**2))

# time of infernence ir 0s, 1s, 1h, 1d and 10y
t_inferences = [0, 1, 3600, 3600*24, 3600*24*365*10] 
drifted_MVM = zeros((len(t_inferences), crossbar_size, batches))

# Wire resistance in Ohms
wire = 0.35
def gen_rpu_config():
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

    rpu_config.noise_model = ReRamCMONoiseModel(g_max = 88.19, g_min = 9.0, acceptance_range=acceptance_range)
    rpu_config.drift_compensation = None
    return rpu_config

model = nn.Linear(crossbar_size,crossbar_size, bias=False)
model.weight.data = matrix
rpu_config = gen_rpu_config()
analog_model = convert_to_analog(model,rpu_config)

#for analog_tile in analog_model.analog_tiles():
    ##if isinstance(analog_tile, InferenceTileWithPeriphery):
        #analog_tile.programmed_weights = matrix

analog_model = analog_model.eval()
analog_model.program_analog_weights(noise_model =rpu_config.noise_model)
baseline = analog_model(input_probe[:,0])
baseline_FP = matmul(input_probe[:,0], matrix.T)
analog_MVM = zeros((crossbar_size, batches))
# compute the MVM for each time of inference considering conductance relaxation
for i, t_inference in enumerate(t_inferences):
    analog_model.drift_analog_weights(t_inference)
    for j in range(batches):
        drifted_MVM[i,:, j] = analog_model(input_probe[:, j])

drifted_MVM = drifted_MVM.detach()
print(drifted_MVM.shape)
print(real_MVM.shape)
rmse_t0 = sqrt(mean((drifted_MVM[0] - real_MVM.T)**2))
rmse_t1 = sqrt(mean((drifted_MVM[1] - real_MVM.T)**2))
rmse_t2 = sqrt(mean((drifted_MVM[2] - real_MVM.T)**2))
rmse_t3 = sqrt(mean((drifted_MVM[3] - real_MVM.T)**2))
rmse_t4 = sqrt(mean((drifted_MVM[4] - real_MVM.T)**2))
print(f"RMSE t0: {rmse_t0}")
print(f"RMSE t = 1s: {rmse_t1}")
print(f"RMSE t = 1h: {rmse_t2}")
print(f"RMSE t = 24h: {rmse_t3}")
print(f"RMSE t = 10y: {rmse_t4}")
#if SAVE_FILES == True:
    #for i in range(drifted_MVM.shape[0]):
        #np.save(f"/Users/mvc/Library/CloudStorage/Box-Box/unicos/Donato_MVM_tol02/t_{t_inferences[i]}_sec_prec_{precision_i}-{precision_o}bit_{runs}runs.npy", drifted_MVM[i])
    #np.save(f"/Users/mvc/Library/CloudStorage/Box-Box/unicos/Donato_MVM_tol02/baseline_MVM.npy", real_MVM)

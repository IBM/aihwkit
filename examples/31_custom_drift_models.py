# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 31: customized conductance drift models

Simple 1-layer network that demonstrates user-defined drift model
capability and impact on output over time due to drift.

Reference paper evaluating a number of different conductance-dependent
drift models exhibiting complex characteristics:
https://onlinelibrary.wiley.com/doi/full/10.1002/aelm.202201190
"""
# pylint: disable=invalid-name
from numpy import asarray

# Imports from PyTorch.
from torch import (
    zeros,
    ones,
    mean,
    std,
    linspace,
)
import matplotlib.pyplot as plt

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.rpu_base import cuda
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter
from aihwkit.inference.noise.pcm import CustomDriftPCMLikeNoiseModel
from aihwkit.simulator.parameters.enums import BoundManagementType
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.configs import TorchInferenceRPUConfig

g_min, g_max = 0.0, 25.
# define custom drift model
custom_drift_model = dict(g_lst=[g_min, 10., g_max],
                          nu_mean_lst=[0.08, 0.05, 0.03],
                          nu_std_lst=[0.03, 0.02, 0.01])

t_inference_times = [1,                         # 1 sec
                     60,                        # 1 min
                     60 * 60,                   # 1 hour
                     24 * 60 * 60,              # 1 day
                     30 * 24 * 60 * 60,         # 1 month
                     12 * 30 * 24 * 60 * 60,    # 1 year
                     ]

IN_FEATURES = 512
OUT_FEATURES = 512
BATCH_SIZE = 1

# define rpu_config
io_params = IOParameters(
    bound_management=BoundManagementType.NONE,
    nm_thres=1.0,
    inp_res=2 ** 8 - 2,
    out_bound=-1,
    out_res=-1,
    out_noise=0.0)

noise_model = CustomDriftPCMLikeNoiseModel(custom_drift_model,
                                           prog_noise_scale=0.0,   # turn off to show drift only
                                           read_noise_scale=0.0,   # turn off to show drift only
                                           drift_scale=1.0,
                                           g_converter=SinglePairConductanceConverter(g_min=g_min,
                                                                                      g_max=g_max),
                                           )

rpu_config = TorchInferenceRPUConfig(noise_model=noise_model, forward=io_params)

# define simple model, weights, and activations
model = AnalogLinear(IN_FEATURES, OUT_FEATURES, bias=False, rpu_config=rpu_config)
weights = linspace(custom_drift_model['g_lst'][0],
                   custom_drift_model['g_lst'][-1],
                   OUT_FEATURES).repeat(IN_FEATURES, 1)
x = ones(BATCH_SIZE, IN_FEATURES)

# set weights
for name, layer in model.named_analog_layers():
    layer.set_weights(weights.T, zeros(OUT_FEATURES))

# Move the model and tensors to cuda if it is available
if cuda.is_compiled():
    x = x.cuda()
    model = model.cuda()

model.eval()
model.drift_analog_weights(t_inference_times[0])  # generate drift (nu) coefficients

# Extract drift coefficients nu as a function of conductance
g_lst, _ = rpu_config.noise_model.g_converter.convert_to_conductances(weights)
nu_lst = model.analog_module.drift_noise_parameters

# Get mean and std drift coefficient (nu) as function of conductance
gs = mean(g_lst[0], dim=0).numpy()
nu_mean = mean(nu_lst[0].T, dim=0).numpy()
nu_std = std(nu_lst[0].T, dim=0).numpy()

# Plot device drift model
plt.figure()
plt.plot(gs, nu_mean)
plt.fill_between(gs, nu_mean - nu_std, nu_mean + nu_std, alpha=0.2)
plt.xlabel(r"$Conductance \ [\mu S]$")
plt.ylabel(r"$\nu \ [1]$")
plt.tight_layout()
plt.savefig('custom_drift_model.png')
plt.close()

# create simple linear layer model
model = AnalogLinear(IN_FEATURES, OUT_FEATURES, bias=False, rpu_config=rpu_config)

# define weights, activations
weights = (1. / 512.) * ones(IN_FEATURES, OUT_FEATURES)
x = ones(BATCH_SIZE, IN_FEATURES)

# set weights
for _, layer in model.named_analog_layers():
    layer.set_weights(weights.T, zeros(OUT_FEATURES))

# Move the model and tensors to cuda if it is available
if cuda.is_compiled():
    x = x.cuda()
    model = model.cuda()

# Eval model at different time steps
model.eval()
out_lst = []
for t_inference in t_inference_times:
    model.drift_analog_weights(t_inference)  # generate new nu coefficients at each time step
    out = model(x)
    out_lst.append(out)

# Plot drift compensated outputs as a function of time
t = asarray(t_inference_times)
out_mean = asarray([mean(out).detach().cpu().numpy() for out in out_lst])
out_std = asarray([std(out).detach().cpu().numpy() for out in out_lst])
plt.figure()
plt.plot(t, out_mean)
plt.fill_between(t, out_mean - out_std, out_mean + out_std, alpha=0.2)
plt.xscale("log")
plt.xticks(t, ['1 sec', '1 min', '1 hour', '1 day', '1 month', '1 year'], rotation='vertical')
plt.xlabel(r"$Time$")
plt.ylabel(r"$Drift \ Compensated \ Outputs \ [1]$")
plt.tight_layout()
plt.savefig('custom_drift_model_output.png')
plt.close()

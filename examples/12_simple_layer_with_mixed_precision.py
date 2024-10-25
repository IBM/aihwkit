# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 12: simple network with one layer using Mixed Precision learning rule.

Mixed precision is based on the paper Nandakumar et al (2020) (see
https://www.frontiersin.org/articles/10.3389/fnins.2020.00406/full).
"""
# pylint: disable=invalid-name

# Imports from PyTorch.
from torch import Tensor
from torch.nn.functional import mse_loss

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    DigitalRankUpdateRPUConfig,
    MixedPrecisionCompound,
    SoftBoundsDevice,
)
from aihwkit.simulator.rpu_base import cuda

# Prepare the datasets (input and expected output).
x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = Tensor([[1.0, 0.5], [0.7, 0.3]])

# Select the device model to use in the training. While one can use a
# presets as well, we here build up the RPU config from more basic
# devices. We use the relevant RPU config for using a digital rank
# update and transfer to analog device (like in mixed precision) and
# set it to a mixed precision compound which in turn uses a
# ConstantStep analog device:
rpu_config = DigitalRankUpdateRPUConfig(device=MixedPrecisionCompound(device=SoftBoundsDevice()))

# print the config (default values are omitted)
print("\nPretty-print of non-default settings:\n")
print(rpu_config)

model = AnalogLinear(4, 2, bias=True, rpu_config=rpu_config)

# print module structure
print("\nModule structure:\n")
print(model)

# a more detailed printout of the instantiated
print("\nC++ RPUCudaTile information:\n")
print(next(model.analog_tiles()).tile)

# Move the model and tensors to cuda if it is available.
if cuda.is_compiled():
    x = x.cuda()
    y = y.cuda()
    model.cuda()

# Define an analog-aware optimizer, preparing it for using the layers.
opt = AnalogSGD(model.parameters(), lr=0.1)
opt.regroup_param_groups(model)

for epoch in range(500):
    # Delete old gradients
    opt.zero_grad()
    # Add the training Tensor to the model (input).
    pred = model(x)
    # Add the expected output Tensor.
    loss = mse_loss(pred, y)
    # Run training (backward propagation).
    loss.backward()

    opt.step()
    print("{}: Loss error: {:.16f}".format(epoch, loss), end="\r" if epoch % 50 else "\n")

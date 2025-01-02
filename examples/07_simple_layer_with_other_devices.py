# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 7: simple network with one layer using other devices.

Simple network that consist of one analog layer. The network aims to learn
to sum all the elements from one array.
"""
# pylint: disable=invalid-name

# Imports from PyTorch.
from torch import Tensor
from torch.nn.functional import mse_loss

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    UnitCellRPUConfig,
    ConstantStepDevice,
    VectorUnitCell,
    VectorUnitCellUpdatePolicy,
)
from aihwkit.simulator.rpu_base import cuda

# Prepare the datasets (input and expected output).
x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = Tensor([[1.0, 0.5], [0.7, 0.3]])

# Define a single-layer network, using a vector device having multiple
# devices per crosspoint. Each device can be arbitrarily defined

rpu_config = UnitCellRPUConfig()
# 3 arbitrary single unit cell devices (of the same type) per cross-point.
rpu_config.device = VectorUnitCell(
    unit_cell_devices=[
        ConstantStepDevice(w_max=0.3),
        ConstantStepDevice(w_max_dtod=0.4),
        ConstantStepDevice(up_down_dtod=0.1),
    ]
)

# Only one of the devices should receive a single update.
# That is selected randomly, the effective weights is the sum of all
# weights.
rpu_config.device.update_policy = VectorUnitCellUpdatePolicy.SINGLE_RANDOM


model = AnalogLinear(4, 2, bias=True, rpu_config=rpu_config)

print(rpu_config)

# Move the model and tensors to cuda if it is available.
if cuda.is_compiled():
    x = x.cuda()
    y = y.cuda()
    model.cuda()

# Define an analog-aware optimizer, preparing it for using the layers.
opt = AnalogSGD(model.parameters(), lr=0.1)
opt.regroup_param_groups(model)

for epoch in range(100):
    # Delete old gradient
    opt.zero_grad()
    # Add the training Tensor to the model (input).
    pred = model(x)
    # Add the expected output Tensor.
    loss = mse_loss(pred, y)
    # Run training (backward propagation).
    loss.backward()

    opt.step()
    print("Loss error: {:.16f}".format(loss))

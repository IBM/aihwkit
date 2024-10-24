# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 9: simple network with one layer using
deterministic pulse trains for update.

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
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice, PulseType
from aihwkit.simulator.rpu_base import cuda

# Prepare the datasets (input and expected output).
x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = Tensor([[1.0, 0.5], [0.7, 0.3]])

# Define a single-layer network, using a constant step device type.
rpu_config = SingleRPUConfig(device=ConstantStepDevice())
rpu_config.update.pulse_type = PulseType.DETERMINISTIC_IMPLICIT
rpu_config.update.desired_bl = 10  # max number in this case
rpu_config.update.update_bl_management = True  # will vary up to 10 on demand
rpu_config.update.d_res_implicit = 0.1  # effective resolution of x bit lines
rpu_config.update.x_res_implicit = 0.1  # effective resolution of d bit lines

model = AnalogLinear(4, 2, bias=True, rpu_config=rpu_config)
print(model)
# Move the model and tensors to cuda if it is available.
if cuda.is_compiled():
    x = x.cuda()
    y = y.cuda()
    model.cuda()

# Define an analog-aware optimizer, preparing it for using the layers.
opt = AnalogSGD(model.parameters(), lr=0.1)
opt.regroup_param_groups(model)

for epoch in range(100):
    # Delete old gradients
    opt.zero_grad()
    # Add the training Tensor to the model (input).
    pred = model(x)
    # Add the expected output Tensor.
    loss = mse_loss(pred, y)
    # Run training (backward propagation).
    loss.backward()

    opt.step()
    print("Loss error: {:.16f}".format(loss))

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

"""aihwkit example 1: simple network with one layer.

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
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import JARTv1bDevice, SoftBoundsDevice, ConstantStepDevice, LinearStepDevice
from aihwkit.simulator.configs.utils import PulseType
# from aihwkit.simulator.rpu_base import cuda

# # Prepare the datasets (input and expected output).
# x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
# y = Tensor([[1.0, 0.5], [0.7, 0.3]])

# # Define a single-layer network, using a constant step device type.
# rpu_config = SingleRPUConfig(device=JARTv1bDevice())
# model = AnalogLinear(4, 2, bias=True,
#                      rpu_config=rpu_config)

# Prepare the datasets (input and expected output).
slope = 0.5
x = Tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])
y = Tensor([[0.0], [slope], [2*slope], [3*slope], [4*slope]])

# Define a single-layer network, using a constant step device type.
rpu_config = SingleRPUConfig(device=JARTv1bDevice(pulse_length=1, base_time_step=1e-2, pulse_voltage_SET=-20, pulse_voltage_RESET=20))
# rpu_config = SingleRPUConfig(device=SoftBoundsDevice())

# rpu_config.update.pulse_type = PulseType.DETERMINISTIC_IMPLICIT
rpu_config.update.desired_bl = 100  # max number in this case
model = AnalogLinear(1, 1, bias=False,
                     rpu_config=rpu_config)

# Move the model and tensors to cuda if it is available.
# if cuda.is_compiled():
#     x = x.cuda()
#     y = y.cuda()
#     model.cuda()

# Define an analog-aware optimizer, preparing it for using the layers.
opt = AnalogSGD(model.parameters(), lr=0.01)
opt.regroup_param_groups(model)

weights = model.get_weights()[0][0][0]
count = 0
print('Epoch%s:weights: {:.24f}'.format(weights)%count)

for epoch in range(10):
    # Add the training Tensor to the model (input).
    pred = model(x)
    # Add the expected output Tensor.
    loss = mse_loss(pred, y)
    # Run training (backward propagation).
    loss.backward()

    opt.step()
    # print('Loss error: {:.16f}'.format(loss))
    weights = model.get_weights()[0][0][0]
    count = count + 1
    print('Epoch%s:weights: {:.24f}'.format(weights)%count)
    # if model.get_weights()[0][0][0] != weights:
    #     weights = model.get_weights()[0][0][0]
    #     count = count + 1
    #     print('Epoch%s:weights: {:.24f}'.format(weights)%count)

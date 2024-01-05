# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""aihwkit example 2: network with multiple layers.

Network that consists of multiple analog layers. It aims to learn to sum all
the elements from one array.
"""
# pylint: disable=invalid-name

# Imports from PyTorch.
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.nn import Sequential

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice

# Prepare the datasets (input and expected output).
x_b = Tensor([[0.1, 0.2, 0.0, 0.0], [0.2, 0.4, 0.0, 0.0]])
y_b = Tensor([[0.3], [0.6]])

# Define a multiple-layer network, using a constant step device type.
model = Sequential(
    AnalogLinear(4, 2, rpu_config=SingleRPUConfig(device=ConstantStepDevice())),
    AnalogLinear(2, 2, rpu_config=SingleRPUConfig(device=ConstantStepDevice())),
    AnalogLinear(2, 1, rpu_config=SingleRPUConfig(device=ConstantStepDevice())),
)

# Define an analog-aware optimizer, preparing it for using the layers.
opt = AnalogSGD(model.parameters(), lr=0.5)
opt.regroup_param_groups(model)

for epoch in range(100):
    opt.zero_grad()

    # Add the training Tensor to the model (input).
    pred = model(x_b)
    # Add the expected output Tensor.
    loss = mse_loss(pred, y_b)
    # Run training (backward propagation).
    loss.backward()

    opt.step()

    print("Loss error: {:.16f}".format(loss))

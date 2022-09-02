
# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021 IBM. All Rights Reserved.
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
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import JARTv1bDevice
from aihwkit.simulator.rpu_base import cuda

import wandb

project_name = "debug"
# project_name = "basic test v1b"
learning_rate=0.1
epochs = 100
CUDA_Enabled = False


if cuda.is_compiled() & CUDA_Enabled:
    wandb.init(project=project_name, group="Multi-Layer Perceptron", job_type="CUDA")
else:
    
    wandb.init(project=project_name, group="Multi-Layer Perceptron", job_type="CPU")

wandb.config = {
  "learning_rate": learning_rate,
  "epochs": epochs,
  "CUDA_Enabled": CUDA_Enabled
}

# Prepare the datasets (input and expected output).
x_b = Tensor([[0.1, 0.2, 0.0, 0.0], [0.2, 0.4, 0.0, 0.0]])
y_b = Tensor([[0.3], [0.6]])

# Define a multiple-layer network, using a constant step device type.
model = Sequential(
    AnalogLinear(4, 2, rpu_config=SingleRPUConfig(device=JARTv1bDevice())),
    AnalogLinear(2, 2, rpu_config=SingleRPUConfig(device=JARTv1bDevice())),
    AnalogLinear(2, 1, rpu_config=SingleRPUConfig(device=JARTv1bDevice()))
)

# Move the model and tensors to cuda if it is available.
if cuda.is_compiled() & CUDA_Enabled:
    x_b = x_b.cuda()
    y_b = y_b.cuda()
    model.cuda()

# Define an analog-aware optimizer, preparing it for using the layers.
opt = AnalogSGD(model.parameters(), lr=learning_rate)
opt.regroup_param_groups(model)

for epoch in range(epochs):
    # Add the training Tensor to the model (input).
    pred = model(x_b)
    # Add the expected output Tensor.
    loss = mse_loss(pred, y_b)
    wandb.log({"loss": loss})
    # Run training (backward propagation).
    loss.backward()

    opt.step()

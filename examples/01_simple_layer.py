# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

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
from aihwkit.simulator.configs import SingleRPUConfig, IdealDevice, ConstantStepDevice
from aihwkit.simulator.rpu_base import cuda
from aihwkit.simulator.parameters import PulseType

# Prepare the datasets (input and expected output).
x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = Tensor([[1.0, 0.5], [0.7, 0.3]])

# Define a single-layer network, using a constant step device type.
device = ConstantStepDevice()
# TODO: Once Python binding is ready, set hs_decay like this:
# device.hs_decay = 0.8  # 20% weight decay for specific HS transitions

rpu_config = SingleRPUConfig(device=device)
rpu_config.update.pulse_type=PulseType.STOCHASTIC_STREAM
rpu_config.update.desired_bl=10

print("HS decay will be applied with default value (0.95) for transitions:")

print(rpu_config.device.dw_min)
model = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config)
 
# Enable HS tracking
model.analog_module.tile.enable_hs_tracking()
print("HS tracking enabled")

# Move the model and tensors to cuda if it is available.
if cuda.is_compiled():
    x = x.cuda()
    y = y.cuda()
    model = model.cuda()

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

    # Get HS counts after update
    hs_counts = model.analog_module.tile.get_hs_transition_counts()
    total_hs_counts = hs_counts.sum().item()

    print("Epoch {}: Loss error: {:.6f}, Total HS counts: {}".format(epoch, loss, total_hs_counts))

    # Print detailed HS counts every 10 epochs
    if epoch % 10 == 0 and total_hs_counts > 0:
        print("  Detailed HS transition counts:")
        for i in range(min(16, len(hs_counts))):
            if hs_counts[i] > 0:
                print(f"    Transition {i}: {hs_counts[i].item()}")


# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details..

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
from aihwkit.simulator.parameters import PulseType

# Prepare the datasets (input and expected output).
x_b = Tensor([[0.1, 0.2, 0.0, 0.0], [0.2, 0.4, 0.0, 0.0]])
y_b = Tensor([[0.3], [0.6]])
rpu_config=SingleRPUConfig(device=ConstantStepDevice())
rpu_config.update.pulse_type=PulseType.HALFSELECTED_STOCHASTIC
# Define a multiple-layer network, using a constant step device type.
model = Sequential(
    AnalogLinear(4, 2, bias=False, rpu_config=rpu_config),
    AnalogLinear(2, 2, bias=False, rpu_config=rpu_config),
    AnalogLinear(2, 1, bias=False, rpu_config=rpu_config),
)

# Enable HS tracking for the first layer only
#model[0].analog_module.tile.enable_hs_tracking()
#model[1].analog_module.tile.enable_hs_tracking()
#model[2].analog_module.tile.enable_hs_tracking()
print("HS tracking enabled for first layer")

# Define an analog-aware optimizer, preparing it for using the layers.
opt = AnalogSGD(model.parameters(), lr=0.05)
opt.regroup_param_groups(model)

for epoch in range(200):
    opt.zero_grad()

    # Add the training Tensor to the model (input).
    pred = model(x_b)
    # Add the expected output Tensor.
    loss = mse_loss(pred, y_b)
    # Run training (backward propagation).
    loss.backward()

    opt.step()

    # Get HS counts from first layer after update
    hs_counts = model[0].analog_module.tile.get_hs_transition_counts()
    total_hs_counts = hs_counts.sum().item()

    print("Epoch {}: Loss error: {:.6f}, First layer HS counts: {}".format(epoch, loss, total_hs_counts))

    # Print detailed HS counts every 20 epochs for first layer
    if epoch % 20 == 0 and total_hs_counts > 0:
        print("  First layer detailed HS transition counts:")
        for i in range(min(16, len(hs_counts))):
            if hs_counts[i] > 0:
                print(f"    Transition {i}: {hs_counts[i].item()}")

# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 8: simple network with one layer using Tiki-taka learning rule.

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
from aihwkit.simulator.configs import UnitCellRPUConfig, TransferCompound, SoftBoundsDevice
from aihwkit.simulator.rpu_base import cuda

# Prepare the datasets (input and expected output).
x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = Tensor([[1.0, 0.5], [0.7, 0.3]])

# The Tiki-taka learning rule can be implemented using the transfer device.
rpu_config = UnitCellRPUConfig(
    device=TransferCompound(
        # Devices that compose the Tiki-taka compound.
        unit_cell_devices=[
            SoftBoundsDevice(w_min=-0.3, w_max=0.3),
            SoftBoundsDevice(w_min=-0.6, w_max=0.6),
        ],
        # Make some adjustments of the way Tiki-Taka is performed.
        units_in_mbatch=True,  # batch_size=1 anyway
        transfer_every=2,  # every 2 batches do a transfer-read
        n_reads_per_transfer=1,  # one forward read for each transfer
        gamma=0.0,  # all SGD weight in second device
        scale_transfer_lr=True,  # in relative terms to SGD LR
        transfer_lr=1.0,  # same transfer LR as for SGD
        fast_lr=0.1,  # SGD update onto first matrix constant
        transfer_columns=True,  # transfer use columns (not rows)
    )
)

# Make more adjustments (can be made here or above).
rpu_config.forward.inp_res = 1 / (2**6 - 2)  # 6 bit DAC

# same backward pass settings as forward
rpu_config.backward = rpu_config.forward

# Same forward/update for transfer-read as for actual SGD.
rpu_config.device.transfer_forward = rpu_config.forward
# SGD update/transfer-update will be done with stochastic pulsing.
rpu_config.device.transfer_update = rpu_config.update

# print the config (default values are omitted)
print("\nPretty-print of non-default settings:\n")
print(rpu_config)

print("\nInfo about all settings:\n")
print(repr(rpu_config))

model = AnalogLinear(4, 2, bias=True, rpu_config=rpu_config)

# a more detailed printout of the instantiated
print("\nInfo about the instantiated C++ tile:\n")
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
    # Delete old gradient
    opt.zero_grad()
    # Add the training Tensor to the model (input).
    pred = model(x)
    # Add the expected output Tensor.
    loss = mse_loss(pred, y)
    # Run training (backward propagation).
    loss.backward()

    opt.step()
    print("{}: Loss error: {:.16f}".format(epoch, loss), end="\r" if epoch % 50 else "\n")

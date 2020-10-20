# -*- coding: utf-8 -*-

# (C) Copyright 2020 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""aihwkit example 7: simple network with one layer using Tiki-taka learning rule.

Simple network that consist of one analog layer. The network aims to learn
to sum all the elements from one array.
"""

# Imports from PyTorch.
from torch import Tensor
from torch.nn.functional import mse_loss

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear
from aihwkit.optim.analog_sgd import AnalogSGD
from aihwkit.simulator.configs import UnitCellRPUConfig
from aihwkit.simulator.configs.devices import (
    TransferCompoundDevice,
    SoftBoundsDevice)
from aihwkit.simulator.rpu_base import cuda

# Prepare the datasets (input and expected output).
x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = Tensor([[1.0, 0.5], [0.7, 0.3]])

# The Tiki-taka learning rule can be implemented using the transfer device.
rpu_config = UnitCellRPUConfig(
    device=TransferCompoundDevice(

        # Devices that compose the Tiki-taka compound.
        unit_cell_devices=[
            SoftBoundsDevice(w_min=-0.3, w_max=0.3),
            SoftBoundsDevice(w_min=-0.6, w_max=0.6)
        ],

        # Make some adjustments of the way Tiki-Taka is performed.
        units_in_mbatch=True,    # batch_size=1 anyway
        transfer_every=2,        # every 2 batches do a transfer-read
        n_cols_per_transfer=1,   # one forward read for each transfer
        gamma=0.0,               # all SGD weight in second device
        scale_transfer_lr=True,  # in relative terms to SGD LR
        transfer_lr=1.0,         # same transfer LR as for SGD
    )
)

# Make more adjustments (can be made here or above).
rpu_config.forward.inp_res = 1/64.  # 6 bit DAC

# Same forward/update for transfer-read as for actual SGD.
rpu_config.device.transfer_forward = rpu_config.forward
# SGD update/transfer-update will be done with stochastic pulsing.
rpu_config.device.transfer_update = rpu_config.update

model = AnalogLinear(4, 2, bias=True, rpu_config=rpu_config)

print(model.analog_tile.tile)

# Move the model and tensors to cuda if it is available.
if cuda.is_compiled():
    x = x.cuda()
    y = y.cuda()
    model.cuda()

# Define an analog-aware optimizer, preparing it for using the layers.
opt = AnalogSGD(model.parameters(), lr=0.1)
opt.regroup_param_groups(model)

for epoch in range(500):
    # Add the training Tensor to the model (input).
    pred = model(x)
    # Add the expected output Tensor.
    loss = mse_loss(pred, y)
    # Run training (backward propagation).
    loss.backward()

    opt.step()
    print('{}: Loss error: {:.16f}'.format(epoch, loss),
          end='\r' if epoch % 50 else '\n')

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
from aihwkit.simulator.configs import DigitalRankUpdateRPUConfig
from aihwkit.simulator.configs.devices import (
    MixedPrecisionCompound,
    SoftBoundsDevice)
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
rpu_config = DigitalRankUpdateRPUConfig(
    device=MixedPrecisionCompound(
        device=SoftBoundsDevice(),
    )
)

# print the config (default values are omitted)
print('\nPretty-print of non-default settings:\n')
print(rpu_config)

print('\nInfo about all settings:\n')
print(repr(rpu_config))

model = AnalogLinear(4, 2, bias=True, rpu_config=rpu_config)

# a more detailed printout of the instantiated
print('\nInfo about the instantiated C++ tile:\n')
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

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

"""aihwkit example 5: simple network hardware-aware training with one layer.

Simple network that consist of one analog layer. The network aims to learn
to sum all the elements from one array.
"""

# Imports from PyTorch.
from torch import Tensor
from torch.nn.functional import mse_loss

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear
from aihwkit.nn.modules.base import drift_analog_weights
from aihwkit.optim.analog_sgd import AnalogSGD
from aihwkit.simulator.configs.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import OutputWeightNoiseType
from aihwkit.simulator.rpu_base import cuda
from aihwkit.simulator.noise_models import PCMLikeNoiseModel

# Prepare the datasets (input and expected output).
x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = Tensor([[1.0, 0.5], [0.7, 0.3]])

# Define a single-layer network, using a constant step device type.
rpu_config = InferenceRPUConfig()
rpu_config.forward.out_res = -1.  # Turn off (output) ADC discretization.
rpu_config.forward.w_noise_type = OutputWeightNoiseType.ADDITIVE_CONSTANT
rpu_config.forward.w_noise = 0.02
rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)

model = AnalogLinear(4, 2, bias=True,
                     rpu_config=rpu_config)

# Move the model and tensors to cuda if it is available.
if cuda.is_compiled():
    x = x.cuda()
    y = y.cuda()
    model.cuda()

# Define an analog-aware optimizer, preparing it for using the layers.
opt = AnalogSGD(model.parameters(), lr=0.1)
opt.regroup_param_groups(model)

print(model.analog_tile.tile)

for epoch in range(100):
    # Add the training Tensor to the model (input).
    pred = model(x)
    # Add the expected output Tensor.
    loss = mse_loss(pred, y)
    # Run training (backward propagation).
    loss.backward()

    opt.step()
    print('Loss error: {:.16f}'.format(loss))

model.eval()

# Do inference with drift.
pred_before = model(x)

print('Correct value:\t {}'.format(y.detach().cpu().numpy().flatten()))
print('Prediction after training:\t {}'.format(pred_before.detach().cpu().numpy().flatten()))

for t_inference in [0., 1., 20., 1000., 1e5]:
    drift_analog_weights(model, t_inference)
    pred_drift = model(x)
    print('Prediction after drift (t={}, correction={:1.3f}):\t {}'.format(
        t_inference, model.analog_tile.alpha.cpu().numpy(),
        pred_drift.detach().cpu().numpy().flatten()))

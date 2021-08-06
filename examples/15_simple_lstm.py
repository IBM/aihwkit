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

"""aihwkit example 15: hardware-aware training of analog LSTM model.

This experiment performs hardware-aware training of an analog LSTM on
a simple temporal sequence. The experiment plots training perplexity,
inference results on the test dataset using analog hardware, and inference
results over time using analog hardware and drift compensation.
"""
# pylint: disable=invalid-name

import os
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

# Imports from PyTorch.
import torch
from torch import nn

# Imports from aihwkit.
from aihwkit.nn import AnalogLSTM
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import (
    WeightNoiseType, WeightClipType, WeightModifierType)
from aihwkit.simulator.presets import GokmenVlasovPreset
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.nn import AnalogLinear, AnalogSequential

LEARNING_RATE = 0.05
NUM_LAYERS = 1
INPUT_SIZE = 1
EMBED_SIZE = 20
HIDDEN_SIZE = 50
OUTPUT_SIZE = 1
DROPOUT_RATIO = 0.0
NOISE = 0.0

EPOCHS = 100
BATCH_SIZE = 5
SEQ_LEN = 501
WITH_EMBEDDING = False  # LSTM with embedding
USE_ANALOG_TRAINING = False  # or hardware-aware training

if USE_ANALOG_TRAINING:
    # Define a RPU configuration for analog training
    rpu_config = SingleRPUConfig(device=GokmenVlasovPreset())

else:
    # Define an RPU configuration using inference/hardware-aware training tile
    rpu_config = InferenceRPUConfig()
    rpu_config.forward.out_res = -1.  # Turn off (output) ADC discretization.
    rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
    rpu_config.forward.w_noise = 0.02  # Short-term w-noise.

    rpu_config.clip.type = WeightClipType.FIXED_VALUE
    rpu_config.clip.fixed_value = 1.0
    rpu_config.modifier.pdrop = 0.03  # Drop connect.
    rpu_config.modifier.type = WeightModifierType.ADD_NORMAL  # Fwd/bwd weight noise.
    rpu_config.modifier.std_dev = 0.1
    rpu_config.modifier.rel_to_actual_wmax = True

    # Inference noise model.
    rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)

    # drift compensation
    rpu_config.drift_compensation = GlobalDriftCompensation()


# Path to store results
RESULTS = os.path.join(os.getcwd(), 'results', 'LSTM')
os.makedirs(RESULTS, exist_ok=True)

# Make dataset
x = torch.linspace(0, 8*np.pi, SEQ_LEN)
y = torch.sin(x)*torch.cos(0.5*x) + 0.5
y_in_1d = y[0:SEQ_LEN-1]
y_out_1d = y[1:SEQ_LEN]

y_in_2d, y_out_2d = [], []
for i in range(BATCH_SIZE):
    y_in_2d.append(torch.roll(y_in_1d, shifts=100*i, dims=0) + NOISE*torch.rand(y_in_1d.shape))
    y_out_2d.append(torch.roll(y_out_1d, shifts=100*i, dims=0) + NOISE*torch.rand(y_out_1d.shape))
y_in = torch.stack(y_in_2d, dim=0).transpose(0, 1).unsqueeze(2)
y_out = torch.stack(y_out_2d, dim=0).transpose(0, 1).unsqueeze(2)


# Various LSTM Network definitions
class AnalogLSTMNetwork(AnalogSequential):
    """Analog LSTM Network definition using AnalogLinear for embedding and decoder."""

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(DROPOUT_RATIO)
        self.embedding = AnalogLinear(INPUT_SIZE, EMBED_SIZE, rpu_config=rpu_config)
        self.lstm = AnalogLSTM(EMBED_SIZE, HIDDEN_SIZE, num_layers=1,
                               dropout=DROPOUT_RATIO, bias=True,
                               rpu_config=rpu_config)
        self.decoder = AnalogLinear(HIDDEN_SIZE, OUTPUT_SIZE, bias=True)

    def forward(self, x_in, in_states):  # pylint: disable=arguments-differ
        embed = self.dropout(self.embedding(x_in))
        out, out_states = self.lstm(embed, in_states)
        out = self.dropout(self.decoder(out))
        return out, out_states


class AnalogLSTMNetwork_noEmbedding(AnalogSequential):
    """Analog LSTM Network definition without embedding layer and using AnalogLinear for decoder."""

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(DROPOUT_RATIO)
        self.lstm = AnalogLSTM(INPUT_SIZE, HIDDEN_SIZE, num_layers=1,
                               dropout=DROPOUT_RATIO, bias=True,
                               rpu_config=rpu_config)
        self.decoder = AnalogLinear(HIDDEN_SIZE, OUTPUT_SIZE, bias=True,
                                    rpu_config=rpu_config)

    def forward(self, x_in, in_states):  # pylint: disable=arguments-differ
        """ Forward pass """
        out, out_states = self.lstm(x_in, in_states)
        out = self.dropout(self.decoder(out))
        return out, out_states


def reset_states():
    """Reset the LSTM states."""
    LSTMState = namedtuple('LSTMState', ['hx', 'cx'])
    out_states = [LSTMState(torch.zeros(BATCH_SIZE, HIDDEN_SIZE),
                            torch.zeros(BATCH_SIZE, HIDDEN_SIZE))
                  for _ in range(NUM_LAYERS)]
    return out_states


if WITH_EMBEDDING:
    model = AnalogLSTMNetwork()
else:
    model = AnalogLSTMNetwork_noEmbedding()

optimizer = AnalogSGD(model.parameters(), lr=LEARNING_RATE)
optimizer.regroup_param_groups(model)
criterion = nn.MSELoss()

# train
losses = []
for i in range(EPOCHS):
    states = reset_states()
    optimizer.zero_grad()
    pred, states = model(y_in, states)

    loss = criterion(pred, y_out)
    print('Epoch = %d: Train Perplexity = %f' % (i, np.exp(loss.detach().numpy())))

    loss.backward()
    optimizer.step()

    losses.append(loss.detach().cpu())

plt.figure()
plt.plot(np.exp(np.asarray(losses)), '-b')
plt.xlabel('# Epochs')
plt.ylabel('Perplexity [1]')
plt.ylim([1.0, 1.4])
plt.savefig(os.path.join(RESULTS, 'train_perplexity'))
plt.close()

# Test.
model.eval()
states = reset_states()
pred, states = model(y_in, states)
loss = criterion(pred, y_out)
print("Test Perplexity = %f" % (np.exp(loss.detach().numpy())))

plt.figure()
plt.plot(y_out[:, 0, 0], '-b')
plt.plot(pred.detach().numpy()[:, 0, 0], '-g')
plt.legend(['truth', 'prediction'])
plt.savefig(os.path.join(RESULTS, 'test'))
plt.close()

# Drift test.
plt.figure()
plt.plot(y_out[:, 0, 0], '-b', label='truth')
for t_inference in [0., 1., 20., 1000., 1e5]:
    model.drift_analog_weights(t_inference)
    states = reset_states()
    pred_drift, states = model(y_in, states)
    plt.plot(pred_drift[:, 0, 0].detach().cpu().numpy(), label='t = ' + str(t_inference) + ' s')
plt.legend()
plt.savefig(os.path.join(RESULTS, 'drift'))
plt.close()

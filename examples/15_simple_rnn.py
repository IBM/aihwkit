# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 15: hardware-aware training of analog RNN model.

This experiment performs hardware-aware training of an analog RNN on
a simple temporal sequence. The experiment plots training perplexity,
inference results on the test dataset using analog hardware, and inference
results over time using analog hardware and drift compensation.
"""
# pylint: disable=invalid-name, arguments-differ, redefined-builtin

import os

import matplotlib.pyplot as plt
import numpy as np

# Imports from PyTorch.
import torch
from torch import nn

# Imports from aihwkit.
from aihwkit.nn import AnalogRNN
from aihwkit.nn import AnalogGRUCell  # or one of AnalogGRUCell, AnalogVanillaRNNCell
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    InferenceRPUConfig,
    WeightNoiseType,
    WeightClipType,
    WeightModifierType,
)
from aihwkit.simulator.presets import GokmenVlasovPreset
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.simulator.rpu_base import cuda

LEARNING_RATE = 0.05
NUM_LAYERS = 1
INPUT_SIZE = 1
EMBED_SIZE = 20
HIDDEN_SIZE = 50
OUTPUT_SIZE = 1
DROPOUT_RATIO = 0.0
NOISE = 0.0

EPOCHS = 50
BATCH_SIZE = 5
SEQ_LEN = 501
RNN_CELL = AnalogGRUCell  # type of RNN cell
WITH_EMBEDDING = True  # RNN with embedding
WITH_BIDIR = False
USE_ANALOG_TRAINING = False  # or hardware-aware training
DEVICE = torch.device("cuda") if cuda.is_compiled() else torch.device("cpu")


if USE_ANALOG_TRAINING:
    # Define a RPU configuration for analog training
    rpu_config = GokmenVlasovPreset()

else:
    # Define an RPU configuration using inference/hardware-aware training tile
    rpu_config = InferenceRPUConfig()
    rpu_config.forward.out_res = -1.0  # Turn off (output) ADC discretization.
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


# Various RNN Network definitions
class AnalogBidirRNNNetwork(AnalogSequential):
    """Analog Bidirectional RNN Network definition using AnalogLinear for
    embedding and decoder."""

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(DROPOUT_RATIO)
        self.embedding = AnalogLinear(INPUT_SIZE, EMBED_SIZE, rpu_config=rpu_config)
        self.rnn = AnalogRNN(
            RNN_CELL,
            EMBED_SIZE,
            HIDDEN_SIZE,
            bidir=True,
            num_layers=1,
            dropout=DROPOUT_RATIO,
            bias=True,
            rpu_config=rpu_config,
        )
        self.decoder = AnalogLinear(2 * HIDDEN_SIZE, OUTPUT_SIZE, bias=True)

    def forward(self, input, in_states=None):
        embed = self.dropout(self.embedding(input))
        out, out_states = self.rnn(embed, in_states)
        out = self.dropout(self.decoder(out))
        return [out, out_states]


class AnalogBidirRNNNetwork_noEmbedding(AnalogSequential):
    """Analog Bidirectional RNN Network definition without embedding layer
    and using AnalogLinear for decoder."""

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(DROPOUT_RATIO)
        self.rnn = AnalogRNN(
            RNN_CELL,
            INPUT_SIZE,
            HIDDEN_SIZE,
            bidir=True,
            num_layers=1,
            dropout=DROPOUT_RATIO,
            bias=True,
            rpu_config=rpu_config,
        )
        self.decoder = AnalogLinear(2 * HIDDEN_SIZE, OUTPUT_SIZE, bias=True, rpu_config=rpu_config)

    def forward(self, input, in_states=None):
        """Forward pass"""
        out, out_states = self.rnn(input, in_states)
        out = self.dropout(self.decoder(out))
        return [out, out_states]


class AnalogRNNNetwork(AnalogSequential):
    """Analog RNN Network definition using AnalogLinear for embedding and
    decoder."""

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(DROPOUT_RATIO)
        self.embedding = AnalogLinear(INPUT_SIZE, EMBED_SIZE, rpu_config=rpu_config)
        self.rnn = AnalogRNN(
            RNN_CELL,
            EMBED_SIZE,
            HIDDEN_SIZE,
            bidir=False,
            num_layers=1,
            dropout=DROPOUT_RATIO,
            bias=True,
            rpu_config=rpu_config,
        )
        self.decoder = AnalogLinear(HIDDEN_SIZE, OUTPUT_SIZE, bias=True)

    def forward(self, input, in_states=None):
        embed = self.dropout(self.embedding(input))
        out, out_states = self.rnn(embed, in_states)
        out = self.dropout(self.decoder(out))
        return [out, out_states]


class AnalogRNNNetwork_noEmbedding(AnalogSequential):
    """Analog RNN Network definition without embedding layer and using AnalogLinear for decoder."""

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(DROPOUT_RATIO)
        self.rnn = AnalogRNN(
            RNN_CELL,
            INPUT_SIZE,
            HIDDEN_SIZE,
            bidir=False,
            num_layers=1,
            dropout=DROPOUT_RATIO,
            bias=True,
            rpu_config=rpu_config,
        )
        self.decoder = AnalogLinear(HIDDEN_SIZE, OUTPUT_SIZE, bias=True, rpu_config=rpu_config)

    def forward(self, input, in_states=None):
        """Forward pass"""
        out, out_states = self.rnn(input, in_states)
        out = self.dropout(self.decoder(out))
        return [out, out_states]


# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "RNN")
os.makedirs(RESULTS, exist_ok=True)

# Make dataset
x = torch.linspace(0, 8 * np.pi, SEQ_LEN, device=DEVICE)
y = torch.sin(x) * torch.cos(0.5 * x) + 0.5
y_in_1d = y[0 : SEQ_LEN - 1]
y_out_1d = y[1:SEQ_LEN]

y_in_2d, y_out_2d = [], []
for i in range(BATCH_SIZE):
    y_in_2d.append(
        torch.roll(y_in_1d, shifts=100 * i, dims=0)
        + NOISE * torch.rand(y_in_1d.shape, device=DEVICE)
    )
    y_out_2d.append(
        torch.roll(y_out_1d, shifts=100 * i, dims=0)
        + NOISE * torch.rand(y_out_1d.shape, device=DEVICE)
    )
y_in = torch.stack(y_in_2d, dim=0).transpose(0, 1).unsqueeze(2)
y_out = torch.stack(y_out_2d, dim=0).transpose(0, 1).unsqueeze(2)

if WITH_EMBEDDING:
    if WITH_BIDIR:
        model = AnalogBidirRNNNetwork()
    else:
        model = AnalogRNNNetwork()
else:
    if WITH_BIDIR:
        model = AnalogBidirRNNNetwork_noEmbedding()
    else:
        model = AnalogRNNNetwork_noEmbedding()

model.to(DEVICE)
optimizer = AnalogSGD(model.parameters(), lr=LEARNING_RATE)
optimizer.regroup_param_groups(model)
criterion = nn.MSELoss()

# train
losses = []
for i in range(EPOCHS):
    optimizer.zero_grad()

    pred, states = model(y_in, None)

    loss = criterion(pred, y_out)
    print("Epoch = %d: Train Perplexity = %f" % (i, np.exp(loss.detach().cpu().numpy())))

    loss.backward()
    optimizer.step()

    losses.append(loss.detach().cpu())

plt.ion()
plt.figure()
plt.plot(np.exp(np.asarray(losses)), "-b")
plt.xlabel("# Epochs")
plt.ylabel("Perplexity [1]")
plt.ylim([1.0, 1.4])

# Test.
model.eval()
pred, states = model(y_in)
loss = criterion(pred, y_out)
print("Test Perplexity = %f" % (np.exp(loss.detach().cpu().numpy())))

plt.figure()
plt.plot(y_out.detach().cpu().numpy()[:, 0, 0], "-b")
plt.plot(pred.detach().cpu().numpy()[:, 0, 0], "-g")
plt.xlabel("x")
plt.ylabel("y")
plt.plot()
plt.legend(["truth", "analog prediction"])

# Drift test.
plt.figure()
plt.plot(y_out.detach().cpu().numpy()[:, 0, 0], "-b", label="truth")
for t_inference in [0.0, 1.0, 20.0, 1000.0, 1e5]:
    model.drift_analog_weights(t_inference)
    pred_drift, states = model(y_in)
    plt.plot(pred_drift.detach().cpu().numpy()[:, 0, 0], label="t = " + str(t_inference) + " s")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()

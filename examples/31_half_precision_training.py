# type: ignore
# pylint: disable-all
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

"""aihwkit example 31: Using half precision training.

This example demonstrates how to use half precision training with aihwkit.

"""
# pylint: disable=invalid-name

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from aihwkit.simulator.configs import InferenceRPUConfig, TorchInferenceRPUConfig
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.parameters.enums import RPUDataType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


if __name__ == "__main__":
    model = Net()
    rpu_config = TorchInferenceRPUConfig()
    model = convert_to_analog(model, rpu_config)
    nll_loss = torch.nn.NLLLoss()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    model = model.to(device=device, dtype=torch.bfloat16)
    optimizer = AnalogSGD(model.parameters(), lr=0.1)
    model = model.train()

    pbar = tqdm.tqdm(enumerate(train_loader))
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device=device, dtype=torch.bfloat16), target.to(
            device=device
        )
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output.float(), target)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss {loss:.4f}")

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

"""aihwkit example 4: analog CNN.

Mnist dataset on a LeNet5 inspired network based on the paper:
https://www.frontiersin.org/articles/10.3389/fnins.2017.00538/full

Learning rates of η = 0.01 for all the epochs with minibatch 8.
"""
# pylint: disable=invalid-name

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# Imports from PyTorch.
import torch
from torch import nn
from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, FloatingPointRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice, FloatingPointDevice
from aihwkit.simulator.rpu_base import cuda

# Check device
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# Path to store datasets
PATH_DATASET = os.path.join('data', 'DATASET')

# Path to store results
RESULTS = os.path.join(os.getcwd(), 'results', 'LENET5')

# Training parameters
SEED = 1
N_EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.01
N_CLASSES = 10

# Select the device model to use in the training.
# * If `SingleRPUConfig(device=ConstantStepDevice())` then analog tiles with
#   constant step devices will be used,
# * If `FloatingPointRPUConfig(device=FloatingPointDevice())` then standard
#   floating point devices will be used
USE_ANALOG_TRAINING = True
if USE_ANALOG_TRAINING:
    RPU_CONFIG = SingleRPUConfig(device=ConstantStepDevice())
else:
    RPU_CONFIG = FloatingPointRPUConfig(device=FloatingPointDevice())


def load_images():
    """Load images for train from torchvision datasets."""

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_data, validation_data


def create_analog_network():
    """Return a LeNet5 inspired analog model."""
    channel = [16, 32, 512, 128]
    model = AnalogSequential(
        AnalogConv2d(in_channels=1, out_channels=channel[0], kernel_size=5, stride=1,
                     rpu_config=RPU_CONFIG),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        AnalogConv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=5, stride=1,
                     rpu_config=RPU_CONFIG),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        nn.Tanh(),
        nn.Flatten(),
        AnalogLinear(in_features=channel[2], out_features=channel[3], rpu_config=RPU_CONFIG),
        nn.Tanh(),
        AnalogLinear(in_features=channel[3], out_features=N_CLASSES, rpu_config=RPU_CONFIG),
        nn.LogSoftmax(dim=1)
    )

    return model


def create_sgd_optimizer(model, learning_rate):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained
        learning_rate (float): global parameter to define learning rate

    Returns:
        nn.Module: Analog optimizer
    """
    optimizer = AnalogSGD(model.parameters(), lr=learning_rate)
    optimizer.regroup_param_groups(model)

    return optimizer


def train_step(train_data, model, criterion, optimizer):
    """Train network.

    Args:
        train_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        optimizer (Optimizer): analog model optimizer

    Returns:
        nn.Module, nn.Module, float:  model, optimizer and loss for per epoch
    """
    total_loss = 0

    model.train()

    for images, labels in train_data:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        # Add training Tensor to the model (input).
        output = model(images)
        loss = criterion(output, labels)

        # Run training (backward propagation).
        loss.backward()

        # Optimize weights.
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    epoch_loss = total_loss / len(train_data.dataset)

    return model, optimizer, epoch_loss


def test_evaluation(validation_data, model, criterion):
    """Test trained network.

    Args:
        validation_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss

    Returns:
        nn.Module, float, float, float:  model, loss, error, and accuracy
    """
    total_loss = 0
    predicted_ok = 0
    total_images = 0

    model.eval()

    for images, labels in validation_data:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        pred = model(images)
        loss = criterion(pred, labels)
        total_loss += loss.item() * images.size(0)

        _, predicted = torch.max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()
        accuracy = predicted_ok/total_images*100
        error = (1-predicted_ok/total_images)*100

    epoch_loss = total_loss / len(validation_data.dataset)

    return model, epoch_loss, error, accuracy


def training_loop(model, criterion, optimizer, train_data, validation_data, epochs, print_every=1):
    """Training loop.

    Args:
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        optimizer (Optimizer): analog model optimizer
        train_data (DataLoader): Validation set to perform the evaluation
        validation_data (DataLoader): Validation set to perform the evaluation
        epochs (int): global parameter to define epochs number
        print_every (int): defines how many times to print training progress

    Returns:
        nn.Module, Optimizer, Tuple: model, optimizer,
            and a tuple of train losses, validation losses, and test
            error
    """
    train_losses = []
    valid_losses = []
    test_error = []

    # Train model
    for epoch in range(0, epochs):
        # Train_step
        model, optimizer, train_loss = train_step(train_data, model, criterion, optimizer)
        train_losses.append(train_loss)

        # Validate_step
        with torch.no_grad():
            model, valid_loss, error, accuracy = test_evaluation(
                validation_data, model, criterion)
            valid_losses.append(valid_loss)
            test_error.append(error)

        if epoch % print_every == (print_every - 1):
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Test error: {error:.2f}%\t'
                  f'Accuracy: {accuracy:.2f}%\t')

    # Save results and plot figures
    np.savetxt(os.path.join(RESULTS, "Test_error.csv"), test_error, delimiter=",")
    np.savetxt(os.path.join(RESULTS, "Train_Losses.csv"), train_losses, delimiter=",")
    np.savetxt(os.path.join(RESULTS, "Valid_Losses.csv"), valid_losses, delimiter=",")
    plot_results(train_losses, valid_losses, test_error)

    return model, optimizer, (train_losses, valid_losses, test_error)


def plot_results(train_losses, valid_losses, test_error):
    """Plot results.

    Args:
        train_losses (List): training losses as calculated in the training_loop
        valid_losses (List): validation losses as calculated in the training_loop
        test_error (List): test error as calculated in the training_loop
    """
    fig = plt.plot(train_losses, 'r-s', valid_losses, 'b-o')
    plt.title('aihwkit LeNet5')
    plt.legend(fig[:2], ['Training Losses', 'Validation Losses'])
    plt.xlabel('Epoch number')
    plt.ylabel('Loss [A.U.]')
    plt.grid(which='both', linestyle='--')
    plt.savefig(os.path.join(RESULTS, 'test_losses.png'))
    plt.close()

    fig = plt.plot(test_error, 'r-s')
    plt.title('aihwkit LeNet5')
    plt.legend(fig[:1], ['Validation Error'])
    plt.xlabel('Epoch number')
    plt.ylabel('Test Error [%]')
    plt.yscale('log')
    plt.ylim((5e-1, 1e2))
    plt.grid(which='both', linestyle='--')
    plt.savefig(os.path.join(RESULTS, 'test_error.png'))
    plt.close()


def main():
    """Train a PyTorch CNN analog model with the MNIST dataset."""
    # Make sure the directory where to save the results exist.
    # Results include: Loss vs Epoch graph, Accuracy vs Epoch graph and vector data.
    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(SEED)

    # Load datasets.
    train_data, validation_data = load_images()

    # Prepare the model.
    model = create_analog_network()
    if USE_CUDA:
        model.cuda()

    print(model)

    print(f'\n{datetime.now().time().replace(microsecond=0)} --- '
          f'Started LeNet5 Example')

    optimizer = create_sgd_optimizer(model, LEARNING_RATE)

    criterion = nn.CrossEntropyLoss()

    model, optimizer, _ = training_loop(model, criterion, optimizer, train_data, validation_data,
                                        N_EPOCHS)

    print(f'{datetime.now().time().replace(microsecond=0)} --- '
          f'Completed LeNet5 Example')


if __name__ == '__main__':
    # Execute only if run as the entry point into the program
    main()

# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 18: resnet32 CNN with CIFAR10.

CIFAR10 dataset on a resnet inspired network based on the paper:
https://arxiv.org/abs/1512.03385
"""
# pylint: disable=invalid-name

# Imports
import os
from datetime import datetime

# Imports from PyTorch.
from torch import nn, Tensor, device, no_grad, manual_seed, save
from torch import max as torch_max
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.optim import AnalogSGD
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.presets import TikiTakaEcRamPreset
from aihwkit.simulator.configs import MappingParameter
from aihwkit.simulator.rpu_base import cuda


# Device to use
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "RESNET")
WEIGHT_PATH = os.path.join(RESULTS, "example_18_model_weight.pth")

# Training parameters
SEED = 1
N_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.1
N_CLASSES = 10

# Device used in the RPU tile
mapping = MappingParameter(weight_scaling_omega=0.6)
RPU_CONFIG = TikiTakaEcRamPreset(mapping=mapping)


class ResidualBlock(nn.Module):
    """Residual block of a residual network with option for the skip connection."""

    def __init__(self, in_ch, hidden_ch, use_conv=False, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_ch)

        if use_conv:
            self.convskip = nn.Conv2d(in_ch, hidden_ch, kernel_size=1, stride=stride)
        else:
            self.convskip = None

    def forward(self, x):
        """Forward pass"""
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.convskip:
            x = self.convskip(x)
        y += x
        return F.relu(y)


def concatenate_layer_blocks(in_ch, hidden_ch, num_layer, first_layer=False):
    """Concatenate multiple residual block to form a layer.

    Returns:
       List: list of layer blocks
    """
    layers = []
    for i in range(num_layer):
        if i == 0 and not first_layer:
            layers.append(ResidualBlock(in_ch, hidden_ch, use_conv=True, stride=2))
        else:
            layers.append(ResidualBlock(hidden_ch, hidden_ch))
    return layers


def create_model():
    """ResNet34 inspired analog model.

    Returns:
       nn.Modules: created model
    """

    block_per_layers = (3, 4, 6, 3)
    base_channel = 16
    channel = (base_channel, 2 * base_channel, 4 * base_channel)

    l0 = nn.Sequential(
        nn.Conv2d(3, channel[0], kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(channel[0]),
        nn.ReLU(),
    )

    l1 = nn.Sequential(
        *concatenate_layer_blocks(channel[0], channel[0], block_per_layers[0], first_layer=True)
    )
    l2 = nn.Sequential(*concatenate_layer_blocks(channel[0], channel[1], block_per_layers[1]))
    l3 = nn.Sequential(*concatenate_layer_blocks(channel[1], channel[2], block_per_layers[2]))
    l4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(channel[2], N_CLASSES))

    return nn.Sequential(l0, l1, l2, l3, l4)


def load_images():
    """Load images for train from torchvision datasets.

    Returns:
        Dataset, Dataset: train data and validation data"""
    mean = Tensor([0.4914, 0.4822, 0.4465])
    std = Tensor([0.2470, 0.2435, 0.2616])

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = datasets.CIFAR10(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.CIFAR10(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_data, validation_data


def create_sgd_optimizer(model, learning_rate):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained
        learning_rate (float): global parameter to define learning rate

    Returns:
        Optimizer: created analog optimizer
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
        nn.Module, Optimizer, float: model, optimizer, and epoch loss
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
    """Test trained network

    Args:
        validation_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss

    Returns:
        nn.Module, float, float, float: model, test epoch loss, test error, and test accuracy
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

        _, predicted = torch_max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()
        accuracy = predicted_ok / total_images * 100
        error = (1 - predicted_ok / total_images) * 100

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
        nn.Module, Optimizer, Tuple: model, optimizer, and a tuple of
            lists of train losses, validation losses, and test error
    """
    train_losses = []
    valid_losses = []
    test_error = []

    # Train model
    for epoch in range(0, epochs):
        # Train_step
        model, optimizer, train_loss = train_step(train_data, model, criterion, optimizer)
        train_losses.append(train_loss)

        if epoch % print_every == (print_every - 1):
            # Validate_step
            with no_grad():
                model, valid_loss, error, accuracy = test_evaluation(
                    validation_data, model, criterion
                )
                valid_losses.append(valid_loss)
                test_error.append(error)

            print(
                f"{datetime.now().time().replace(microsecond=0)} --- "
                f"Epoch: {epoch}\t"
                f"Train loss: {train_loss:.4f}\t"
                f"Valid loss: {valid_loss:.4f}\t"
                f"Test error: {error:.2f}%\t"
                f"Test accuracy: {accuracy:.2f}%\t"
            )

    return model, optimizer


def main():
    """Train a PyTorch CNN analog model with the MNIST dataset."""
    # Make sure the directory where to save the results exist.
    # Results include: Loss vs Epoch graph, Accuracy vs Epoch graph and vector data.
    os.makedirs(RESULTS, exist_ok=True)
    manual_seed(SEED)

    # Load datasets.
    train_data, validation_data = load_images()

    # Load the pytorch model
    model = create_model()

    # Convert the model to its analog version
    model = convert_to_analog(model, RPU_CONFIG)
    # Load saved weights if previously saved
    # model.load_state_dict(load(WEIGHT_PATH))

    if USE_CUDA:
        model.cuda()

    optimizer = create_sgd_optimizer(model, LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{datetime.now().time().replace(microsecond=0)} --- " f"Started ResNet Training")

    model, optimizer = training_loop(
        model, criterion, optimizer, train_data, validation_data, N_EPOCHS
    )

    print(f"{datetime.now().time().replace(microsecond=0)} --- " f"Completed ResNet Training")

    save(model.state_dict(), WEIGHT_PATH)


if __name__ == "__main__":
    # Execute only if run as the entry point into the program
    main()

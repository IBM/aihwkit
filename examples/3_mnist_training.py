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

"""aihwkit example 3: MNIST training.

MNIST training example based on the paper:
https://www.frontiersin.org/articles/10.3389/fnins.2016.00333/full

Uses learning rates of η = 0.01, 0.005, and 0.0025
for epochs 0–10, 11–20, and 21–30, respectively.
"""

from time import time

# Imports from PyTorch.
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear
from aihwkit.optim.analog_sgd import AnalogSGD
from aihwkit.simulator.devices import ConstantStepResistiveDevice

# Path where the datasets will be stored.
TRAIN_DATASET = 'data/TRAIN_DATASET'
TEST_DATASET = 'data/TEST_DATASET'

# Network definition.
INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10

# Training parameters.
EPOCHS = 30
BATCH_SIZE = 64


def load_images():
    """Load images for train from the torchvision datasets."""
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the images.
    train_set = datasets.MNIST(TRAIN_DATASET,
                               download=True, train=True, transform=transform)
    val_set = datasets.MNIST(TEST_DATASET,
                             download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=True)

    return train_data, validation_data


def create_analog_network(input_size, hidden_sizes, output_size):
    """Create the neural network using analog and digital layers.

    Args:
        input_size (int): size of the Tensor at the input.
        hidden_sizes (list): list of sizes of the hidden layers (2 layers).
        output_size (int): size of the Tensor at the output.
    """
    model = nn.Sequential(
        AnalogLinear(input_size, hidden_sizes[0], True,
                     resistive_device=ConstantStepResistiveDevice()),
        nn.Sigmoid(),
        AnalogLinear(hidden_sizes[0], hidden_sizes[1], True,
                     resistive_device=ConstantStepResistiveDevice()),
        nn.Sigmoid(),
        AnalogLinear(hidden_sizes[1], output_size, True,
                     resistive_device=ConstantStepResistiveDevice()),
        nn.LogSoftmax(dim=1)
    )
    print(model)
    return model


def create_sgd_optimizer(model):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained.
    """
    optimizer = AnalogSGD(model.parameters(), lr=0.05)
    optimizer.regroup_param_groups(model)

    return optimizer


def train(model, train_set):
    """Train the network.

    Args:
        model (nn.Module): model to be trained.
        train_set (DataLoader): dataset of elements to use as input for training.
    """
    classifier = nn.NLLLoss()
    optimizer = create_sgd_optimizer(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    time_init = time()
    for epoch_number in range(EPOCHS):
        total_loss = 0
        for images, labels in train_set:
            # Flatten MNIST images into a 784 vector.
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            # Add training Tensor to the model (input).
            output = model(images)
            loss = classifier(output, labels)

            # Run training (backward propagation).
            loss.backward()

            # Optimize weights.
            optimizer.step()

            total_loss += loss.item()

        print('Epoch {} - Training loss: {:.16f}'.format(
            epoch_number, total_loss / len(train_set)))

        # Decay learning rate if needed.
        scheduler.step()

    print('\nTraining Time (s) = {}'.format(time()-time_init))


def test_evaluation(model, val_set):
    """Test trained network

    Args:
        model (nn.Model): Trained model to be evaluated
        val_set (DataLoader): Validation set to perform the evaluation
    """
    # Setup counter of images predicted to 0.
    predicted_ok = 0
    total_images = 0

    for images, labels in val_set:
        # Predict image.
        for i in range(len(labels)):
            image = images[i].view(1, INPUT_SIZE)
            with torch.no_grad():
                pred = model(image)

        probabilities_tensor = torch.exp(pred)
        probabilities = list(probabilities_tensor.numpy()[0])

        # Get labels.
        predicted_label = probabilities.index(max(probabilities))
        validation_label = labels.numpy()[-1]

        # Check if predicted image match with validation label.
        if validation_label == predicted_label:
            predicted_ok += 1
        total_images += 1

    print('\nNumber Of Images Tested = {}'.format(total_images))
    print('Model Accuracy = {}'.format(predicted_ok/total_images))


def main():
    """Train a PyTorch analog model with the MNIST dataset."""
    # Load datasets.
    train_dataset, validation_dataset = load_images()

    # Prepare the model.
    model = create_analog_network(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)

    # Train the model.
    train(model, train_dataset)

    # Evaluate the trained model.
    test_evaluation(model, validation_dataset)


if __name__ == '__main__':
    # Execute only if run as the entry point into the program.
    main()

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

"""aihwkit example 3: MNIST training.

MNIST training example based on the paper:
https://www.frontiersin.org/articles/10.3389/fnins.2016.00333/full

Uses learning rates of η = 0.01, 0.005, and 0.0025
for epochs 0–10, 11–20, and 21–30, respectively.
"""
# pylint: disable=invalid-name

import os

# Imports from PyTorch.
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import JARTv1bDevice
from aihwkit.simulator.rpu_base import cuda

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="YAML Configuration File", action="store_true")
args = parser.parse_args()

if args.config:
    config_file = args.config
else:
    config_file = "noise_free.yml"

import yaml
stream = open(config_file, "r")
config_dictionary = yaml.safe_load(stream)

project_name = config_dictionary["project_name"]
CUDA_Enabled = config_dictionary["CUDA_Enabled"]
USE_wandb = config_dictionary["USE_wandb"]
if USE_wandb:
    import wandb
del config_dictionary["project_name"]
del config_dictionary["CUDA_Enabled"]
del config_dictionary["USE_wandb"]

if "Repeat_Times" in config_dictionary:
    Repeat_Times = config_dictionary["Repeat_Times"]
    del config_dictionary["Repeat_Times"]
else:
    Repeat_Times = 1

# Network definition.
INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10

# Check device
USE_CUDA = 0
if cuda.is_compiled() & CUDA_Enabled:
    USE_CUDA = 1
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# Path where the datasets will be stored.
PATH_DATASET = os.path.join('data', 'DATASET')
def load_images():
    """Load images for train from the torchvision datasets."""
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the images.
    train_set = datasets.MNIST(PATH_DATASET,
                               download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET,
                             download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(
        train_set, batch_size=config_dictionary["batch_size"], shuffle=True)
    validation_data = torch.utils.data.DataLoader(
        val_set, batch_size=config_dictionary["batch_size"], shuffle=True)

    return train_data, validation_data


def create_analog_network(input_size, hidden_sizes, output_size):
    """Create the neural network using analog and digital layers.

    Args:
        input_size (int): size of the Tensor at the input.
        hidden_sizes (list): list of sizes of the hidden layers (2 layers).
        output_size (int): size of the Tensor at the output.

    Returns:
        nn.Module: created analog model
    """
    
    rpu_config = SingleRPUConfig(device=JARTv1bDevice(read_voltage=config_dictionary["pulse_related"]["read_voltage"],
                                                      pulse_voltage_SET=config_dictionary["pulse_related"]["pulse_voltage_SET"],
                                                      pulse_voltage_RESET=config_dictionary["pulse_related"]["pulse_voltage_RESET"],
                                                      pulse_length=config_dictionary["pulse_related"]["pulse_length"],
                                                      base_time_step=config_dictionary["pulse_related"]["base_time_step"],
                                                      w_max_dtod=config_dictionary["noise"]["w_max"]["device_to_device"],
                                                      w_min_dtod=config_dictionary["noise"]["w_min"]["device_to_device"],
                                                      Ndiscmax_dtod=config_dictionary["noise"]["Ndiscmax"]["device_to_device"],
                                                      Ndiscmax_std=config_dictionary["noise"]["Ndiscmax"]["cycle_to_cycle_direct"],
                                                      Ndiscmax_upper_bound=config_dictionary["noise"]["Ndiscmax"]["upper_bound"],
                                                      Ndiscmax_lower_bound=config_dictionary["noise"]["Ndiscmax"]["lower_bound"],
                                                      Ndiscmin_dtod=config_dictionary["noise"]["Ndiscmin"]["device_to_device"],
                                                      Ndiscmin_std=config_dictionary["noise"]["Ndiscmin"]["cycle_to_cycle_direct"],
                                                      Ndiscmin_upper_bound=config_dictionary["noise"]["Ndiscmin"]["upper_bound"],
                                                      Ndiscmin_lower_bound=config_dictionary["noise"]["Ndiscmin"]["lower_bound"],
                                                      ldet_dtod=config_dictionary["noise"]["ldet"]["device_to_device"],
                                                      ldet_std=config_dictionary["noise"]["ldet"]["cycle_to_cycle_direct"],
                                                      ldet_std_slope=config_dictionary["noise"]["ldet"]["cycle_to_cycle_slope"],
                                                      ldet_upper_bound=config_dictionary["noise"]["ldet"]["upper_bound"],
                                                      ldet_lower_bound=config_dictionary["noise"]["ldet"]["lower_bound"],
                                                      rdet_dtod=config_dictionary["noise"]["rdet"]["device_to_device"],
                                                      rdet_std=config_dictionary["noise"]["rdet"]["cycle_to_cycle_direct"],
                                                      rdet_std_slope=config_dictionary["noise"]["rdet"]["cycle_to_cycle_slope"],
                                                      rdet_upper_bound=config_dictionary["noise"]["rdet"]["upper_bound"],
                                                      rdet_lower_bound=config_dictionary["noise"]["rdet"]["lower_bound"]))
    model = AnalogSequential(
        AnalogLinear(input_size, hidden_sizes[0], True,rpu_config),
        nn.Sigmoid(),
        AnalogLinear(hidden_sizes[0], hidden_sizes[1], True,rpu_config),
        nn.Sigmoid(),
        AnalogLinear(hidden_sizes[1], output_size, True,rpu_config),
        nn.LogSoftmax(dim=1)
    )

    if USE_CUDA:
        model.cuda()

    print(model)
    return model


def create_sgd_optimizer(model):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained.
    Returns:
        nn.Module: optimizer
    """
    optimizer = AnalogSGD(model.parameters(), lr=config_dictionary["learning_rate"])
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
    scheduler = StepLR(optimizer, step_size=config_dictionary["scheduler"]["step_size"], gamma=config_dictionary["scheduler"]["gamma"])

    for epoch_number in range(config_dictionary["epochs"]):
        total_loss = 0
        for images, labels in train_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
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

        training_loss = total_loss / len(train_set)
        if USE_wandb:
            wandb.log({"loss": training_loss, "epoch": epoch_number})
        else:
            print('Epoch {} - Training loss: {:.16f}'.format(
                epoch_number, training_loss))

        # Decay learning rate if needed.
        scheduler.step()


def test_evaluation(model, val_set):
    """Test trained network

    Args:
        model (nn.Model): Trained model to be evaluated
        val_set (DataLoader): Validation set to perform the evaluation
    """
    # Setup counter of images predicted to 0.
    predicted_ok = 0
    total_images = 0

    model.eval()

    for images, labels in val_set:
        # Predict image.
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        images = images.view(images.shape[0], -1)
        pred = model(images)

        _, predicted = torch.max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()

    Accuracy = predicted_ok/total_images
    if USE_wandb:
        wandb.log({"Number Of Images Tested": total_images, "Model Accuracy": Accuracy})
    else:
        print('\nNumber Of Images Tested = {}'.format(total_images))
        print('Model Accuracy = {}'.format(Accuracy))


def main():
    """Train a PyTorch analog model with the MNIST dataset."""
    # Load datasets.

    for repeat in range(Repeat_Times):
        if USE_wandb:
            if cuda.is_compiled() & CUDA_Enabled:
                wandb.init(project=project_name, group="Fully Connected MNIST", job_type="CUDA")
                wandb.config.update(config_dictionary)
            else:
                wandb.init(project=project_name, group="Fully Connected MNIST", job_type="CPU")
                wandb.config.update(config_dictionary)
        train_dataset, validation_dataset = load_images()

        # Prepare the model.
        model = create_analog_network(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)

        # Train the model.
        train(model, train_dataset)

        # Evaluate the trained model.
        test_evaluation(model, validation_dataset)

        wandb.finish()


if __name__ == '__main__':
    # Execute only if run as the entry point into the program.
    main()

# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 6: analog CNN with hardware aware training.

Mnist dataset on a LeNet5 inspired network.
"""
# pylint: disable=invalid-name

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# Imports from PyTorch.
from torch import nn, device, manual_seed, no_grad
from torch import max as torch_max
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    RPUDataType,
    InferenceRPUConfig,
    WeightRemapType,
    WeightModifierType,
    WeightClipType,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.inference import PCMLikeNoiseModel
from aihwkit.simulator.rpu_base import cuda

# Check device
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = device("cuda" if USE_CUDA else "cpu")
DATA_TYPE = RPUDataType.FLOAT

# Path to store datasets
PATH_DATASET = os.path.join("data", "DATASET")

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "LENET5")
N_CLASSES = 10


def load_images(batch_size):
    """Load images for train from torchvision datasets.

    Args:
        batch_size (int): dtto

    Returns:
        DataLoader, DataLoader: train data and validation data
    """
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_data = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_data, validation_data


def create_analog_network(rpu_config):
    """Return a LeNet5 inspired analog model.

    Args:
        rpu_config (InferenceRPUConfig): hardware and HWA training settings to use

    Returns:
        nn.Module: lenet analog model
    """
    channel = [16, 32, 512, 128]
    model = AnalogSequential(
        AnalogConv2d(
            in_channels=1, out_channels=channel[0], kernel_size=5, stride=1, rpu_config=rpu_config
        ),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        AnalogConv2d(
            in_channels=channel[0],
            out_channels=channel[1],
            kernel_size=5,
            stride=1,
            rpu_config=rpu_config,
        ),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        nn.Tanh(),
        nn.Flatten(),
        AnalogLinear(in_features=channel[2], out_features=channel[3], rpu_config=rpu_config),
        nn.Tanh(),
        AnalogLinear(in_features=channel[3], out_features=N_CLASSES, rpu_config=rpu_config),
        nn.LogSoftmax(dim=1),
    )

    return model


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


def train_step(data, model, criterion, optimizer):
    """Train network.

    Args:
        data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        optimizer (Optimizer): analog model optimizer

    Returns:
        nn.Module, Optimizer, float: model, optimizer, and epoch loss
    """
    total_loss = 0

    model.train()

    for images, labels in data:
        images = images.to(device=DEVICE, dtype=DATA_TYPE.as_torch())
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
    epoch_loss = total_loss / len(data.dataset)

    return model, optimizer, epoch_loss


@no_grad()
def test_evaluation(data, model, criterion):
    """Test trained network

    Args:
        data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss


    Returns:
        float, float, float: test epoch loss, test error, and test accuracy
    """
    total_loss = 0
    predicted_ok = 0
    total_images = 0

    model.eval()

    for images, labels in data:
        images = images.to(device=DEVICE, dtype=DATA_TYPE.as_torch())
        labels = labels.to(DEVICE)

        pred = model(images)
        loss = criterion(pred, labels)
        total_loss += loss.item() * images.size(0)

        _, predicted = torch_max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()

    accuracy = predicted_ok / total_images * 100
    error = (1 - predicted_ok / total_images) * 100
    epoch_loss = total_loss / len(data.dataset)

    return epoch_loss, error, accuracy


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
            and a tuple of lists of train losses, validation losses, and test
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
        valid_loss, error, accuracy = test_evaluation(validation_data, model, criterion)
        valid_losses.append(valid_loss)
        test_error.append(error)

        if epoch % print_every == (print_every - 1):
            print(
                f"{datetime.now().time().replace(microsecond=0)} --- "
                f"Epoch: {epoch}\t"
                f"Train loss: {train_loss:.4f}\t"
                f"Valid loss: {valid_loss:.4f}\t"
                f"Test error: {error:.2f}%\t"
                f"Accuracy: {accuracy:.2f}%\t"
            )

    # Save results and plot figures
    np.savetxt(os.path.join(RESULTS, "Test_error.csv"), test_error, delimiter=",")
    np.savetxt(os.path.join(RESULTS, "Train_Losses.csv"), train_losses, delimiter=",")
    np.savetxt(os.path.join(RESULTS, "Valid_Losses.csv"), valid_losses, delimiter=",")

    return model, optimizer, (train_losses, valid_losses, test_error)


def plot_results(train_losses, valid_losses, test_error, t_inference_times, inference_test_error):
    """Plot results.

    Args:
        train_losses (List): training losses as calculated in the training_loop
        valid_losses (List): validation losses as calculated in the training_loop
        test_error (List): test error as calculated in the training_loop
        t_inference_times (List): inference times
        inference_test_error (List): Inference test error
    """
    plt.ion()
    plt.figure(figsize=[14, 5])
    plt.subplot(1, 3, 1)
    h = plt.plot(train_losses, "r-s", valid_losses, "b-o")
    plt.title("LeNet5 - HWA training")
    plt.legend(h[:2], ["Training Losses", "Validation Losses"])
    plt.xlabel("Epoch number")
    plt.ylabel("Loss [A.U.]")
    plt.grid(which="both", linestyle="--")

    plt.subplot(1, 3, 2)
    handle = plt.plot(test_error, "r-s")
    plt.title("Test w/o prog. noise and drift")
    plt.legend(handle[:1], ["Validation test error"])
    plt.xlabel("Epoch number")
    plt.ylabel("Test Error [%]")
    plt.yscale("log")
    plt.ylim((5e-1, 1e2))
    plt.grid(which="both", linestyle="--")

    plt.subplot(1, 3, 3)
    handle = plt.plot(t_inference_times, inference_test_error, "r-s")
    plt.title("Eval. w/ prog. noise and drift)")
    plt.legend(handle[:1], ["Validation test error"])
    plt.xlabel("Time of inference [s]")
    plt.ylabel("Test Error [%]")
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim((5e-1, 1e2))
    plt.grid(which="both", linestyle="--")

    plt.show()
    plt.tight_layout()


def training_phase(model, criterion, optimizer, train_data, validation_data):
    """Training phase.

    Args:
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        optimizer (Optimizer): analog model optimizer
        train_data (DataLoader): Validation set to perform the evaluation
        validation_data (DataLoader): Validation set to perform the evaluation

    Returns:
       Tuple: results from the training phase
    """
    print("\n ********************************************************* \n")
    print(f"\n{datetime.now().time().replace(microsecond=0)} --- " f"Started LeNet5 Training")

    model, optimizer, res = training_loop(
        model, criterion, optimizer, train_data, validation_data, N_EPOCHS
    )

    print(f"{datetime.now().time().replace(microsecond=0)} --- " f"Completed LeNet5 Training")

    return res


@no_grad()
def inference_phase(t_inference_times, model, criterion, validation_data):
    """Inference phase.

    Args:
        t_inference_times (list): list of times to do inference
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        validation_data (DataLoader): Validation set to perform the evaluation

    Returns:
       Tuple: results from the training phase
    """
    # pylint: disable=too-many-locals

    _, error_pre, accuracy_pre = test_evaluation(validation_data, model, criterion)
    print(
        f"{datetime.now().time().replace(microsecond=0)} --- "
        f"Error after training: {error_pre:.2f}%\t"
        f"Accuracy after training: {accuracy_pre:.2f}%\t"
    )

    error_lst = []
    accuracy_lst = []

    # Simulation of inference pass at different times after training.
    for t_inference in t_inference_times:
        model.drift_analog_weights(t_inference)

        _, error_post, accuracy_post = test_evaluation(validation_data, model, criterion)

        print(
            f"{datetime.now().time().replace(microsecond=0)} --- "
            f"Error after inference: {error_post:.2f}%\t"
            f"Accuracy after inference: {accuracy_post:.2f}%\t"
            f"Drift t={t_inference: .2e}\t"
        )

        error_lst.append(error_post)
        accuracy_lst.append(accuracy_post)

    return error_lst, accuracy_lst


if __name__ == "__main__":
    # Make sure the directory where to save the results exist.
    # Results include: Loss vs Epoch graph, Accuracy vs Epoch graph and vector data.

    os.makedirs(RESULTS, exist_ok=True)
    manual_seed(1)

    # Training parameters
    N_EPOCHS = 30
    BATCH_SIZE = 50
    LEARNING_RATE = 0.1

    # Load datasets.
    training_data, valid_data = load_images(BATCH_SIZE)

    # Define the properties of the neural network in terms of noise simulated during
    # the inference/training pass
    my_rpu_config = InferenceRPUConfig()
    my_rpu_config.mapping.digital_bias = True
    my_rpu_config.mapping.out_scaling_columnwise = True
    my_rpu_config.mapping.learn_out_scaling = True
    my_rpu_config.mapping.weight_scaling_omega = 1.0
    my_rpu_config.mapping.weight_scaling_columnwise = False
    my_rpu_config.mapping.max_input_size = 512
    my_rpu_config.mapping.max_output_size = 512

    my_rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    my_rpu_config.remap.type = WeightRemapType.CHANNELWISE_SYMMETRIC
    my_rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN
    my_rpu_config.clip.sigma = 2.5

    # train input clipping
    my_rpu_config.forward.noise_management = NoiseManagementType.NONE
    my_rpu_config.forward.bound_management = BoundManagementType.NONE
    my_rpu_config.forward.out_bound = 10.0  # quite restrictive
    my_rpu_config.pre_post.input_range.enable = True
    my_rpu_config.pre_post.input_range.manage_output_clipping = True
    my_rpu_config.pre_post.input_range.decay = 0.001
    my_rpu_config.pre_post.input_range.input_min_percentage = 0.95
    my_rpu_config.pre_post.input_range.output_min_percentage = 0.95

    my_rpu_config.modifier.type = WeightModifierType.ADD_NORMAL
    my_rpu_config.modifier.std_dev = 0.1

    my_rpu_config.runtime.data_type = DATA_TYPE

    # Prepare the model.
    analog_model = create_analog_network(my_rpu_config)
    if USE_CUDA:
        analog_model = analog_model.cuda()
    print(analog_model)

    opt = create_sgd_optimizer(analog_model, LEARNING_RATE)
    crit = nn.CrossEntropyLoss()

    # Train the model
    results = training_phase(analog_model, crit, opt, training_data, valid_data)

    # Test model inference over time
    t_inference_lst = [0.0, 1.0, 20.0, 1000.0, 1e5, 1e7]
    inference_error, _ = inference_phase(t_inference_lst, analog_model, crit, valid_data)
    plot_results(*results, t_inference_lst, inference_error)

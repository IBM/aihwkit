# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 27: Post-training input range calibration.

NOTE: You need to run example 25 first and save the model to RESULTS (see ex. 25).
"""
# pylint: disable=invalid-name

import os

# Imports from PyTorch.
import torch
from torch import nn, manual_seed, no_grad
from torch import max as torch_max
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential
from aihwkit.simulator.configs import (
    InferenceRPUConfig,
    TorchInferenceRPUConfig,
    WeightRemapType,
    WeightModifierType,
    WeightClipType,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.inference import PCMLikeNoiseModel
from aihwkit.inference.calibration import calibrate_input_ranges, InputRangeCalibrationType
from aihwkit.simulator.rpu_base import cuda

# Check device
DEVICE = "cuda" if cuda.is_compiled() else "cpu"

# Path to store datasets
PATH_DATASET = os.path.join("data", "DATASET")
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


@no_grad()
def test_evaluation(data, model):
    """Test trained network

    Args:
        data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated


    Returns:
        float, float, float: test epoch loss, test error, and test accuracy
    """
    predicted_ok = 0
    total_images = 0

    model.eval()

    for images, labels in data:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        pred = model(images)

        _, predicted = torch_max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()

    accuracy = predicted_ok / total_images * 100
    error = (1 - predicted_ok / total_images) * 100
    print(f"Accuracy is {accuracy}%")
    return error, accuracy


class Sampler:
    """Example of a sampler used for calibration."""

    def __init__(self, loader):
        self.loader = iter(loader)
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        x, _ = next(self.loader)
        self.idx += 1
        if self.idx > 10:
            raise StopIteration
        return (x.to(DEVICE), {})  # args and kwargs


# Training parameters
N_EPOCHS = 1
BATCH_SIZE = 50
LEARNING_RATE = 0.1
manual_seed(1)

# Load datasets.
training_data, valid_data = load_images(BATCH_SIZE)


# Define the properties of the neural network in terms of noise simulated during
# the inference/training pass
def populate_rpu_config(rpu_config: InferenceRPUConfig):
    """
    Populate the rpu config fields.

    Args:
        rpu_config (Union[TorchInferenceRPUConfig, InferenceRPUConfig]): The config to populate.

    Returns:
        Union[TorchInferenceRPUConfig, InferenceRPUConfig]: The same rpu config that was passed.
    """
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.out_scaling_columnwise = False
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.max_input_size = 512
    rpu_config.mapping.max_output_size = 512

    rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    rpu_config.remap.type = WeightRemapType.LAYERWISE_SYMMETRIC
    rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN
    rpu_config.clip.sigma = 2.0
    rpu_config.clip.fixed_value = 1.0

    # train input clipping
    rpu_config.forward.is_perfect = False
    rpu_config.forward.noise_management = NoiseManagementType.NONE
    rpu_config.forward.bound_management = BoundManagementType.NONE
    rpu_config.forward.out_bound = 10.0
    rpu_config.forward.inp_bound = 1.0
    rpu_config.forward.out_noise = 0.04
    rpu_config.forward.inp_res = 2**8
    rpu_config.forward.out_res = 2**8

    rpu_config.mapping.max_input_size = 256
    rpu_config.mapping.max_output_size = 256

    rpu_config.modifier.type = WeightModifierType.ADD_NORMAL
    rpu_config.modifier.std_dev = 0.1
    rpu_config.modifier.per_batch_sample = True
    rpu_config.pre_post.input_range.enable = True
    rpu_config.pre_post.input_range.manage_output_clipping = False

    return rpu_config


torch_rpu_config = populate_rpu_config(TorchInferenceRPUConfig())

# Train the model with the purely torch-based tile.
torch_analog_model = create_analog_network(torch_rpu_config)
torch_analog_model.load_state_dict(torch.load(os.path.join(RESULTS, "lenet_torch_tile_model.th")))
torch_analog_model = torch_analog_model.to(DEVICE)

test_evaluation(valid_data, torch_analog_model)

calibrate_input_ranges(
    model=torch_analog_model,
    calibration_type=InputRangeCalibrationType.CACHE_QUANTILE,
    dataloader=Sampler(training_data),
)
test_evaluation(valid_data, torch_analog_model)

calibrate_input_ranges(
    model=torch_analog_model,
    calibration_type=InputRangeCalibrationType.MOVING_QUANTILE,
    dataloader=Sampler(training_data),
)
test_evaluation(valid_data, torch_analog_model)

calibrate_input_ranges(
    model=torch_analog_model,
    calibration_type=InputRangeCalibrationType.MOVING_STD,
    dataloader=Sampler(training_data),
)
test_evaluation(valid_data, torch_analog_model)

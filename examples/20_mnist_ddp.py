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

"""aihwkit example 20: MNIST training with PyTorch Distributed Data Parallel (DDP).

MNIST training example based on the paper:
https://www.frontiersin.org/articles/10.3389/fnins.2016.00333/full

Uses learning rates of η = 0.01, 0.005, and 0.0025
for epochs 0–10, 11–20, and 21–30, respectively.
"""
# pylint: disable=invalid-name
# pylint: disable=too-many-locals

import os
from time import time

# Imports from PyTorch.
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR

from torchvision import datasets, transforms


# Imports from aihwkit.
from aihwkit.nn import AnalogLinear, AnalogLinearMapped, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import InferenceRPUConfig

# Check device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path where the datasets will be stored.
PATH_DATASET = os.path.join('data', 'DATASET')

# Network definition.
INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10

# Training parameters.
EPOCHS = 30
BATCH_SIZE = 64


def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    print("init process: ", rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29411'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn()


def cleanup():
    """ Destroy distributed processes once they are complete. """
    dist.destroy_process_group()


def load_images():
    """Load images for train from the torchvision datasets."""
    rank = dist.get_rank()
    size = dist.get_world_size()
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the images.
    train_set = datasets.MNIST(PATH_DATASET,
                               download=True, train=True, transform=transform)

    val_set = datasets.MNIST(PATH_DATASET,
                             download=True, train=False, transform=transform)

    train_sampler = torch.utils.data.DistributedSampler(train_set, num_replicas=size, rank=rank,
                                                        shuffle=True, seed=42)

    train_data = torch.utils.data.DataLoader(train_set,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=size,
                                             sampler=train_sampler,
                                             pin_memory=True)

    validation_data = torch.utils.data.DataLoader(val_set,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True,
                                                  num_workers=size,
                                                  pin_memory=True)

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
    model = AnalogSequential(
        AnalogLinear(input_size, hidden_sizes[0], True,
                     rpu_config=InferenceRPUConfig()),
        nn.Sigmoid(),
        AnalogLinear(hidden_sizes[0], hidden_sizes[1], True,
                     rpu_config=InferenceRPUConfig()),
        nn.Sigmoid(),
        AnalogLinearMapped(hidden_sizes[1], output_size, True,
                           rpu_config=InferenceRPUConfig()),
        nn.LogSoftmax(dim=1)
    )

    return model


def create_sgd_optimizer(model):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained.
    Returns:
        nn.Module: optimizer
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
    rank = dist.get_rank()
    size = dist.get_world_size()
    device = torch.device('cuda', rank)

    classifier = nn.NLLLoss()
    optimizer = create_sgd_optimizer(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    time_init = time()
    total_time = [torch.zeros(1, dtype=torch.float).to(device) for _ in range(size)]
    for epoch_number in range(EPOCHS):
        total_loss = torch.zeros(1, dtype=torch.float).to(device)
        total_images = torch.zeros(1, dtype=torch.int).to(device)
        for images, labels in train_set:
            images = images.to(device)
            labels = labels.to(device)
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

            total_images += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_images, op=dist.ReduceOp.SUM)

        if rank == 0:
            train_loss = total_loss.item() / total_images.item()
            print('Epoch {} - Training loss: {:.16f}'.format(epoch_number, train_loss))

        # Decay learning rate if needed.
        scheduler.step()

    dist.all_gather(total_time, torch.tensor(time()-time_init).to(device))

    if rank == 0:
        avg_train_time = torch.mean(torch.cat(total_time, 0))
        print('\nAverage Training Time (s) = {}'.format(avg_train_time))


def test_evaluation(model, val_set):
    """Test trained network

    Args:
        model (nn.Model): Trained model to be evaluated
        val_set (DataLoader): Validation set to perform the evaluation
    """
    rank = dist.get_rank()
    size = dist.get_world_size()
    device = torch.device('cuda', rank)

    # Setup counter of images predicted to 0.
    predicted_ok = 0
    total_images = 0

    # make list to collect test ccuracies for each gpu
    acc_list = [torch.zeros(1, dtype=torch.float).to(device) for _ in range(size)]

    model.eval()

    for images, labels in val_set:
        # Predict image.
        images = images.to(device)
        labels = labels.to(device)

        images = images.view(images.shape[0], -1)
        pred = model(images)

        _, predicted = torch.max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()

    dist.all_gather(acc_list, torch.tensor(predicted_ok/total_images).to(device))

    if rank == 0:
        acc = torch.mean(torch.cat(acc_list, 0))
        print('\nNumber Of Images Tested = {}'.format(total_images))
        print('Model Accuracy = {}'.format(acc))


def main():
    """Train a PyTorch analog model with the MNIST dataset."""
    rank = dist.get_rank()
    device = torch.device('cuda', rank)

    # Load datasets.
    train_dataset, validation_dataset = load_images()

    # Prepare the model.
    model = create_analog_network(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)

    if rank == 0:
        print(model)

    model.prepare_for_ddp()
    model.to(device)

    # enable parallel training
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Train the model.
    train(model, train_dataset)

    # Evaluate the trained model.
    test_evaluation(model, validation_dataset)

    cleanup()


if __name__ == '__main__':
    # Execute only if run as the entry point into the program
    world_size = 2
    print("Device count: ", world_size)
    processes = []
    ctx = mp.get_context("spawn")

    for world_rank in range(world_size):
        print("Process: ", world_rank)
        p = ctx.Process(target=init_process, args=(world_rank, world_size, main))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

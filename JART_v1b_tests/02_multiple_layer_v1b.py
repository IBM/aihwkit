
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

"""aihwkit example 2: network with multiple layers.

Network that consists of multiple analog layers. It aims to learn to sum all
the elements from one array.
"""
# pylint: disable=invalid-name

# Imports from PyTorch.
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.nn import Sequential

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear
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
del config_dictionary["project_name"]
del config_dictionary["CUDA_Enabled"]
del config_dictionary["USE_wandb"]

if "Repeat_Times" in config_dictionary:
    Repeat_Times = config_dictionary["Repeat_Times"]
    del config_dictionary["Repeat_Times"]
else:
    Repeat_Times = 1

for repeat in range(Repeat_Times):
    if USE_wandb:
        import wandb
        if cuda.is_compiled() & CUDA_Enabled:
            wandb.init(project=project_name, group="Multi-Layer Perceptron", job_type="CUDA")
            wandb.config.update(config_dictionary)
        else:
            wandb.init(project=project_name, group="Multi-Layer Perceptron", job_type="CPU")
            wandb.config.update(config_dictionary)

    # Prepare the datasets (input and expected output).
    x_b = Tensor([[0.1, 0.2, 0.0, 0.0], [0.2, 0.4, 0.0, 0.0]])
    y_b = Tensor([[0.3], [0.6]])

    # Define a multiple-layer network.
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
    model = Sequential(
        AnalogLinear(4, 2, rpu_config=rpu_config),
        AnalogLinear(2, 2, rpu_config=rpu_config),
        AnalogLinear(2, 1, rpu_config=rpu_config)
    )

    # Move the model and tensors to cuda if it is available.
    if cuda.is_compiled() & CUDA_Enabled:
        x_b = x_b.cuda()
        y_b = y_b.cuda()
        model.cuda()

    # Define an analog-aware optimizer, preparing it for using the layers.
    opt = AnalogSGD(model.parameters(), lr=config_dictionary["learning_rate"])
    opt.regroup_param_groups(model)

    for epoch in range(config_dictionary["epochs"]):
        # Add the training Tensor to the model (input).
        pred = model(x_b)
        # Add the expected output Tensor.
        loss = mse_loss(pred, y_b)
        if USE_wandb:
            wandb.log({"Loss": loss, "epoch": (epoch+1)})
        else:
            print('Epoch {} - Loss: {:.16f}'.format(
                (epoch+1), loss))
        # Run training (backward propagation).
        loss.backward()

        opt.step()

    wandb.finish()
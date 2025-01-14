
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
import torch
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.nn import Sequential

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
import JART_v1b_tests.yaml_loader as yaml_loader
from aihwkit.simulator.rpu_base import cuda

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="YAML Configuration File")
args = parser.parse_args()

if args.config:
    config_file = args.config
else:
    config_file = "noise_free.yml"

job_type, project_name, CUDA_Enabled, USE_wandb, USE_0_initialization, USE_bias, Repeat_Times, config_dictionary, JART_rpu_config = yaml_loader.from_yaml(config_file)

for repeat in range(Repeat_Times):
    if USE_wandb:
        import wandb
        wandb.init(project=project_name, group="Multi-Layer Perceptron", job_type=job_type)
        wandb.config.update(config_dictionary)

    # Prepare the datasets (input and expected output).
    x_b = Tensor([[0.1, 0.2, 0.0, 0.0], [0.2, 0.4, 0.0, 0.0]])
    y_b = Tensor([[0.3], [0.6]])

    # Define a multiple-layer network.
    rpu_config = SingleRPUConfig(device=JARTv1bDevice(w_max=config_dictionary["w_max"],
                                                      w_min=config_dictionary["w_min"],

                                                      read_voltage=config_dictionary["pulse_related"]["read_voltage"],
                                                      pulse_voltage_SET=config_dictionary["pulse_related"]["pulse_voltage_SET"],
                                                      pulse_voltage_RESET=config_dictionary["pulse_related"]["pulse_voltage_RESET"],
                                                      pulse_length=config_dictionary["pulse_related"]["pulse_length"],
                                                      base_time_step=config_dictionary["pulse_related"]["base_time_step"],

                                                      enable_w_max_w_min_bounds=config_dictionary["noise"]["enable_w_max_w_min_bounds"],
                                                      w_max_dtod=config_dictionary["noise"]["w_max"]["device_to_device"],
                                                      w_max_dtod_upper_bound=config_dictionary["noise"]["w_max"]["dtod_upper_bound"],
                                                      w_max_dtod_lower_bound=config_dictionary["noise"]["w_max"]["dtod_lower_bound"],
                                                      w_min_dtod=config_dictionary["noise"]["w_min"]["device_to_device"],
                                                      w_min_dtod_upper_bound=config_dictionary["noise"]["w_min"]["dtod_upper_bound"],
                                                      w_min_dtod_lower_bound=config_dictionary["noise"]["w_min"]["dtod_lower_bound"],

                                                      Ndiscmax_dtod=config_dictionary["noise"]["Ndiscmax"]["device_to_device"],
                                                      Ndiscmax_dtod_upper_bound=config_dictionary["noise"]["Ndiscmax"]["dtod_upper_bound"],
                                                      Ndiscmax_dtod_lower_bound=config_dictionary["noise"]["Ndiscmax"]["dtod_lower_bound"],
                                                      Ndiscmax_std=config_dictionary["noise"]["Ndiscmax"]["cycle_to_cycle_direct"],
                                                      Ndiscmax_ctoc_upper_bound=config_dictionary["noise"]["Ndiscmax"]["ctoc_upper_bound"],
                                                      Ndiscmax_ctoc_lower_bound=config_dictionary["noise"]["Ndiscmax"]["ctoc_lower_bound"],

                                                      Ndiscmin_dtod=config_dictionary["noise"]["Ndiscmin"]["device_to_device"],
                                                      Ndiscmin_dtod_upper_bound=config_dictionary["noise"]["Ndiscmin"]["dtod_upper_bound"],
                                                      Ndiscmin_dtod_lower_bound=config_dictionary["noise"]["Ndiscmin"]["dtod_lower_bound"],
                                                      Ndiscmin_std=config_dictionary["noise"]["Ndiscmin"]["cycle_to_cycle_direct"],
                                                      Ndiscmin_ctoc_upper_bound=config_dictionary["noise"]["Ndiscmin"]["ctoc_upper_bound"],
                                                      Ndiscmin_ctoc_lower_bound=config_dictionary["noise"]["Ndiscmin"]["ctoc_lower_bound"],

                                                      ldisc_dtod=config_dictionary["noise"]["ldisc"]["device_to_device"],
                                                      ldisc_dtod_upper_bound=config_dictionary["noise"]["ldisc"]["dtod_upper_bound"],
                                                      ldisc_dtod_lower_bound=config_dictionary["noise"]["ldisc"]["dtod_lower_bound"],
                                                      ldisc_std=config_dictionary["noise"]["ldisc"]["cycle_to_cycle_direct"],
                                                      ldisc_std_slope=config_dictionary["noise"]["ldisc"]["cycle_to_cycle_slope"],
                                                      ldisc_ctoc_upper_bound=config_dictionary["noise"]["ldisc"]["ctoc_upper_bound"],
                                                      ldisc_ctoc_lower_bound=config_dictionary["noise"]["ldisc"]["ctoc_lower_bound"],

                                                      rdisc_dtod=config_dictionary["noise"]["rdisc"]["device_to_device"],
                                                      rdisc_dtod_upper_bound=config_dictionary["noise"]["rdisc"]["dtod_upper_bound"],
                                                      rdisc_dtod_lower_bound=config_dictionary["noise"]["rdisc"]["dtod_lower_bound"],
                                                      rdisc_std=config_dictionary["noise"]["rdisc"]["cycle_to_cycle_direct"],
                                                      rdisc_std_slope=config_dictionary["noise"]["rdisc"]["cycle_to_cycle_slope"],
                                                      rdisc_ctoc_upper_bound=config_dictionary["noise"]["rdisc"]["ctoc_upper_bound"],
                                                      rdisc_ctoc_lower_bound=config_dictionary["noise"]["rdisc"]["ctoc_lower_bound"]))
    model = Sequential(
        AnalogLinear(4, 2, rpu_config=rpu_config),
        AnalogLinear(2, 2, rpu_config=rpu_config),
        AnalogLinear(2, 1, rpu_config=rpu_config)
    )
    
    if USE_0_initialization:
        for layer in model:
            if hasattr(layer, 'get_weights'):
                weights, bias = layer.get_weights()
                if USE_bias:
                    layer.set_weights(torch.zeros_like(weights), torch.zeros_like(bias))
                else:
                    layer.set_weights(torch.zeros_like(weights))

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

    if USE_wandb:
        wandb.finish()
# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 28: advanced (time-dependent) IR drop effects

Shows the effects of time-dependent IR drop on an analog layer using
three different modes for applying activations: 1) conventional
PWM/DAC activations, 2) split-mode PWM/DAC activations, and 3) bit-wise
PWM/DAC activations.

1) Conventional Mode: activations are encoded using a PWM/DAC and
quantized according to bit resolution of the PWM/DAC. This mode applies
one set of durations to the analog crossbar array.

2) Split Mode: activations are quantized according to PWM/DAC resolution
and broken into two different segments: a) lower (remainder) bits which are
applied first and outputs stored, b) upper bits which are applied secondly.
Output is shifted (i.e. multiplied) by the appropriate amount and summed
with the lower bit results to obtain the final MVM. This mode applies two
sets of durations to the analog crossbar array.

3) Bit-wise Mode: activations are quantized according to PWM/DAC resolution
and applied bit-by-bit, with the outputs shifted according to the bit
position and all outputs subsequently summed to obtain the final MVM.
This mode applies n-1 sets of activations to the analog crossbar array,
where n is the number of bits in specified by the PWM/DAC resolution.

"""
# pylint: disable=invalid-name

from typing import Type, AnyStr

from math import floor
import time
import matplotlib.pyplot as plt

# Imports from PyTorch.
from torch import cat, Tensor, mean, std, zeros, ones, randn, clamp, linspace, rand, argsort
from torch.nn import Module

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter
from aihwkit.inference import PCMLikeNoiseModel
from aihwkit.simulator.configs import InferenceRPUConfig, TorchInferenceRPUConfigIRDropT
from aihwkit.simulator.parameters.enums import BoundManagementType, AnalogMVType

from aihwkit.simulator.parameters.base import RPUConfigBase

from aihwkit.simulator.rpu_base import cuda

# specs
IN_FEATURES = 512
OUT_FEATURES = 512
BATCH_SIZE = 10
N_ONES = 1
SEGMENTS = 32
USE_CUDA = True


def rpu_config_modifications(rpu_config: Type[RPUConfigBase]) -> Type[RPUConfigBase]:
    """
    Ensures same rpu_config modifications are made to each model
    for fair comparison.

    Args:
        rpu_config: rpu configuration parameters

    Returns: modified rpu_config
    """
    rpu_config.noise_model = PCMLikeNoiseModel(
        prog_noise_scale=0.0,
        read_noise_scale=0.0,
        drift_scale=0.0,
        g_converter=SinglePairConductanceConverter(g_min=0.1, g_max=15.0),
    )
    rpu_config.forward.bound_management = BoundManagementType.NONE
    if isinstance(rpu_config, TorchInferenceRPUConfigIRDropT):
        rpu_config.forward.ir_drop_segments = SEGMENTS
        rpu_config.forward.ir_drop_v_read = 0.4
    rpu_config.forward.ir_drop = 1.0
    rpu_config.forward.nm_thres = 1.0
    rpu_config.forward.inp_res = 2**10
    rpu_config.forward.out_bound = -1  # 10 - quite restrictive
    rpu_config.forward.out_res = -1
    rpu_config.forward.out_noise = 0.0  # prevent bit-wise mode from amplifying noise too much
    return rpu_config


def network_comparison(
    model_baseline: Module, model: Module, x: Tensor, model_name: AnyStr, dataset_name: AnyStr
) -> None:
    """
    Creates comparison plots between default model and advanced
    IR drop models using different operating modes.

    Args:
        model_baseline: default model, which serves as reference
        model: new model to be examined
        x: input activations applied to model
        model_name: string identifier for model
        dataset_name: string identifier for dataset

    Returns: None
    """
    plot_model_names_dict = {
        "conventional_model_dt_irdrop": r"Conventional \ Mode \ Advanced \ IR \ Drop",
        "split_mode_pwm_dt_irdrop": r"Split \ Mode \ Advanced \ IR \ Drop",
        "bit_wise_dt_irdrop": r"Bit \ Wise \ Mode \ Advanced \ IR \ Drop",
    }

    # Move the model and tensors to cuda if it is available.
    device = "cpu"
    if cuda.is_compiled() and USE_CUDA:
        x = x.cuda()
        model = model.cuda()
        model_baseline = model_baseline.cuda()
        device = "cuda"

    model.eval()
    model_baseline.eval()

    # dummy run
    _ = model(x)
    _ = model_baseline(x)

    start_time = time.time()
    out = model(x)
    model_time = time.time() - start_time
    print("\n\t%s model time = %f s (%s)" % (model_name, model_time, device))

    start_time = time.time()
    out_default = model_baseline(x)
    default_irdrop_time = time.time() - start_time
    print("\tdefault IR drop model time = %f s (%s) " % (default_irdrop_time, device))

    print(
        "\t\t%s model is %0.1f times slower (%s)"
        % (model_name, (model_time / default_irdrop_time), device)
    )

    plt.figure()
    plt.plot(out.detach().cpu().numpy().flatten(), ".")
    plt.plot(out_default.detach().cpu().numpy().flatten(), ".")

    # plt.xlabel(r"$Default \ IR \ Drop \ Output \ [1]$")
    plt.xlabel(r"$Flattened \ Output \ Idx$")
    plt.ylabel(r"$%s \ Output \ [1]$" % plot_model_names_dict[model_name])
    plt.title(dataset_name)
    plt.tight_layout()

    plt.figure()
    errors = (out - out_default).detach().cpu().numpy().flatten()
    mu = mean((out - out_default).detach().cpu().flatten()).numpy()
    sigma = std((out - out_default).detach().cpu().flatten()).numpy()
    plt.hist(errors, density=True, bins=100)
    plt.xlabel(r"$IR \ Drop \ \Delta Output \ [1]$")
    plt.ylabel(r"$Density \ [1]$")
    plt.gca().text(
        0.05,
        0.95,
        r"$\mu = %0.2f $" % mu + "\n" + r"$\sigma = %0.2f$" % sigma,
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=1.0),
    )
    plt.tight_layout()


if __name__ == "__main__":
    n_zeros_1 = int((IN_FEATURES - N_ONES) / 2)
    n_zeros_2 = IN_FEATURES - N_ONES - n_zeros_1

    shuffled_weights = linspace(-1.0, 1.0, IN_FEATURES).view(-1, 1) * ones(OUT_FEATURES).view(1, -1)
    for i in range(OUT_FEATURES):
        j = floor(i / OUT_FEATURES * IN_FEATURES)
        if j == 0:
            continue
        idx = argsort(rand(IN_FEATURES))[:j]
        shuffled_weights[idx, i] = shuffled_weights[idx.sort()[0], i]

    data_dict = {
        "out_bound_test": {
            "x": cat(
                (
                    zeros(BATCH_SIZE, n_zeros_1),
                    ones(BATCH_SIZE, N_ONES),
                    zeros(BATCH_SIZE, n_zeros_2),
                ),
                1,
            ),
            "weights": cat(
                (
                    zeros(n_zeros_1, OUT_FEATURES),
                    ones(N_ONES, OUT_FEATURES),
                    zeros(n_zeros_2, OUT_FEATURES),
                ),
                0,
            ),
        },
        "normal_distribution": {
            "x": clamp((1.0 / 3.0) * randn(BATCH_SIZE, IN_FEATURES), min=-1, max=1),
            "weights": clamp((1.0 / 3.0) * randn(IN_FEATURES, OUT_FEATURES), min=-1, max=1),
        },
        "structured_shuffle": {"x": ones(BATCH_SIZE, IN_FEATURES), "weights": shuffled_weights},
    }

    model_dict = {}

    # conventional time-dependent ir drop
    rpu_config_conventional_dt_irdrop = rpu_config_modifications(TorchInferenceRPUConfigIRDropT())
    model_conventional_dt_irdrop = AnalogLinear(
        IN_FEATURES, OUT_FEATURES, bias=False, rpu_config=rpu_config_conventional_dt_irdrop
    )
    model_dict.update({"conventional_model_dt_irdrop": model_conventional_dt_irdrop})

    # split mode pwm time-dependent ir drop
    rpu_config_split_mode_dt_irdrop = rpu_config_modifications(TorchInferenceRPUConfigIRDropT())
    rpu_config_split_mode_dt_irdrop.forward.mv_type = AnalogMVType.SPLIT_MODE
    rpu_config_split_mode_dt_irdrop.forward.ir_drop_bit_shift = 3
    rpu_config_split_mode_dt_irdrop.forward.split_mode_pwm = AnalogMVType.SPLIT_MODE
    model_split_mode_dt_irdrop = AnalogLinear(
        IN_FEATURES, OUT_FEATURES, bias=False, rpu_config=rpu_config_split_mode_dt_irdrop
    )
    model_dict.update({"split_mode_pwm_dt_irdrop": model_split_mode_dt_irdrop})

    # bit wise time-dependent ir drop
    rpu_config_bitwise_dt_irdrop = rpu_config_modifications(TorchInferenceRPUConfigIRDropT())
    rpu_config_bitwise_dt_irdrop.forward.mv_type = AnalogMVType.BIT_WISE
    model_bitwise_dt_irdrop = AnalogLinear(
        IN_FEATURES, OUT_FEATURES, bias=False, rpu_config=rpu_config_bitwise_dt_irdrop
    )
    model_dict.update({"bit_wise_dt_irdrop": model_bitwise_dt_irdrop})

    # default model
    rpu_config_default = rpu_config_modifications(InferenceRPUConfig())
    model_default = AnalogLinear(
        IN_FEATURES, OUT_FEATURES, bias=False, rpu_config=rpu_config_default
    )

    plt.ion()
    for model_identifier, model_test in model_dict.items():
        for dataset_identifier, dataset_dict in data_dict.items():
            weights = dataset_dict["weights"]

            for name, layer in model_default.named_analog_layers():
                layer.set_weights(weights.T, zeros(OUT_FEATURES))

            for name, layer in model_test.named_analog_layers():
                layer.set_weights(weights.T, zeros(OUT_FEATURES))

            network_comparison(
                model_default, model_test, dataset_dict["x"], model_identifier, dataset_identifier
            )

    plt.show()

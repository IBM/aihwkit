# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 32: weight programming options

Showcases four different weight programming options:

    1. SinglePairConductanceConverter - conventional mode uses 1 pair of analog
    memory devices per unit cell.

    2. DualPairConductanceConverter - mode uses 2 pair of analog memory devices
    per unit cell, with significance between the two pair determined by f_lst.
    Only programs in more signifance pair once the range of the lower significant
    pair has been exhausted. This helps to prevent unnecessary amplification of
    programming errors and read noise.

    3. NPairConductanceConverter - allows for N pairs of analog memory devices
    per unit cell, with significance between the pairs determined by f_lst.
    Only programs in more signifance pair once the range of the lower significant
    pair has been exhausted. This helps to prevent unnecessary amplification of
    programming errors and read noise. This is a generalized form of the
    DualPairConductanceConverter.

    4. CustomPairConductanceConverter - enables more complex and custom weight
    programming strategies across N pairs of analog memory devices. Similary,
    f_lst specifies the relative significance between pairs. Enables weight
    programming strategies such as those detailed in

        C. Mackin, et al., "Optimised weight programming for analogue memory-based
            deep neural networks" 2022. https://www.nature.com/articles/s41467-022-31405-1

"""
# pylint: disable=invalid-name
from typing import Type

# Imports for distrubtion generation
from random import uniform

# Imports for plotting
import matplotlib.pyplot as plt

# Imports from PyTorch
from torch import zeros, randn, clamp, linspace, allclose, Tensor

# Imports from aihwkit
from aihwkit.nn import AnalogLinear
from aihwkit.inference.converter.conductance import (
    BaseConductanceConverter,
    SinglePairConductanceConverter,
    DualPairConductanceConverter,
    NPairConductanceConverter,
    CustomPairConductanceConverter
)
from aihwkit.inference import PCMLikeNoiseModel
from aihwkit.simulator.configs import InferenceRPUConfig

# specifications
IN_FEATURES = 512
OUT_FEATURES = 512
BATCH_SIZE = 10
SEGMENTS = 32
USE_CUDA = True

time_dict = {'1 second': 1,
             '1 month' : 1 * 60 * 60 * 24 * 30}

g_min, g_max = 0.1, 15.
single_pair_g_converter = SinglePairConductanceConverter(g_min=g_min, g_max=g_max)
dual_pair_g_converter = DualPairConductanceConverter(f_lst=[1.0, 3.0], g_min=g_min, g_max=g_max)
npair_g_converter = NPairConductanceConverter(f_lst=[1.0, 2.0, 3.0], g_min=g_min, g_max=g_max)

# custom programming model A (curved)
k_lst = [0.5, 0.8]
g_plus_ = [g_min] + [k * (g_max - g_min) + g_min for k in k_lst] + [g_max]
g_minus_ = [g_min] + [(k - (i / (len(k_lst) + 1))) * (g_max - g_min) + g_min
                      for i, k in enumerate(k_lst, 1)] + [g_min]
g_plus = g_minus_[::-1][:-1] + g_plus_
g_minus = g_plus_[::-1][:-1] + g_minus_
prog_model = {'A': [g_plus, g_minus]}

# custom programming model B (random)
n_pts = 5
w_pts = linspace(0, 1, n_pts).detach().cpu().numpy().tolist()
lower_bounds = [(g_max - g_min) * w + g_min for w in w_pts]
g_plus_ = [uniform(lb, g_max) for lb in lower_bounds]
g_minus_ = [g - lb + g_min for g, lb in zip(g_plus_, lower_bounds)]
g_plus = g_minus_[::-1][:-1] + g_plus_
g_minus = g_plus_[::-1][:-1] + g_minus_
prog_model.update({'B': [g_plus, g_minus]})

custom_g_converter = CustomPairConductanceConverter(f_lst=[1.0],
                                                    g_lst=prog_model['A'],
                                                    g_min=g_min,
                                                    g_max=g_max,
                                                    invertibility_test=True)


def plot_weights(g_converter: Type[BaseConductanceConverter],
                 ideal_weights: Tensor,
                 drifted_weights: Tensor,
                 suffix: str = '') -> None:
    """Plots weight programming strategy
    """

    g_lst, params = g_converter.convert_to_conductances(drifted_weights)
    return_weights = g_converter.convert_back_to_weights(g_lst, params)

    assert allclose(drifted_weights, return_weights, atol=0.0001), \
        "conversion error: weights don't match for %s" % str(g_converter)

    g_converter_name = str(g_converter).split('(', maxsplit=1)[0]
    rows = int(len(g_lst) / 2) + 1
    width, height = 7, 4
    plt.subplots(rows, 1, figsize=(width, rows * height))
    plt.subplot(rows, 1, 1)
    if suffix != 'ideal':
        title_str = "%s @ %s" % (g_converter_name, suffix)
    else:
        title_str = "Ideal %s" % g_converter_name
    plt.title(title_str)
    plt.plot(ideal_weights.detach().cpu().numpy().flatten(),
             return_weights.detach().cpu().numpy().flatten(),
             '.',
             ms=1)
    plt.xlabel(r"$Ideal \ Weights \ [1]$")
    plt.ylabel(r"$Actual \ Weights \ [1]$")

    for i, (gp, gm) in enumerate(zip(g_lst[::2], g_lst[1::2])):
        plt.subplot(rows, 1, i + 2)
        plt.plot(ideal_weights.detach().cpu().numpy().flatten(),
                 gp.detach().cpu().numpy().flatten(),
                 '.',
                 ms=1,
                 label=r"$G^+_%d$" % i)
        plt.plot(ideal_weights.detach().cpu().numpy().flatten(),
                 gm.detach().cpu().numpy().flatten(),
                 '.',
                 ms=1,
                 label=r"$G^-_%d$" % i)
        plt.legend()
        plt.xlabel(r"$Ideal \ Weights \ [1]$")
        plt.ylabel(r"$Conductance \ [\mu S]$")

    plt.savefig("%s_%s.png" % (g_converter_name, suffix))
    plt.close()


def main():
    """Compare weight programming strategies (i.e. g_converters)
    """

    # create dataset
    x = clamp((1.0 / 3.0) * randn(BATCH_SIZE, IN_FEATURES), min=-1, max=1)
    ideal_weights = clamp((1.0 / 3.0) * randn(IN_FEATURES, OUT_FEATURES), min=-1, max=1)

    # compare each g_converter
    for g_converter in [single_pair_g_converter,
                        dual_pair_g_converter,
                        npair_g_converter,
                        custom_g_converter]:

        # g_converter applied to ideal weights
        plot_weights(g_converter, ideal_weights, ideal_weights, 'ideal')

        # create simple model using g_converter
        rpu_config = InferenceRPUConfig()
        rpu_config.noise_model = PCMLikeNoiseModel(prog_noise_scale=1.0,
                                                   read_noise_scale=1.0,
                                                   drift_scale=1.0,
                                                   g_converter=g_converter)
        model = AnalogLinear(IN_FEATURES, OUT_FEATURES, bias=False, rpu_config=rpu_config)

        # set weights
        for _, layer in model.named_analog_layers():
            layer.set_weights(ideal_weights.T, zeros(OUT_FEATURES))

        # compare programming strategies at different time steps
        model.eval()
        for t_name, t_inference in time_dict.items():

            model.drift_analog_weights(t_inference)
            _ = model(x)    # dummy inference applies programming errors and read noise
            drifted_weights, _ = model.get_weights()

            # g_converter applied to non-ideal weights
            plot_weights(g_converter, ideal_weights, drifted_weights.T, suffix="%s" % t_name)


if __name__ == "__main__":

    main()

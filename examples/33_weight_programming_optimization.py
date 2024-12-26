# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 33: Example using weight programming optimization to improve BERT transformer
accuracy on SQuAD task.  Weight Programming Optimization implementation which is similar to the
framework reported in the following paper:

    C. Mackin, et al., "Optimised weight programming for analogue memory-based
        deep neural networks" 2022. https://www.nature.com/articles/s41467-022-31405-1.

The example is adapted from code in
    https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb
"""
# pylint: disable=invalid-name, too-many-locals, import-error
# pylint: disable=too-many-branches, too-many-lines, too-many-statements
import os
import pickle
from typing import Type, Dict, List

from argparse import ArgumentParser
from collections import OrderedDict, defaultdict
from copy import deepcopy
from numpy import argsort

import numpy as np
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
)

import torch

from evaluate import load
from datasets import load_dataset

from aihwkit.simulator.configs import (
    InferenceRPUConfig,
    WeightClipType,
    WeightNoiseType,
    BoundManagementType,
    NoiseManagementType,
    WeightClipParameter,
    MappingParameter,
)

from aihwkit.simulator.presets import PresetIOParameters
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.nn.conversion import convert_to_analog

from aihwkit.nn import AnalogLinear
from aihwkit.inference.converter.conductance import (
    BaseConductanceConverter,
    SinglePairConductanceConverter,
    DualPairConductanceConverter,
)
from aihwkit.inference.converter.wpo import (
    WeightProgrammingOptimizer,
    loss_weights,
    downsample_weight_distribution,
)

# Check device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# max length and stride specific to pretrained model
MAX_LENGTH = 320
DOC_STRIDE = 128

# specifications
IN_FEATURES = 512
OUT_FEATURES = 1024
BATCH_SIZE = 512
SUFFIX = 'bert'

rmin, rmax = (0.00, 0.80)   # blue color specs
bmin, bmax = (0.27, 1.00)
gmin, gmax = (0.80, 0.93)

# BERT model from Hugging Face model hub fine-tuned on SQuAD v1
MODEL_NAME = "csarron/bert-base-uncased-squad-v1"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

# Parse some arguments
parser = ArgumentParser("bert example")
parser.add_argument('--result_dir', default='33_results', type=str)
parser.add_argument('--sub_dir', default='m0.1_r0.3_kw1_lm0.2', type=str)
parser.add_argument('--mutation', default=0.1, type=float)
parser.add_argument('--recombination', default=0.3, type=float)
parser.add_argument('--use_max_kw', default=1, type=int)
parser.add_argument('--loss_margin', default=0.2, type=float)
parser.add_argument('--test_out_features', default=20, type=int)
parser.add_argument('--optimize', default=1, type=int)
parser.add_argument('--inference', default=1, type=int)
parser.add_argument('--optimize_f_lst', default=1, type=int)
args = parser.parse_args()


args.sub_dir = 'm' + str(args.mutation) + '_r' + str(args.recombination) \
    + '_kw' + str(args.use_max_kw) + '_lm' + str(args.loss_margin)
args.result_dir = os.path.join(args.result_dir, args.sub_dir)
os.makedirs(args.result_dir, exist_ok=True)


def plot_g_converters(g_converter_baseline: Type[BaseConductanceConverter],
                      g_converter_optimized: Type[BaseConductanceConverter],
                      weights: torch.Tensor, suffix: str = ''):
    """Plots comparison on weight programming strategies (baseline vs optimized)"""
    max_abs_w_unitless = np.amax(np.abs(weights.detach().cpu().numpy()))

    rows, cols = 1, 2
    width, height = 6, 5
    plt.subplots(rows, cols, figsize=(cols * width, rows * height))

    plt.subplot(rows, cols, 1)
    w_unitless = np.linspace(-max_abs_w_unitless, max_abs_w_unitless, 1000)
    w_uS = np.zeros_like(w_unitless)
    f_lst = [1.0] if isinstance(g_converter_baseline,
                                SinglePairConductanceConverter) else g_converter_baseline.f_lst
    for j, f in enumerate(f_lst):
        g_lst, _ = g_converter_baseline.convert_to_conductances(torch.Tensor(w_unitless))
        gp_j, gm_j = g_lst[0::2][j], g_lst[1::2][j]
        plt.plot(w_unitless, gp_j, label=r"$g_{%d}^+(W)$" % j)
        plt.plot(w_unitless, gm_j, label=r"$g_{%d}^-(W)$" % j)
        w_uS += f * (np.asarray(gp_j) - np.asarray(gm_j))
    plt.xlabel(r"$Weight \ [1]$")
    plt.ylabel(r"$Conductance \ [\mu S]$")
    title_str = type(g_converter_baseline).__name__ + \
        ' (' + ', '.join([r"$f_{%d}=%0.1f$" % (i, f) for i, f in enumerate(f_lst)]) + ')'
    plt.title(title_str)
    plt.legend()

    plt.subplot(rows, cols, 2)
    w_unitless = np.linspace(-max_abs_w_unitless,
                             max_abs_w_unitless,
                             len(g_converter_optimized.g_lst[0]))
    w_uS = np.zeros_like(w_unitless)
    f_lst = g_converter_optimized.f_lst
    for j, f in enumerate(f_lst):
        gp_j, gm_j = g_converter_optimized.g_lst[0::2][j], g_converter_optimized.g_lst[1::2][j]
        plt.plot(w_unitless, gp_j, label=r"$g_{%d}^+(W)$" % j)
        plt.plot(w_unitless, gm_j, label=r"$g_{%d}^-(W)$" % j)
        w_uS += f * (np.asarray(gp_j) - np.asarray(gm_j))
    plt.xlabel(r"$Weight \ [1]$")
    plt.ylabel(r"$Conductance \ [\mu S]$")
    title_str = type(g_converter_optimized).__name__ + \
        ' (' + ', '.join([r"$f_{%d}=%0.1f$" % (i, f) for i, f in enumerate(f_lst)]) + ')'
    plt.title(title_str)
    plt.legend()

    plt.savefig(os.path.join(args.result_dir, type(g_converter_baseline).__name__
                             + '_vs_' + type(g_converter_optimized).__name__
                             + '_' + suffix + '.png'))
    plt.close()


def plot_weights(weights, filename=''):
    """Plots histogram of weight distribution"""
    plt.figure()
    plt.hist(weights.flatten(), bins=100, density=True)
    plt.xlabel('Weights [1]')
    plt.ylabel('Density [1]')
    plt.savefig(os.path.join(args.result_dir, filename))
    plt.close()


def plot_contour(ideal_vals: torch.Tensor, time_dict: Dict,
                 effective_vals_lst: List[np.ndarray],
                 xlabel: str = '', ylabel: str = ''):
    """
    Generates color coded contour plots which show maximum weight dispersion as
    a function of time.

    Args:
        ideal_vals: ideal values (weights or activations)
        time_dict: dictionary with numerical time steps at which weight programming
            was evaluated with corresponding time step labels for plotting
        effective_vals_lst: list of effective values at each time step
        xlabel: x axis label for graph
        ylabel: y axis label for graph

    Returns: None
    """

    ideal_vals = ideal_vals.detach().cpu().numpy().flatten()
    effective_vals_lst = [w.flatten()
                          for w in effective_vals_lst]

    for i, (label, effective_vals) in enumerate(zip(time_dict.keys(),
                                                    effective_vals_lst)):

        density, xbins, ybins = np.histogram2d(ideal_vals,
                                               effective_vals,
                                               bins=200,
                                               range=[[np.amin(ideal_vals),
                                                       np.amax(ideal_vals)],
                                                      [np.amin(effective_vals_lst[-1]),
                                                       np.amax(effective_vals_lst[-1])]],
                                               density=True)

        ratio = (float(i) + 1.) / len(effective_vals_lst)
        rbg = (np.clip(rmin + (rmax - rmin) * ratio, rmin, rmax),
               np.clip(bmin + (bmax - bmin) * ratio, bmin, bmax),
               np.clip(gmin + (gmax - gmin) * ratio, gmin, gmax))

        _ = plt.contour(density.transpose(),
                        extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                        linewidths=1,
                        colors=[rbg],
                        levels=[0.0])
        plt.plot(10, 10, color=rbg, label=label)

    ax = plt.gca()
    color_sel = (rmin, bmin, gmin)
    ax.spines['bottom'].set_color(color_sel)
    ax.spines['top'].set_color(color_sel)
    ax.spines['right'].set_color(color_sel)
    ax.spines['left'].set_color(color_sel)
    ax.tick_params(axis='x', colors=color_sel)
    ax.tick_params(axis='y', colors=color_sel)
    ax.xaxis.label.set_color(color_sel)
    ax.yaxis.label.set_color(color_sel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(np.amin(ideal_vals), np.amax(ideal_vals))
    plt.ylim(np.amin(effective_vals_lst[-1]), np.amax(effective_vals_lst[-1]))
    for text in ax.legend(loc='upper left').get_texts():
        color_sel = (rmin, bmin, gmin)
        text.set_color(color_sel)


def plot_errors(ideal_weights: torch.Tensor, time_dict: Dict,
                effective_weights_time_lst: List[np.ndarray],
                xlabel: str = '', ylabel: str = ''):
    """
    Generates color coded contour plots which show maximum weight dispersion as
    a function of time.

    Args:
        ideal_weights: ideal weights we wish to program
        time_dict: dictionary with numerical time steps at which weight programming
            was evaluated with corresponding time step labels for plotting
        effective_weights_time_lst: list of effective weights at each time step
        xlabel: x axis label for graph
        ylabel: y axis label for graph

    Returns: None
    """

    ideal_weights = ideal_weights.detach().cpu().numpy().flatten()
    effective_weights_time_lst = [w.flatten()
                                  for w in effective_weights_time_lst]

    for i, (label, effective_weights) in enumerate(zip(time_dict.keys(),
                                                       effective_weights_time_lst)):

        ratio = (float(i) + 1.) / len(effective_weights_time_lst)
        rbg = (np.clip(rmin + (rmax - rmin) * ratio, rmin, rmax),
               np.clip(bmin + (bmax - bmin) * ratio, bmin, bmax),
               np.clip(gmin + (gmax - gmin) * ratio, gmin, gmax))

        plt.hist(effective_weights - ideal_weights,
                 density=True,
                 bins=100,
                 color=rbg,
                 zorder=len(effective_weights_time_lst) - i)

        plt.plot(10, 10, color=rbg, label=label)

    ax = plt.gca()
    color_sel = (rmin, bmin, gmin)
    ax.spines['bottom'].set_color(color_sel)
    ax.spines['top'].set_color(color_sel)
    ax.spines['right'].set_color(color_sel)
    ax.spines['left'].set_color(color_sel)
    ax.tick_params(axis='x', colors=color_sel)
    ax.tick_params(axis='y', colors=color_sel)
    ax.xaxis.label.set_color(color_sel)
    ax.yaxis.label.set_color(color_sel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(np.amin(ideal_weights), np.amax(ideal_weights))
    for text in ax.legend(loc='upper left').get_texts():
        color_sel = (rmin, bmin, gmin)
        text.set_color(color_sel)


def plot_weight_comparison(d: Dict, suffix: str = ''):
    """Plots weight programming strategy

    Args:
        d: dictionary containing ideal_weights, time_dict, list of effective
        weights, title string, and correspond loss for plotting
        suffix: string with model name for plot naming

    Returns: None
    """
    print("Plotting weight dispersion comparison over time: %s"
          % os.path.join(args.result_dir, d['baseline_title'] + '_comparison.png'))

    xmax = np.amax(np.abs(np.asarray(d['ideal_weights'].detach().cpu().numpy())))
    ymax = np.amax(np.abs(np.asarray(
        d['effective_weights_baseline_time_lst'] + d['effective_weights_optimized_time_lst'])))
    xymax = max([xmax, ymax])

    rows, cols = 1, 2
    width, height = 6, 5
    plt.subplots(rows, cols, figsize=(cols * width, rows * height))
    plt.subplot(rows, cols, 1)
    plot_contour(d['ideal_weights'],
                 d['time_dict'],
                 d['effective_weights_baseline_time_lst'],
                 xlabel=r'$Ideal \ Weights \ [1]$',
                 ylabel=r'$Effective \ Weights \ [1]$')
    plt.xlim(-xmax, xmax)
    plt.ylim(-ymax, ymax)
    plt.plot(np.linspace(-xymax, xymax, 2), np.linspace(-xymax, xymax, 2), '--', c='gray')
    title_str = "%s (Loss = %0.6f)" % (d['baseline_title'],
                                       np.round(d['baseline_loss'],
                                                decimals=6))
    plt.title(title_str, color=(rmin, bmin, gmin))
    plt.subplot(rows, cols, 2)
    plot_contour(d['ideal_weights'],
                 d['time_dict'],
                 d['effective_weights_optimized_time_lst'],
                 xlabel=r'$Ideal \ Weights \ [1]$',
                 ylabel=r'$Effective \ Weights \ [1]$')
    title_str = "%s (Loss = %0.6f)" % (d['optimized_title'],
                                       np.round(d['optimized_loss'],
                                                decimals=6))
    plt.xlim(-xmax, xmax)
    plt.ylim(-ymax, ymax)
    plt.plot(np.linspace(-xymax, xymax, 2), np.linspace(-xymax, xymax, 2), '--', c='gray')
    plt.title(title_str, color=(rmin, bmin, gmin))
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_dir,
                             d['baseline_title'] + '_w_comparison_' + suffix + '.png'))
    plt.close()

    # make corresponding error plots
    plt.subplots(rows, cols, figsize=(cols * width, rows * height))
    plt.subplot(rows, cols, 1)
    plot_errors(d['ideal_weights'],
                d['time_dict'],
                d['effective_weights_baseline_time_lst'],
                xlabel=r'$Weight \ Errors \ [1]$',
                ylabel=r'$Density \ [1]$')
    title_str = "%s (Loss = %0.6f)" % (d['baseline_title'],
                                       np.round(d['baseline_loss'],
                                                decimals=6))
    plt.gca().set_yscale('log')
    plt.title(title_str, color=(rmin, bmin, gmin))
    plt.subplot(rows, cols, 2)
    plot_errors(d['ideal_weights'],
                d['time_dict'],
                d['effective_weights_optimized_time_lst'],
                xlabel=r'$Weight \ Errors \ [1]$',
                ylabel=r'$Density \ [1]$')
    title_str = "%s (Loss = %0.6f)" % (d['optimized_title'],
                                       np.round(d['optimized_loss'],
                                                decimals=6))
    plt.gca().set_yscale('log')
    plt.title(title_str, color=(rmin, bmin, gmin))
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_dir,
                             d['baseline_title'] + '_w_error_comparison_' + suffix + '.png'))
    plt.close()


def plot_z_comparison(d: Dict, suffix: str = ''):
    """Plots weight programming strategy

    Args:
        d: dictionary containing ideal_weights, time_dict, list of effective
        weights, title string, and correspond loss for plotting
        suffix: string with model name to label plots

    Returns: None
    """
    print("Plotting z dispersion comparison over time: %s"
          % os.path.join(args.result_dir, d['baseline_title'] + '_comparison.png'))

    xmax = np.amax(np.abs(np.asarray(d['z_ideal'].detach().cpu().numpy())))
    ymax = np.amax(np.abs(np.asarray(
        d['z_baseline_time_lst'] + d['z_optimized_time_lst'])))
    xymax = max([xmax, ymax])

    rows, cols = 1, 2
    width, height = 6, 5
    plt.subplots(rows, cols, figsize=(cols * width, rows * height))
    plt.subplot(rows, cols, 1)
    plot_contour(d['z_ideal'],
                 d['time_dict'],
                 d['z_baseline_time_lst'],
                 xlabel=r'$Ideal \ Activations \ [1]$',
                 ylabel=r'$Actual \ Activations \ [1]$')
    plt.xlim(-xmax, xmax)
    plt.ylim(-ymax, ymax)
    plt.plot(np.linspace(-xymax, xymax, 2), np.linspace(-xymax, xymax, 2), '--', c='gray')
    title_str = "%s (Loss = %0.6f)" % (d['baseline_title'],
                                       np.round(d['baseline_loss'],
                                                decimals=6))
    plt.title(title_str, color=(rmin, bmin, gmin))
    plt.subplot(rows, cols, 2)
    plot_contour(d['z_ideal'],
                 d['time_dict'],
                 d['z_optimized_time_lst'],
                 xlabel=r'$Ideal \ Activations \ [1]$',
                 ylabel=r'$Actual \ Activations \ [1]$')
    title_str = "%s (Loss = %0.6f)" % (d['optimized_title'],
                                       np.round(d['optimized_loss'],
                                                decimals=6))
    plt.xlim(-xmax, xmax)
    plt.ylim(-ymax, ymax)
    plt.plot(np.linspace(-xymax, xymax, 2), np.linspace(-xymax, xymax, 2), '--', c='gray')
    plt.title(title_str, color=(rmin, bmin, gmin))
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_dir,
                             d['baseline_title'] + '_z_comparison_' + suffix + '.png'))
    plt.close()

    # make corresponding error plots
    plt.subplots(rows, cols, figsize=(cols * width, rows * height))
    plt.subplot(rows, cols, 1)
    plot_errors(d['z_ideal'],
                d['time_dict'],
                d['z_baseline_time_lst'],
                xlabel=r'$Activations \ Errors \ [1]$',
                ylabel=r'$Density \ [1]$')
    title_str = "%s (Loss = %0.6f)" % (d['baseline_title'],
                                       np.round(d['baseline_loss'],
                                                decimals=6))
    plt.gca().set_yscale('log')
    plt.title(title_str, color=(rmin, bmin, gmin))
    plt.subplot(rows, cols, 2)
    plot_errors(d['z_ideal'],
                d['time_dict'],
                d['z_optimized_time_lst'],
                xlabel=r'$Activations \ Errors \ [1]$',
                ylabel=r'$Density \ [1]$')
    title_str = "%s (Loss = %0.6f)" % (d['optimized_title'],
                                       np.round(d['optimized_loss'],
                                                decimals=6))
    plt.gca().set_yscale('log')
    plt.title(title_str, color=(rmin, bmin, gmin))
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_dir,
                             d['baseline_title'] + '_z_error_comparison_' + suffix + '.png'))
    plt.close()


def plot_accuracy_comparison(acc_dict: Dict, suffix: str = ''):
    """Plots accuracies to compare baseline and optimized weight
    programming strategies"""

    print("Plotting accuracy comparison over time: %s"
          % os.path.join(args.result_dir, 'accuracy_comparison.png'))
    time_steps = np.asarray(list(acc_dict['time_dict'].values()))

    rows, cols = 1, 2
    width, height = 6, 5
    plt.subplots(rows, cols, figsize=(cols * width, rows * height))
    plt.subplot(rows, cols, 1)
    plt.plot(time_steps,
             np.asarray(acc_dict['accuracy_baseline']['f1_lst']),
             label=acc_dict['label_baseline'])
    plt.plot(time_steps,
             np.asarray(acc_dict['accuracy_optimized']['f1_lst']),
             label=acc_dict['label_optimized'])
    plt.gca().set_xscale('log')
    plt.gca().set_xticks(list(acc_dict['time_dict'].values()))
    plt.gca().set_xticklabels(list(acc_dict['time_dict'].keys()))
    plt.xlabel('Time [1]')
    plt.ylabel('F1 Score [1]')
    plt.legend()

    plt.subplot(rows, cols, 2)
    plt.plot(time_steps,
             np.asarray(acc_dict['accuracy_baseline']['exact_match_lst']),
             label=acc_dict['label_baseline'])
    plt.plot(time_steps,
             np.asarray(acc_dict['accuracy_optimized']['exact_match_lst']),
             label=acc_dict['label_optimized'])
    plt.gca().set_xscale('log')
    plt.gca().set_xticks(list(acc_dict['time_dict'].values()))
    plt.gca().set_xticklabels(list(acc_dict['time_dict'].keys()))
    plt.xlabel('Time [1]')
    plt.ylabel('Exact Match [1]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_dir, 'accuracy_comparison_' + suffix + '.png'))
    plt.close()


def plot_device_models(rpu_config, filename):
    """Plots device models (programming errors, drift coffecients, read noise) as a reference"""

    g_lin = torch.linspace(rpu_config.noise_model.g_converter.g_min,
                           rpu_config.noise_model.g_converter.g_max,
                           1000)
    gs = g_lin.repeat(10000, 1)
    g_prog = rpu_config.noise_model.apply_programming_noise_to_conductance(gs)
    g_nu = rpu_config.noise_model.generate_drift_coefficients(gs)
    g_read_noise = rpu_config.noise_model.apply_drift_noise_to_conductance(gs,
                                                                           torch.zeros_like(gs),
                                                                           1.0)

    rows, cols = 1, 3
    plt.subplots(rows, cols, figsize=(5 * cols, 4))
    plt.subplot(rows, cols, 1)
    plt.plot(g_lin, g_prog.mean(0) - g_lin)
    plt.fill_between(g_lin, -g_prog.std(0), g_prog.std(0), alpha=0.2)
    plt.xlabel(r"$Conductance \ [\mu S]$")
    plt.ylabel(r"$Programming \ Error \ [\mu S]$")

    plt.subplot(rows, cols, 2)
    plt.plot(g_lin, g_nu.mean(0))
    plt.fill_between(g_lin,
                     g_nu.mean(0) - g_nu.std(0),
                     g_nu.mean(0) + g_nu.std(0), alpha=0.2)
    plt.xlabel(r"$Conductance \ [\mu S]$")
    plt.ylabel(r"$Drift \ Coefficient \ [1]$")

    plt.subplot(rows, cols, 3)
    plt.plot(g_lin, g_read_noise.mean(0) - g_lin)
    plt.fill_between(g_lin, -g_read_noise.std(0), g_read_noise.std(0), alpha=0.2)
    plt.xlabel(r"$Conductance \ [\mu S]$")
    plt.ylabel(r"$Read \ Noise \ [\mu S]$")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def create_rpu_config(tile_size=512, dac_res=256, adc_res=256):
    """Create RPU Config emulated typical PCM Device"""

    rpu_config = InferenceRPUConfig(
        clip=WeightClipParameter(type=WeightClipType.FIXED_VALUE, fixed_value=1.0),
        mapping=MappingParameter(
            digital_bias=True,
            learn_out_scaling=True,
            weight_scaling_omega=1.0,
            out_scaling_columnwise=True,
            weight_scaling_columnwise=True,
            max_input_size=tile_size,
            max_output_size=0,
        ),
        forward=PresetIOParameters(
            w_noise_type=WeightNoiseType.PCM_READ,
            w_noise=0.0175,
            inp_res=dac_res,
            out_res=adc_res,
            out_bound=10.0,
            out_noise=0.04,
            bound_management=BoundManagementType.ITERATIVE,
            noise_management=NoiseManagementType.ABS_MAX,
        ),
        noise_model=PCMLikeNoiseModel(),
        drift_compensation=GlobalDriftCompensation(),
    )
    return rpu_config


def create_model(rpu_config):
    """Return Question Answering model and whether or not it was loaded from a checkpoint"""

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    model = convert_to_analog(model, rpu_config)
    model.remap_analog_weights()

    return model


def preprocess_validation(dataset):
    """Preprocess the validation set"""
    # Some of the questions have lots of whitespace on the left,
    # which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space).
    # So we remove that
    # left whitespace
    dataset["question"] = [q.lstrip() for q in dataset["question"]]

    # Tokenize our dataset with truncation and maybe padding,
    # but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long,
    # each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_dataset = TOKENIZER(
        dataset["question"],
        dataset["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context,
    # we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_dataset.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_dataset["example_id"] = []

    for i in range(len(tokenized_dataset["input_ids"])):
        # Grab the sequence corresponding to that example
        # (to know what is the context and what is the question).
        sequence_ids = tokenized_dataset.sequence_ids(i)
        context_index = 1

        # One example can give several spans,
        # this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_dataset["example_id"].append(dataset["id"][sample_index])

        # Set to None the offset_mapping that are not
        # part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_dataset["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_dataset["offset_mapping"][i])
        ]

    return tokenized_dataset


def postprocess_predictions(
    examples, features, raw_predictions, n_best_size=20, max_answer_length=30
):
    """Postprocess raw predictions"""
    features.set_format(type=features.format["type"], columns=list(features.features.keys()))
    all_start_logits, all_end_logits = raw_predictions

    # Map examples ids to index
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}

    # Create dict of lists, mapping example indices with corresponding feature indices
    features_per_example = defaultdict(list)

    for i, feature in enumerate(features):
        # For each example, take example_id, map to corresponding index
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill
    predictions = OrderedDict()

    print(
        f"Post-processing {len(examples)} example predictions "
        f"split into {len(features)} features."
    )

    # Loop over all examples
    for example_index, example in enumerate(examples):
        # Find the feature indices corresponding to the current example
        feature_indices = features_per_example[example_index]

        # Store valid answers
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            # This is what will allow us to map some the positions in our
            # logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are
                    # out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue

                    # Don't consider answers with a length
                    # that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    # Map the start token to the index of the start of that token in the context
                    # Map the end token to the index of the end of that token in the context
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]

                    # Add the answer
                    # Score is the sum of logits for the start and end position of the answer
                    # Include the text which is taken directly from the context
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char:end_char],
                        }
                    )

        # If we have valid answers, choose the best one
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction,
            # we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Choose the best answer as the prediction for the current example
        predictions[example["id"]] = best_answer["text"]

    return predictions


def create_datasets():
    """Load the SQuAD dataset, the tokenized version, and the validation set"""
    print("Creating dataset.")
    squad = load_dataset("squad", split='validation[:100%]')    # can select fractional amount

    eval_data = squad.map(
        preprocess_validation, batched=True, remove_columns=squad.column_names
    )

    return squad, eval_data


def make_trainer(model, eval_data):
    """Create the Huggingface Trainer"""
    training_args = TrainingArguments(
        output_dir=args.result_dir,
        save_strategy="no",
        per_device_eval_batch_size=4,
        no_cuda=False,
        report_to="none",   # no wandb
    )

    collator = DefaultDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        eval_dataset=eval_data,
        tokenizer=TOKENIZER,
    )

    return trainer


def do_inference(model, trainer, squad, eval_data, time_dict):
    """Perform inference experiment at weight noise level specified at runtime.
    SQuAD exact match and f1 metrics are captured in Tensorboard
    """

    # Helper functions
    def predict():
        # Perform inference + evaluate metric here
        raw_predictions = trainer.predict(eval_data)
        predictions = postprocess_predictions(
            squad, eval_data, raw_predictions.predictions
        )
        # Format to list of dicts instead of a large dict
        formatted_preds = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        out_metric = metric.compute(predictions=formatted_preds, references=ground_truth)
        return out_metric["f1"], out_metric["exact_match"]

    model.eval()
    metric = load("squad", experiment_id=args.sub_dir)
    ground_truth = [{"id": ex["id"], "answers": ex["answers"]} for ex in squad]
    time_steps = list(time_dict.values())

    f1_lst, exact_match_lst = [], []
    for t_inference in time_steps:
        model.program_analog_weights()              # new prog errors + drift coeffs
        model.drift_analog_weights(t_inference)     # drift weights + new read noise + drift alphas
        f1, exact_match = predict()
        print("Inference time = %f: f1 = %f, exact match = %f" % (t_inference, f1, exact_match))
        f1_lst.append(f1)
        exact_match_lst.append(exact_match)

    results_dict = {'f1_lst': f1_lst,
                    'exact_match_lst': exact_match_lst,
                    'time_dict': time_dict,
                    }
    return results_dict


def extract_weights(model):
    """Extracts analog weights from network"""
    params = []
    for name, param in model.named_parameters():
        if 'analog' in name and 'weight' in name:
            params.append(param.flatten())
    weights = torch.cat(params, 0)
    return weights


def plot_weight_distribution(weights, suffix=''):
    """Plots histogram of weight distribution"""
    plt.figure()
    plt.hist(weights.detach().cpu().numpy(), bins=100, density=True)
    plt.xlabel('Weights [1]')
    plt.ylabel('Density [1]')
    plt.title("%0.1f million weights" % (1.e-6 * weights.numel()))
    plt.savefig(os.path.join(args.result_dir, suffix + '_weight_distribution.png'))
    plt.close()


def save_object(obj, filename):
    """Save object"""
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """Loads object"""
    try:
        with open(filename, 'rb') as handle:
            obj = pickle.load(handle)
        return deepcopy(obj)
    except Exception as error:
        raise ValueError("%s does not exist" % filename) from error


def main():
    """Compare weight programming strategies (i.e. g_converters)
    """
    # times to optimize for
    time_dict = OrderedDict([('1 second', 1),
                             ('1 minute', 1 * 60),
                             ('1 hour', 1 * 60 * 60),
                             ('1 day', 1 * 60 * 60 * 24),
                             ('1 month', 1 * 60 * 60 * 24 * 30),
                             ('1 year', 1 * 60 * 60 * 24 * 30 * 12),
                             ])
    time_steps = list(time_dict.values())

    # model name for labeling
    suffix = SUFFIX

    # set up baseline g_converter
    g_min, g_max = 0., 25.
    g_converter_baseline = DualPairConductanceConverter(f_lst=[1.0, 1.0],
                                                        g_min=g_min,
                                                        g_max=g_max)

    # Define RPU configuration and use it to create model and tokenizer
    rpu_config_baseline = create_rpu_config()
    rpu_config_baseline.noise_model.g_converter = g_converter_baseline

    # make baseline model
    model_baseline = create_model(rpu_config_baseline)

    # device models being optimized for
    plot_device_models(rpu_config_baseline,
                       os.path.join(args.result_dir, 'device_models.png'))

    # extract and save model weights
    weights_ideal = extract_weights(model_baseline)
    torch.save(weights_ideal, os.path.join(args.result_dir, 'ideal_weights_' + suffix + '.pt'))

    # plot model weights
    max_abs_weight = torch.max(torch.abs(weights_ideal))
    plot_weight_distribution(weights_ideal, suffix=suffix)

    # weight programming optimization
    if args.optimize:

        # get f_lst from g_converter if exists
        f_lst = g_converter_baseline.f_lst if hasattr(g_converter_baseline, 'f_lst') else [1.0]

        # keep or optimize f_lst params
        f_lst = [1.0] + [None] * (len(f_lst) - 1) if args.optimize_f_lst else f_lst

        wpo = WeightProgrammingOptimizer(weights_ideal,
                                         f_lst,
                                         rpu_config_baseline,
                                         time_steps,
                                         g_converter_baseline,
                                         mutation=args.mutation,
                                         recombination=args.recombination,
                                         use_max_kw=args.use_max_kw,
                                         test_out_features=args.test_out_features,
                                         loss_margin=args.loss_margin,
                                         )
        g_converter_optimized, success = wpo.run_optimizer()
        print("Successful convergence: %s" % str(success))

        # make optimized rpu_config
        rpu_config_optimized = create_rpu_config()
        rpu_config_optimized.noise_model.g_converter = g_converter_optimized

        # save optimization results + baseline
        save_object(rpu_config_baseline,
                    os.path.join(args.result_dir, 'rpu_config_baseline_' + suffix + '.pkl'))
        save_object(rpu_config_optimized,
                    os.path.join(args.result_dir, 'rpu_config_optimized_' + suffix + '.pkl'))
        save_object(wpo,
                    os.path.join(args.result_dir, 'WeightProgrammingOptimizer_' + suffix + '.pkl'))
    else:
        # load optimization results + baseline
        try:
            rpu_config_baseline = load_object(
                os.path.join(args.result_dir, 'rpu_config_baseline_' + suffix + '.pkl'))
            rpu_config_optimized = load_object(
                os.path.join(args.result_dir, 'rpu_config_optimized_' + suffix + '.pkl'))
            wpo = load_object(
                os.path.join(args.result_dir, 'WeightProgrammingOptimizer_' + suffix + '.pkl'))
        except Exception as error:
            raise FileNotFoundError("All requisite data not properly loaded.") from error

    # shorthand
    g_converter_baseline = rpu_config_baseline.noise_model.g_converter
    g_converter_optimized = rpu_config_optimized.noise_model.g_converter

    # plot weight programming strategy
    plot_g_converters(g_converter_baseline, g_converter_optimized, weights_ideal, suffix=suffix)

    # evaluate weight programming optimization
    if args.inference or args.optimize:

        # get effective weights for baseline strategy at different time steps
        fc_baseline = AnalogLinear(IN_FEATURES, OUT_FEATURES, bias=False,
                                   rpu_config=rpu_config_baseline).to(DEVICE).eval()

        fc_optimized = AnalogLinear(IN_FEATURES, OUT_FEATURES, bias=False,
                                    rpu_config=rpu_config_optimized).to(DEVICE).eval()

        weights_test = downsample_weight_distribution(weights_ideal,
                                                      torch.Size([IN_FEATURES,
                                                                  OUT_FEATURES])).to(DEVICE)

        # set weights
        for layer in fc_baseline.analog_layers():
            layer.set_weights(weights_test.T, torch.zeros(OUT_FEATURES))

        for layer in fc_optimized.analog_layers():
            layer.set_weights(weights_test.T, torch.zeros(OUT_FEATURES))

        # dummy activations
        x = 0.5 * torch.randn(BATCH_SIZE, IN_FEATURES).to(DEVICE)
        z_ideal = torch.matmul(x, weights_test)

        # save weight and activation distributions over time
        effective_weights_baseline_time_lst, effective_weights_optimized_time_lst = [], []
        z_baseline_time_lst, z_optimized_time_lst = [], []
        for t_inference in time_dict.values():

            fc_baseline.program_analog_weights()
            fc_baseline.drift_analog_weights(t_inference)
            weights_baseline_tile_lst = []
            for layer in fc_baseline.analog_layers():   # only one layer
                for tile in layer.analog_tiles():
                    weights_baseline_tile_lst.append(
                        tile.alpha * tile.get_weights()[0].T.to(DEVICE))  # includes get_scales()
            effective_weights_baseline_time_lst.append(
                torch.cat(weights_baseline_tile_lst, 1).detach().cpu().numpy())
            z_baseline_time_lst.append(fc_baseline(x).detach().cpu().numpy())

            fc_optimized.program_analog_weights()
            fc_optimized.drift_analog_weights(t_inference)
            weights_optimized_tile_lst = []
            for layer in fc_optimized.analog_layers():  # only one layer
                for tile in layer.analog_tiles():
                    weights_optimized_tile_lst.append(
                        tile.alpha * tile.get_weights()[0].T.to(DEVICE))
            effective_weights_optimized_time_lst.append(
                torch.cat(weights_optimized_tile_lst, 1).detach().cpu().numpy())
            z_optimized_time_lst.append(fc_optimized(x).detach().cpu().numpy())

        # create weight info
        weight_dict = {'ideal_weights': weights_test,
                       'time_dict': time_dict,
                       'effective_weights_baseline_time_lst':
                           effective_weights_baseline_time_lst,
                       'effective_weights_optimized_time_lst':
                           effective_weights_optimized_time_lst,
                       'baseline_title': type(g_converter_baseline).__name__,
                       'optimized_title': type(g_converter_optimized).__name__,
                       'baseline_loss': loss_weights(fc_baseline.to('cpu'),
                                                     time_steps, weights_test.to('cpu'),
                                                     max_abs_weight.to('cpu'), 0, 0,
                                                     get_baseline=True),
                       'optimized_loss': loss_weights(fc_optimized.to('cpu'),
                                                      time_steps, weights_test.to('cpu'),
                                                      max_abs_weight.to('cpu'), 0, 0,
                                                      get_baseline=True),
                       }

        # create activation info
        z_dict = {'z_ideal': z_ideal,
                  'time_dict': time_dict,
                  'z_baseline_time_lst': z_baseline_time_lst,
                  'z_optimized_time_lst': z_optimized_time_lst,
                  'baseline_title': type(rpu_config_baseline.noise_model.g_converter).__name__,
                  'optimized_title': type(rpu_config_optimized.noise_model.g_converter).__name__,
                  'baseline_loss': np.mean(np.stack([(z - z_ideal.detach().cpu().numpy()) ** 2
                                                     for z in z_baseline_time_lst], axis=1)),
                  'optimized_loss': np.mean(np.stack([(z - z_ideal.detach().cpu().numpy()) ** 2
                                                      for z in z_optimized_time_lst], axis=1)),
                  }

        # save results
        save_object(weight_dict, os.path.join(args.result_dir, 'weight_dict_' + suffix + '.pkl'))
        save_object(z_dict, os.path.join(args.result_dir, 'z_dict_' + suffix + '.pkl'))

        # plot weight dispersion over time
        plot_weight_comparison(weight_dict, suffix=suffix)

        # plot accuracy comparison over time
        plot_z_comparison(z_dict, suffix=suffix)

        # create dataset
        squad, eval_data = create_datasets()

        # evaluate accuracy with baseline weight programming strategy
        print("Evaluating model with baseline weight programming.")
        trainer_baseline = make_trainer(model_baseline, eval_data)
        accuracy_baseline = do_inference(model_baseline, trainer_baseline,
                                         squad, eval_data, time_dict)

        # evaluate accuracy with weight programming optimization
        model_optimized = create_model(rpu_config_optimized)
        # model_optimized = deepcopy(model_baseline)
        # for _, tile in model_optimized.named_analog_tiles():
        #     tile.rpu_config.noise_model.g_converter = g_converter_optimized

        print("Evaluating model with optimized weight programming.")
        trainer_optimized = make_trainer(model_optimized, eval_data)
        accuracy_optimized = do_inference(model_optimized, trainer_optimized,
                                          squad, eval_data, time_dict)

        # create and save accuracy info
        accuracy_dict = {'accuracy_baseline': accuracy_baseline,
                         'accuracy_optimized': accuracy_optimized,
                         'label_baseline': type(g_converter_baseline).__name__,
                         'label_optimized': type(g_converter_optimized).__name__,
                         'time_dict': time_dict,
                         }
        save_object(accuracy_dict,
                    os.path.join(args.result_dir, 'accuracy_dict_' + suffix + '.pkl'))

        # compare accuracy results
        plot_accuracy_comparison(accuracy_dict, suffix=suffix)

    else:
        try:
            # load results
            weight_dict = load_object(
                os.path.join(args.result_dir, 'weight_dict_' + suffix + '.pkl'))
            accuracy_dict = load_object(
                os.path.join(args.result_dir, 'accuracy_dict_' + suffix + '.pkl'))
            z_dict = load_object(
                os.path.join(args.result_dir, 'z_dict_' + suffix + '.pkl'))

            # plot weight dispersion over time
            plot_weight_comparison(weight_dict, suffix=suffix)

            # plot test output activations comparison over time
            plot_z_comparison(z_dict, suffix=suffix)

            # plot accuracy comparisin over time
            plot_accuracy_comparison(accuracy_dict, suffix=suffix)

        except Exception as error:
            raise FileNotFoundError("Run optimization and inference first") from error


if __name__ == "__main__":

    main()

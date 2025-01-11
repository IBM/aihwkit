# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 25: Simple correlation detection with analog optimizers.
"""
# pylint: disable=invalid-name, too-many-locals, too-many-statements

from typing import Union, Tuple, Optional, List, Dict

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

# Imports from PyTorch.
from torch import Tensor, randn
from torch import device as torch_device

# Imports from aihwkit.
from aihwkit.simulator.rpu_base import cuda
from aihwkit.simulator.presets import StandardIOParameters
from aihwkit.simulator.tiles import AnalogTile
from aihwkit.simulator.tiles.transfer import TorchTransferTile
from aihwkit.simulator.configs import (
    build_config,
    RPUDataType,
    UnitCellRPUConfig,
    SingleRPUConfig,
    SoftBoundsReferenceDevice,
    MixedPrecisionCompound,
    ChoppedTransferCompound,
    DynamicTransferCompound,
)

# Check device
DEVICE = torch_device("cuda" if cuda.is_compiled() else "cpu")
DATA_TYPE = RPUDataType.FLOAT
USE_TORCH_TRANSFER = False  # whether to use torch transfer implementation


def get_rpu_config(
    algorithm: str = "ttv2", construction_seed: int = 123
) -> Union[UnitCellRPUConfig, SingleRPUConfig]:
    """Returns a rpu_config of a given type.

    Args:
        algorithm: One of 'ttv2', 'ttv3', 'ttv4', 'sgd', 'mp'
        construction_seed: seed of the device construction

    Returns:
        rpu_config: rpu configuration of the requested type

    Raises:
        ConfigError: if ``rpu_type`` is not known
    """

    sb_device = SoftBoundsReferenceDevice(
        dw_min=0.1,
        w_max_dtod=0.3,
        w_min_dtod=0.3,
        w_min=-1.0,
        w_max=1.0,
        up_down_dtod=0.0,
        up_down=0.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        slope_down_dtod=0.0,
        slope_up_dtod=0.0,
        enforce_consistency=True,
        dw_min_dtod_log_normal=True,
        mult_noise=False,
        subtract_symmetry_point=True,
    )

    rpu_config = build_config(
        algorithm, sb_device, StandardIOParameters, construction_seed=construction_seed
    )

    # Some specific settings for mixed precision
    if isinstance(rpu_config.device, MixedPrecisionCompound):
        rpu_config.device.transfer_every = 5
        rpu_config.device.n_rows_per_transfer = 1
        rpu_config.device.granularity = 0.1
        rpu_config.device.n_x_bins = 4
        rpu_config.device.n_d_bins = 4

    # All higher tiki-taka variants inherit from ChoppedTransferCompound.
    if isinstance(rpu_config.device, ChoppedTransferCompound):
        # Common parameters in ttv2 ttv3 ttv4.
        rpu_config.device.transfer_every = 1
        rpu_config.device.auto_granularity = 200
        rpu_config.device.auto_scale = True
        rpu_config.device.in_chop_random = False

    if isinstance(rpu_config.device, DynamicTransferCompound):
        # Common parameters in ttv4.
        rpu_config.device.tail_weightening = 5

    return rpu_config


def create_analog_tile(
    weight: Tensor, rpu_config: UnitCellRPUConfig, device: torch_device = DEVICE
) -> AnalogTile:
    """Creates an analog tile with given weights.

    Args:
        weight: weight tensor_a
        rpu_config: user defined rpu_config
        device: torch device

    Returns:
        AnalogTile: created analog tile
    """
    lr = 0.1
    analog_tile = rpu_config.tile_class(weight.shape[0], weight.shape[1], rpu_config)
    analog_tile.set_weights(weight)

    if device.type == "cuda":
        analog_tile = analog_tile.cuda(device)
    analog_tile.set_learning_rate(lr)

    return analog_tile


def run_updates(analog_tile: AnalogTile, x_data: Tensor, d_data: Tensor) -> Tuple[ArrayLike, Dict]:
    """Runs the update and returns the weight traces.

    Args:
        analog_tile: Tile to use for updates
        x_data: X-data from forward
        d_data: D-data from backward

    Returns:
        weight_trace, hidden_weight_dic
    """

    x_data = x_data.to(DEVICE)
    d_data = d_data.to(DEVICE)

    n_iter = x_data.shape[0]
    w_trace = []
    h_trace = []
    for i in range(n_iter):
        analog_tile.update(x_data[i][None, :], d_data[i][None, :])
        w_trace.append(analog_tile.tile.get_weights())
        h_trace.append(analog_tile.get_hidden_parameters())

    w_trace = np.stack(w_trace, axis=2)
    names = analog_tile.tile.get_hidden_parameter_names()
    h_dic = {}
    for key in names:
        h_dic[key] = np.stack([h[key].cpu().numpy() for h in h_trace], axis=2)

    return w_trace, h_dic


def plot_traces(w_trace: ArrayLike, h_trace_dic: Dict, h_names: Optional[List[str]] = None) -> None:
    """Plots the weight traces.

    Args:
        w_trace: weight traces
        h_trace_dic: hidden weight traces dictionary
        h_names: hidden weight names
    """

    def get_diags(trace):
        diag = np.eye(trace.shape[0]) == 1
        return np.stack([trace[:, :, i][diag] for i in range(trace.shape[2])], axis=1)

    def get_off_diags(trace):
        off_diag = np.eye(trace.shape[0]) != 1
        return np.stack([trace[:, :, i][off_diag] for i in range(trace.shape[2])], axis=1)

    if h_names is None:
        h_names = [k for k in h_trace_dic.keys() if "weight" in k]

    plt.clf()

    w_on = get_diags(w_trace).T
    w_off = get_off_diags(w_trace).T

    plt.subplot(3, 1, 1)
    h1 = plt.plot(w_on, "r")
    h2 = plt.plot(w_off, "b")
    plt.ylim([-1, 1])
    plt.ylabel("Weight")
    plt.legend([h1[0], h2[0]], ["Correlated", "Uncorrelated"])
    for i, idx in enumerate([(0, 0), (1, 0)]):
        plt.subplot(3, 1, i + 2)
        for name in h_names:
            scale = 1.0
            if "momentum" in name:
                scale = 0.1
            plt.plot(h_trace_dic[name][idx[0], idx[1], :] * scale, label=name)
            plt.ylabel("Weight")
            plt.ylim([-1, 1])
    plt.xlabel("# updates")
    plt.legend()
    plt.ion()
    plt.show()


if __name__ == "__main__":
    n = 5

    weight_matrix = 0.001 * randn(n, n)
    t = 5000
    alpha = 0.4  # correlation factor

    x_values = randn(t, n, dtype=DATA_TYPE.as_torch())
    d_values = randn(t, n, dtype=DATA_TYPE.as_torch())
    d_values = alpha * x_values + (1 - alpha) * d_values

    # Algorithm can be one of: 'tiki-taka', 'ttv2', 'c-ttv2', 'agad', 'sgd', 'mp'

    # The better the algorithm, the better the separation of the red
    # correlated traces from the blue uncorrelated (which should stay
    # around zero).

    training_algorithm = "mp"
    # training_algorithm = "ttv2"
    # training_algorithm = "c-ttv2"
    # training_algorithm = "agad"

    my_rpu_config = get_rpu_config(training_algorithm)

    if USE_TORCH_TRANSFER:
        my_rpu_config.tile_class = TorchTransferTile

    my_rpu_config.runtime.data_type = DATA_TYPE
    tile = create_analog_tile(weight_matrix, my_rpu_config)

    dest = {}
    tile.state_dict(dest)
    tile.load_state_dict(dest)

    w_traces, h_traces = run_updates(tile, x_values, d_values)

    plot_traces(w_traces, h_traces)

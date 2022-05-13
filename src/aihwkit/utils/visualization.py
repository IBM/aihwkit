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

"""Visualization utilities.

This module includes plotting and visualization utilities for ``aihwkit``.
Using this module has extra dependencies that can be installed via the
extras mechanism::

    pip install aihwkit[visualization]
"""

# Allow untyped calls for `np.*`, as proper support for numpy typing requires
# 1.20+ (https://numpy.org/devdocs/reference/typing.html).
# mypy: disallow-untyped-calls=False

from copy import deepcopy
from typing import Any, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import ticker
from matplotlib.figure import Figure
from numpy import ndarray
from torch import device as torch_device
from torch import eye, from_numpy, ones

from aihwkit.exceptions import ConfigError
from aihwkit.simulator.configs import SingleRPUConfig, UnitCellRPUConfig
from aihwkit.simulator.configs.devices import PulsedDevice, UnitCell
from aihwkit.simulator.configs.utils import (
    BoundManagementType, IOParameters, NoiseManagementType, PulseType,
    UpdateParameters, WeightNoiseType
)
from aihwkit.simulator.tiles import AnalogTile, BaseTile
from aihwkit.simulator.rpu_base import cuda
from aihwkit.inference.noise.base import BaseNoiseModel
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel


def compute_pulse_response(
        analog_tile: BaseTile,
        direction: ndarray,
        use_forward: bool = False
) -> ndarray:
    """Compute the pulse response of a given device configuration.

    Args:
        analog_tile: Base tile used for computing the weight traces
        direction: numpy vector of directions to sequentially apply (-1 or 1)
        use_forward: Whether to use the (noisy) forward pass to read out the weights
            (otherwise returns exact weight value).

    Returns:
        An numpy array ``w_trace`` of dimensions ``len(direction) x out_size x in_size``
    """
    out_size = analog_tile.out_size
    in_size = analog_tile.in_size

    if analog_tile.is_cuda:
        device = torch_device('cuda')
    else:
        device = torch_device('cpu')

    total_iters = len(direction)
    w_trace = np.zeros((total_iters, out_size, in_size))
    in_vector = -ones(1, in_size, device=device)
    out_vector = ones(1, out_size, device=device)
    in_eye = eye(in_size, device=device)
    dir_tensor = from_numpy(direction).float().to(device)

    for i in range(total_iters):
        # Update the pulses.
        analog_tile.update(in_vector, out_vector * dir_tensor[i])

        if use_forward:
            # Save weights by using the forward pass (to get the short-term read noise).
            w_trace[i, :, :] = analog_tile.forward(in_eye).detach().cpu().numpy().T
        else:
            # Noise free.
            w_trace[i, :, :] = analog_tile.get_weights()[0].detach().cpu().numpy()

    return w_trace


def plot_pulse_response(
        analog_tile: BaseTile,
        direction: ndarray,
        use_forward: bool = False
) -> ndarray:
    """Plot the pulse response of a direction vector for each weight of
    the analog tile.

    Args:
        analog_tile: Base tile used for computing the weight traces
        direction: vector of directions to sequentially apply (-1 or 1)
        use_forward: Whether to use the (noisy) forward pass to read out the weights
            (otherwise returns exact weight value).

    Returns:
        w_trace: from :func:`compute_pulse_response`
    """
    w_trace = compute_pulse_response(analog_tile, direction, use_forward)

    plt.plot(w_trace.reshape(w_trace.shape[0], -1))
    if use_forward:
        plt.title(analog_tile.rpu_config.device.__class__.__name__)
    else:
        plt.title('{} (without cycle/read noise)'
                  .format(analog_tile.rpu_config.device.__class__.__name__))
    plt.ylabel('Weight [conductance]')
    plt.xlabel('Pulse number #')

    return w_trace


def compute_pulse_statistics(
        w_nodes: ndarray,
        w_trace: ndarray,
        direction: ndarray,
        up_direction: bool,
        smoothness: float = 0.5
) -> Tuple[ndarray, ndarray]:
    """Compute the statistics of the step trace from :func:`compute_pulse_response`.

    Args:
        w_nodes: weight range vector to estimate the step histogram
        w_trace: weight trace from :func:`compute_pulse_response`
        direction: direction vector used to generate the weight traces
        up_direction: whether and plot to compute the statistics for up or down direction
        smoothness: value for smoothing the estimation of the
            statistical step response curves

    Returns:
        Tuple of ``(dw_mean, dw_std)``.
    """
    # pylint: disable=too-many-locals

    def calc_mean_and_std(
            node: ndarray,
            w_values: ndarray,
            delta_w: ndarray,
            lam: float
    ) -> Tuple[ndarray, ndarray]:
        """Calculate the mean and std of a w location (node).

        Note:
            In case there are multiple trials then it also includes
            device-to-device variation.
        """
        alpha = np.exp(-0.5*(node - w_values)**2/lam**2)
        beta = alpha.sum(axis=0)
        alpha[:, beta < 0.1] = np.nan
        alpha /= np.expand_dims(beta, axis=0)
        mean = np.sum(alpha*delta_w, axis=0)
        std = np.sqrt(np.sum(alpha*(delta_w - np.expand_dims(mean, axis=0))**2, axis=0))
        return (mean, std)

    # dw statistics.
    delta_w = np.diff(w_trace, axis=0)
    w_trace_s = w_trace[:-1, :, :]
    if up_direction:
        msk = np.logical_and(direction[:-1] > 0, np.diff(direction) == 0)
        w_values = w_trace_s[msk, :, :]
        delta_w_values = delta_w[msk, :, :]
    else:
        msk = np.logical_and(direction[:-1] < 0, np.diff(direction) == 0)
        w_values = w_trace_s[msk, :, :]
        delta_w_values = delta_w[msk, :, :]

    dw_mean = np.zeros((len(w_nodes), w_trace.shape[1], w_trace.shape[2]))
    dw_std = np.zeros((len(w_nodes), w_trace.shape[1], w_trace.shape[2]))

    lam = (w_nodes[1]-w_nodes[0])/2*smoothness
    for i, node in enumerate(w_nodes):
        dw_mean[i, :, :], dw_std[i, :, :] = calc_mean_and_std(node, w_values, delta_w_values, lam)

    return dw_mean, dw_std


def plot_pulse_statistics(
        w_trace: ndarray,
        direction: ndarray,
        up_direction: bool,
        num_nodes: int = 100,
        smoothness: float = 0.5
) -> Tuple[ndarray, ndarray, ndarray]:
    """Plot the dG-G curve from a given weight trace and direction vector.

    Args:
        w_trace: weight trace from :func:`compute_pulse_response`
        direction: direction vector used to generate ``w_trace``
        up_direction: whether and plot to compute the statistics for up or down direction
        num_nodes: number of nodes for estimation of the step histogram
        smoothness: value for smoothing the estimation of the
            statistical step response curves

    Returns:
        A tuple (w_nodes, dw_mean, dw_std) from :func:`compute_pulse_statistics`
    """

    def errorbar_patch(x: ndarray, mean: ndarray, std: ndarray) -> None:
        """Plot a patchy error bar."""
        axis = plt.plot(x, mean)[0]
        plt.fill_between(x, mean - std, mean + std, edgecolor=None,
                         facecolor=axis.get_color(), alpha=0.5)

    # Compute statistics.
    w_nodes = np.linspace(w_trace.min(), w_trace.max(), num_nodes)
    dw_mean, dw_std = compute_pulse_statistics(w_nodes, w_trace, direction, up_direction,
                                               smoothness)

    n_traces = dw_mean.shape[1] * dw_mean.shape[2]

    for i in range(n_traces):
        errorbar_patch(w_nodes,
                       dw_mean.reshape(-1, n_traces)[:, i],
                       dw_std.reshape(-1, n_traces)[:, i])

    plt.xlabel('Weight $w$')
    plt.ylabel('Avg. step $\\Delta w$')
    if up_direction:
        plt.title('up-direction')
    else:
        plt.title('down-direction')

    return w_nodes, dw_mean, dw_std


def get_tile_for_plotting(
        rpu_config: Union[SingleRPUConfig, UnitCellRPUConfig],
        n_traces: int,
        use_cuda: bool = False,
        noise_free: bool = False
) -> BaseTile:
    """Return an analog tile for plotting the response curve.

    Args:
        rpu_config: RPU Configuration to use for plotting
        n_traces: Number of traces to plot
        use_cuda: Whether to use the CUDA implementation (if available)
        noise_free: Whether to turn-off cycle-to-cycle noises (if possible)

    Returns:
        Instantiated tile.
    """

    def set_noise_free(dev: Any) -> Any:
        if hasattr(dev, 'dw_min_std'):
            dev.dw_min_std = 0.0  # Noise free.

        if hasattr(dev, 'refresh_forward'):
            setattr(dev, 'refresh_forward', IOParameters(is_perfect=True))

        if hasattr(dev, 'refresh_update'):
            setattr(dev, 'refresh_update', UpdateParameters(pulse_type=PulseType.NONE))

        if hasattr(dev, 'transfer_forward'):
            setattr(dev, 'refresh_forward', IOParameters(is_perfect=True))

        if hasattr(dev, 'transfer_update'):
            setattr(dev, 'transfer_update', UpdateParameters(pulse_type=PulseType.NONE))

        if (hasattr(dev, 'write_noise_std') and
           getattr(dev, 'write_noise_std') > 0.0):
            # Just make very small to avoid hidden parameter mismatch.
            setattr(dev, 'write_noise_std', 1e-6)

    config = deepcopy(rpu_config)

    # Make sure we use single pulses for the overview.
    config.update.update_bl_management = False
    config.update.update_management = False
    config.update.desired_bl = 1

    if noise_free:
        config.forward.is_perfect = True

        set_noise_free(config.device)
        if hasattr(config.device, 'unit_cell_devices'):
            for dev in getattr(config.device, 'unit_cell_devices'):
                set_noise_free(dev)
        if hasattr(config.device, 'device'):
            set_noise_free(getattr(config.device, 'device'))

    analog_tile = AnalogTile(n_traces, 1, config)  # type: BaseTile
    analog_tile.set_learning_rate(1)
    w_min = getattr(config.device.as_bindings(), 'w_min', -1.0)

    weights = w_min * ones((n_traces, 1))
    analog_tile.set_weights(weights)

    if use_cuda and cuda.is_compiled():
        return analog_tile.cuda()
    return analog_tile


def estimate_n_steps(rpu_config: Union[SingleRPUConfig, UnitCellRPUConfig]) -> int:
    """Estimate the n_steps.

    Note:
        The estimate of the number of update pulses needed to drive
        from smallest to largest conductance. The estimation just
        assumes linear behavior, thus only be a rough estimate for
        non-linear response curves.

    Args:
        rpu_config: RPU Configuration to use for plotting

    Returns:
        Guessed number of steps

    Raises:
        ConfigError: If rpu_config.device does not have the w_min
            attribute (which is only ensured for
            :class:`~aihwkit.simulator.configs.devices.PulseDevice`)
    """
    if not isinstance(rpu_config, SingleRPUConfig):
        return 1000

    device_binding = rpu_config.device.as_bindings()

    if not hasattr(device_binding, 'w_min'):
        raise ConfigError('n_step estimation only for PulsedDevice. ' +
                          'Provide n_step explicitly.')

    weight_granularity = device_binding.calc_weight_granularity()
    w_min = device_binding.w_min
    w_max = device_binding.w_max

    n_steps = int(np.round((w_max - w_min) / weight_granularity))
    return n_steps


def plot_response_overview(
        rpu_config: Union[SingleRPUConfig, UnitCellRPUConfig],
        n_loops: int = 5,
        n_steps: int = None,
        n_traces: int = 5,
        use_cuda: bool = False,
        smoothness: float = 0.1
) -> None:
    """Plot the step response and statistics of a given device configuration.

    Args:
        rpu_config: RPU Configuration to use for plotting
        n_loops: How many hyper-cycles (up/down pulse sequences) to plot
        n_steps: Number of up/down steps per cycle. If not given, will
            be tried to be estimated (only for ``PulsedDevice``
            possible otherwise defaults to 1000 if ``n_steps=None``).
        n_traces: Number of traces to plot
        use_cuda: Whether to use the CUDA implementation (if available)
        smoothness: value for smoothing the estimation of the
            statistical step response curves
    """
    if n_steps is None:
        n_steps = estimate_n_steps(rpu_config)

    total_iters = min(max(n_loops*2*n_steps, 1000), max(50000, 2*n_steps))
    direction = np.sign(np.sin(np.pi*(np.arange(total_iters)+1)/n_steps))

    plt.clf()

    # 1. Noisy tile.
    analog_tile = get_tile_for_plotting(rpu_config, n_traces, use_cuda, noise_free=False)
    plt.subplot(3, 1, 1)
    plot_pulse_response(analog_tile, direction, use_forward=True)

    # 2. Noise-free tile.
    analog_tile_noise_free = get_tile_for_plotting(rpu_config, n_traces, use_cuda, noise_free=True)
    analog_tile_noise_free.set_hidden_parameters(analog_tile.get_hidden_parameters())

    plt.subplot(3, 1, 2)
    w_trace = plot_pulse_response(analog_tile_noise_free, direction, use_forward=False)

    num_nodes = min(n_steps, 100)
    # 3. Plot up statistics.
    plt.subplot(3, 2, 5)
    plot_pulse_statistics(w_trace, direction, True, num_nodes, smoothness)

    # 4. Plot down statistics.
    plt.subplot(3, 2, 6)
    plot_pulse_statistics(w_trace, direction, False, num_nodes, smoothness)

    plt.tight_layout()


def plot_device(device: Union[PulsedDevice, UnitCell], w_noise: float = 0.0, **kwargs: Any) -> None:
    """Plot the step response figure for a given device (preset).

    Note:
        It will use an amount of read weight noise ``w_noise`` for
        reading the weights.

    Args:
        device: PulsedDevice parameters
        w_noise: Weight noise standard deviation during read
        kwargs: for other parameters, see :func:`plot_response_overview`
    """
    plt.figure(figsize=[7, 7])
    # To simulate some weight read noise.
    io_pars = IOParameters(out_noise=0.0,    # no out noise
                           w_noise=w_noise,  # quite low
                           inp_res=-1.,      # turn off DAC
                           out_bound=100.,   # not limiting
                           out_res=-1.,      # turn off ADC
                           bound_management=BoundManagementType.NONE,
                           noise_management=NoiseManagementType.NONE,
                           w_noise_type=WeightNoiseType.ADDITIVE_CONSTANT)

    if isinstance(device, PulsedDevice):
        plot_response_overview(SingleRPUConfig(device=device, forward=io_pars), **kwargs)
    else:
        plot_response_overview(UnitCellRPUConfig(device=device, forward=io_pars), **kwargs)


def plot_device_compact(
        device: Union[PulsedDevice, UnitCell],
        w_noise: float = 0.0,
        n_steps: int = None,
        n_traces: int = 3,
        use_cuda: bool = False,
) -> Figure:
    """Plot a compact step response figure for a given device (preset).

    Note:
        It will use an amount of read weight noise ``w_noise`` for
        reading the weights.

    Args:
        device: ``PulsedDevice`` or ``UnitCell`` parameters
        w_noise: Weight noise standard deviation during read
        n_steps: Number of up/down steps per cycle. If not given, will
            be tried to be estimated (only for ``PulsedDevice``
            possible otherwise defaults to 1000 if ``n_steps=None``).
        n_traces: Number of traces to plot (for device-to-device variation)
        use_cuda: Whether to use CUDA for the computation

    Returns:
        the compact step response figure.
    """
    # pylint: disable=too-many-locals,too-many-statements
    def get_rpu_config(device: Union[PulsedDevice, UnitCell], io_pars: IOParameters) \
            -> Union[SingleRPUConfig, UnitCellRPUConfig]:
        if isinstance(device, PulsedDevice):
            return SingleRPUConfig(device=device, forward=io_pars)
        return UnitCellRPUConfig(device=device, forward=io_pars)

    figure = plt.figure(figsize=[12, 4])

    # To simulate some weight read noise.
    io_pars = IOParameters(out_noise=0.0,    # no out noise
                           w_noise=w_noise,  # quite low
                           inp_res=-1.,      # turn off DAC
                           out_bound=100.,   # not limiting
                           out_res=-1.,      # turn off ADC
                           bound_management=BoundManagementType.NONE,
                           noise_management=NoiseManagementType.NONE,
                           w_noise_type=WeightNoiseType.ADDITIVE_CONSTANT)

    rpu_config = get_rpu_config(device, io_pars)

    if n_steps is None:
        n_steps = estimate_n_steps(rpu_config)

    # Noisy tile response curves.
    n_loops = 2
    total_iters = n_loops*2*n_steps
    direction = np.sign(np.sin(np.pi*(np.arange(total_iters)+1)/n_steps))

    analog_tile = get_tile_for_plotting(rpu_config, n_traces, use_cuda, noise_free=False)
    w_trace = compute_pulse_response(analog_tile, direction, use_forward=True)\
        .reshape(-1, n_traces)
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(w_trace, linewidth=1)
    axis.set_title(analog_tile.rpu_config.device.__class__.__name__)
    axis.set_xlabel('Pulse number #')
    limit = np.abs(w_trace).max()*1.2
    axis.set_ylim(-limit, limit)
    axis.set_xlim(0, total_iters-1)
    axis.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

    # Noise-free tile for statistics.
    n_loops = 1
    total_iters = min(max(n_loops*2*n_steps, 1000), max(50000, 2*n_steps))
    direction = np.sign(np.sin(np.pi*(np.arange(total_iters)+1)/n_steps))

    analog_tile_noise_free = get_tile_for_plotting(rpu_config, n_traces, use_cuda, noise_free=True)
    analog_tile_noise_free.set_hidden_parameters(analog_tile.get_hidden_parameters())

    w_trace = compute_pulse_response(analog_tile_noise_free, direction, False)

    # Compute statistics.
    num_nodes = min(n_steps, 100)
    w_nodes = np.linspace(w_trace.min(), w_trace.max(), num_nodes)

    dw_mean_up = compute_pulse_statistics(w_nodes, w_trace, direction, True)[0]\
        .reshape(-1, n_traces)
    dw_mean_down = compute_pulse_statistics(w_nodes, w_trace, direction, False)[0]\
        .reshape(-1, n_traces)

    # Plot mean up statistics.
    pos = axis.get_position().bounds
    space = 0.1
    gap = 0.01
    axis.set_position([pos[0] + gap + space, pos[1], pos[2] - 2*gap - 2*space, pos[3]])
    axis.set_yticks([])

    axis_left = figure.add_axes([pos[0], pos[1], space, pos[3]])
    dw_mean_up = dw_mean_up.reshape(-1, n_traces)
    for i in range(n_traces):
        axis_left.plot(dw_mean_up[:, i], w_nodes)

    axis_left.set_position([pos[0], pos[1], space, pos[3]])
    axis_left.set_xlabel('Up pulse size')
    axis_left.set_ylabel('Weight \n [conductance]')
    axis_left.set_ylim(-limit, limit)

    # Plot mean down statistics.
    axis_right = figure.add_axes([pos[0] + pos[2] - space, pos[1], space, pos[3]])
    dw_mean_down = dw_mean_down.reshape(-1, n_traces)
    for i in range(n_traces):
        axis_right.plot(np.abs(dw_mean_down[:, i]), w_nodes)

    axis_right.set_yticks([])
    axis_right.set_xlabel('Down pulse size')
    axis_right.set_ylim(-limit, limit)

    # Set xlim's.
    limit = np.maximum(np.nanmax(np.abs(dw_mean_down)),
                       np.nanmax(np.abs(dw_mean_up))) * 1.2
    axis_left.set_xlim(0.0, limit)
    axis_right.set_xlim(0.0, limit)

    return figure


def plot_device_symmetry(
        device: PulsedDevice,
        w_noise: float = 0.0,
        n_pulses: int = 10000,
        n_traces: int = 3,
        use_cuda: bool = False,
        w_init: float = 1.0,
) -> None:
    """Plot the response figure for a given device (preset).

    It will show the response to alternating up down pulses.

    Note:
        It will use an amount of read weight noise ``w_noise`` for
        reading the weights.

    Args:
        device: PulsedDevice parameters
        n_pulses: total number of pulses
        w_noise: Weight noise standard deviation during read
        n_traces: Number of device traces
        use_cuda: Whether to use CUDA,
        w_init: Initial value of the weights
    """
    plt.figure(figsize=[10, 5])

    io_pars = IOParameters(out_noise=0.0,    # no out noise
                           w_noise=w_noise,  # quite low
                           inp_res=-1.,      # turn off DAC
                           out_bound=100.,   # not limiting
                           out_res=-1.,      # turn off ADC
                           bound_management=BoundManagementType.NONE,
                           noise_management=NoiseManagementType.NONE,
                           w_noise_type=WeightNoiseType.ADDITIVE_CONSTANT)

    rpu_config = SingleRPUConfig(device=device, forward=io_pars)

    direction = np.sign(np.cos(np.pi*np.arange(n_pulses)))
    plt.clf()

    analog_tile = get_tile_for_plotting(rpu_config, n_traces, use_cuda, noise_free=False)
    weights = w_init*ones((n_traces, 1))
    analog_tile.set_weights(weights)

    plot_pulse_response(analog_tile, direction, use_forward=False)
    plt.ylim([-1, 1])
    plt.grid(True)


def plot_weight_drift(noise_model: BaseNoiseModel = None,
                      t_inference_list: ndarray = None,
                      w_inits: ndarray = None,
                      n_repeats: int = 25) -> None:
    """Plots the weight drift behavior of a given noise model over time.

    Args:
        noise_model: Noise model of derived from
            :class:`~aihwkit.simulator.noise_models.BaseNoiseModel`
        t_inference_list: Numpy array of times of inference after
            programming at time 0 (in seconds)
        w_inits: Numpy array of target weights to program
        n_repeats: How many repeats to estimate the standard deviation
    """

    plt.figure(figsize=[10, 5])
    if noise_model is None:
        noise_model = PCMLikeNoiseModel()
    if t_inference_list is None:
        t_inference_list = np.logspace(0., 7.0, 15)
    if w_inits is None:
        w_inits = np.linspace(-1., 1., 9)

    weights = w_inits.flatten()
    weights.sort()
    weights = np.tile(weights, [n_repeats, 1])

    m_list = []
    s_list = []
    for t_inference in t_inference_list:
        noisy_weights = noise_model.apply_noise(from_numpy(weights), t_inference).numpy()
        m_list.append(noisy_weights.mean(axis=0))
        s_list.append(noisy_weights.std(axis=0))

    m_array = np.stack(m_list, axis=1)
    s_array = np.stack(s_list, axis=1)

    for i in range(w_inits.size):
        curve = plt.plot(t_inference_list, m_array[i])

        plt.fill_between(t_inference_list,
                         m_array[i] - s_array[i],
                         m_array[i] + s_array[i],
                         edgecolor=None, linewidth=0, alpha=0.3, antialiased=True,
                         facecolor=curve[0].get_color())

    plt.gca().set_xscale('log')
    plt.xlabel('Time after programming [sec]')
    plt.ylabel('Weight value [norm. units]')

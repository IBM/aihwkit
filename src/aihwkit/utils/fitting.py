# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=too-many-locals, too-many-branches

"""Fitting utilities.

This module includes fitting utilities for ``aihwkit``.
Using this module has extra dependencies that can be installed via the
extras mechanism::

    pip install aihwkit[fitting]
"""

from typing import Union, Dict, TypeVar, Tuple, Optional, List, Any
from copy import deepcopy
from dataclasses import fields

from numpy import array, concatenate, newaxis, ndarray
from torch import from_numpy, ones, stack, float32
from lmfit import minimize, Parameters, report_fit

from aihwkit.exceptions import ArgumentError, ConfigError
from aihwkit.simulator.configs.devices import PulsedDevice
from aihwkit.simulator.configs.configs import SingleRPUConfig
from aihwkit.simulator.tiles.analog import AnalogTile

RPUConfigGeneric = TypeVar("RPUConfigGeneric")


def _apply_parameters_to_config(
    device_config: Union[PulsedDevice, RPUConfigGeneric], params: Parameters
) -> None:
    """Apply the fit parameters to the device config.

    Args:
         device_config: device config to be set (in place)
         params: lmfit.Parameters structure

    Raises: ConfigError if parameter was not found"""

    parvals = params.valuesdict()
    if isinstance(device_config, PulsedDevice):
        device = device_config
    else:
        device = getattr(device_config, "device")  # type ignore

    for par, value in parvals.items():
        if not hasattr(device, par):
            raise ConfigError(f" Cannot find parameter '{par}' in device config.")
        setattr(device, par, value)


def fit_measurements(
    parameters: Union[Dict, Parameters],
    pulse_data: Union[Tuple[ndarray], ndarray],
    response_data: Union[Tuple[ndarray], ndarray],
    device_config: Union[PulsedDevice, RPUConfigGeneric],
    suppress_device_noise: bool = True,
    max_pulses: Optional[int] = 1,
    n_traces: int = 1,
    fit_weights: Optional[Union[Tuple[int], int]] = None,
    method: str = "powell",
    verbose: bool = False,
    **fit_kwargs: Any,
) -> Tuple[Any, Union[PulsedDevice, RPUConfigGeneric], List[ndarray]]:
    """Fit pulse response measurement to the given device model using lmfit.

    For example:

    .. code-block:: python

        # responses are conductance data in response to pulses (-1, 1)

        # choose device model and parameter to fit
        device_config = SoftBoundsDevice(w_min=-1.0, w_max=1.0)
        params = {'dw_min': (0.1, 0.001, 5.0),
                  'up_down': (0.0, -0.99, 0.99),
                  'w_max': (1.0, 0.1, 5.0)}

        # fit the response
        fit_res, fit_device_config, model_response = fit_measurements(
            params, pulses, responses,
            device_config=device_config,
            suppress_device_noise=True,
            method='powell',
            fit_weights=fit_weights,
        )
        # fit parameter
        print(fit_res.params.valuesdict())
        # device of best fit
        print(fit_device_config)


    Args:

        parameters: Parameter to vary. Dictionary with parameter names
            (attributes of the device config). Each value is either a
            single value (thus only set, not varied) or a tuple
            ``(x_init, x_min, x_max)``. ``lmfit.Parameters`` class can
            also given directly.

        pulse_data: Pulse data, ie array of number of pulses in up
            (pos) or down (neg) direction. Can be a tuple of multiple
            measurements

        response_data: Corresponfing measured responses to the pulses
            given by ``pulse_data`` as numpy array or list.

            Caution:
                ``axes=1`` can be used for multiple device
                fit. However, then all pulse data needs to have the
                same axis=0 dimension

        device_config: base device configuration

        suppress_device_noise: sets all dtod and std parameters of the device to 0

        n_traces: how many traces to simulate simulaenously

        max_pulses: constrain the number of pulses given.

        fit_weights: the weightening of the individual response traces
            in the loss function

        method: fitting method from ``lmfit`` (default "powell")

        verbose: whether to print fitting results

        fit_kwargs: additional parameter passed to ``lmfit.minimize``

    Returns:

         fit_results: Result of the fit in ``lmfit`` format
         device_config: Device config with found parameter applied
         model_response: Model response of parameter fit

    Raises:
         ArgumentError: in case wrong arguments are given

    """

    if isinstance(pulse_data, tuple) != isinstance(response_data, tuple):
        raise ArgumentError("Either all data inputs need to be tuples or None. ")

    device_config = deepcopy(device_config)
    if isinstance(device_config, PulsedDevice):
        rpu_config = SingleRPUConfig(device=device_config)

        # single pulse mode
        if max_pulses is not None:
            rpu_config.update.desired_bl = max_pulses
            rpu_config.update.update_bl_management = False
            rpu_config.update.update_management = False

    else:
        rpu_config = device_config  # type: ignore

    if suppress_device_noise:
        for field in fields(rpu_config.device):
            if field.name.endswith("dtod") or field.name.endswith("std"):
                setattr(rpu_config.device, field.name, 0.0)

    params = Parameters()
    if isinstance(parameters, Parameters):
        params = parameters
    elif isinstance(parameters, dict):
        for par, values in parameters.items():
            if isinstance(values, tuple):
                x_init, x_min, x_max = values
                params.add(par, value=x_init, min=x_min, max=x_max, vary=True)
            else:
                params.add(par, value=values, vary=False)
    else:
        raise ArgumentError("Expect dict or Parameters for parmeters.")

    # fit parameters
    args = (pulse_data, response_data, rpu_config, n_traces, fit_weights, verbose)
    result = minimize(model_response, params, args=args, method=method, **fit_kwargs)
    if verbose:
        report_fit(result)

    best_model_res = model_response(result.params, *args, only_response=True)

    _apply_parameters_to_config(device_config, result.params)
    return result, device_config, best_model_res  # type: ignore


def model_response(
    params: Parameters,
    pulse_data: Union[Tuple[ndarray], ndarray],
    response_data: Union[Tuple[ndarray], ndarray],
    rpu_config: RPUConfigGeneric,
    n_traces: int = 1,
    fit_weights: Optional[Union[Tuple[int], int]] = None,
    verbose: bool = True,
    only_response: bool = False,
) -> Union[ndarray, List[ndarray]]:
    """Compute the model respunses given the pulses.

    Args:
        params: ``lmfit.Parameters`` of the current parameter setting

        pulse_data: Pulse data, ie array of number of pulses in up
            (pos) or down (neg) direction. Can be a tuple of multiple
            measurements

        response_data: Corresponfing measured responses to the pulses
            given by ``pulse_data`` as numpy array or list.

            Caution:
                ``axes=1`` can be used for multiple device
                fit. However, then all pulse data needs to have the
                same axis=0 dimension

        rpu_config: base device configuration (will be modified)

        fit_weights: the weightening of the individual response traces
            in the loss function

        n_traces: how many traces to simulate simulaenously

        verbose: whether to print std of deviation

        only_response: whether to returns a list of model response
           instead of the deviation

    Returns:
        deviation vector or list of model responses (weight traces)

    Note:
        overwrites the given rpu_config

    """

    _apply_parameters_to_config(rpu_config, params)

    # likley somewhat inefficient since we need to always create a new
    # tile, repeats are quick though
    no_list = False
    if not isinstance(pulse_data, tuple):
        pulse_data = (pulse_data,)
        response_data = (response_data,)  # type: ignore
        if fit_weights is not None:
            fit_weights = (fit_weights,)  # type: ignore
        no_list = True

    numpy_pulses = array(pulse_data[0])
    n_devices = 1
    if numpy_pulses.ndim > 1:
        n_devices = numpy_pulses.shape[1]

    analog_tile = AnalogTile(n_traces, n_devices, rpu_config)  # type: ignore
    analog_tile.set_learning_rate(1)

    deviation = array([], "float")
    model_responses = []
    for idx, (numpy_pulses, response) in enumerate(zip(pulse_data, response_data)):
        if numpy_pulses.ndim == 1:
            numpy_pulses = numpy_pulses.reshape(-1, 1)
        if response.ndim == 1:
            response = response.reshape(-1, 1)

        w_init = response[0, :]
        weights = from_numpy(array(w_init).flatten()[newaxis, :]).to(dtype=float32) * ones(
            (n_traces, n_devices), dtype=float32
        )
        analog_tile.set_weights(weights)
        pulses = from_numpy(numpy_pulses).to(dtype=float32)
        w_trace = [weights]
        for pulse in pulses[:-1]:
            analog_tile.update(
                pulse * ones(n_devices, dtype=float32), -ones((n_traces), dtype=float32)
            )
            w_trace.append(analog_tile.tile.get_weights())

        stacked_w_trace = stack(w_trace).cpu().numpy()
        # compute square error
        num_samples = response.shape[0]
        avg_w_trace = stacked_w_trace.mean(axis=1)[:num_samples, :]
        model_responses.append(avg_w_trace)
        dev = avg_w_trace - response
        if fit_weights is not None:
            dev = dev * array(fit_weights[idx])[: dev.shape[1]]  # type: ignore
        deviation = concatenate([deviation, dev.flatten()])

    if only_response:
        if no_list:
            return model_responses[0]
        return model_responses
    if verbose:
        print(deviation.std())
    return deviation

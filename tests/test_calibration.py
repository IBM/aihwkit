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

"""Tests for input range calibration."""

import random
from typing import Union
from copy import deepcopy
from pytest import mark
from torch import allclose, randn, manual_seed
from torch.nn import LogSoftmax, Flatten, MaxPool2d
import numpy as np

from aihwkit.nn import AnalogLinear, AnalogSequential, AnalogConv2d
from aihwkit.simulator.configs import (
    TorchInferenceRPUConfig,
    InferenceRPUConfig,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.inference.calibration import calibrate_input_ranges, InputRangeCalibrationType

STRATEGIES = [
    InputRangeCalibrationType.CACHE_QUANTILE,
    InputRangeCalibrationType.MOVING_QUANTILE,
    InputRangeCalibrationType.MOVING_STD,
]


def create_analog_network(rpu_config):
    """Create test network."""
    channel = [16, 32, 512, 128]
    model = AnalogSequential(
        AnalogConv2d(
            in_channels=1, out_channels=channel[0], kernel_size=5, stride=1, rpu_config=rpu_config
        ),
        MaxPool2d(kernel_size=2),
        AnalogConv2d(
            in_channels=channel[0],
            out_channels=channel[1],
            kernel_size=5,
            stride=1,
            rpu_config=rpu_config,
        ),
        MaxPool2d(kernel_size=2),
        Flatten(),
        AnalogLinear(in_features=channel[2], out_features=channel[3], rpu_config=rpu_config),
        AnalogLinear(in_features=channel[3], out_features=10, rpu_config=rpu_config),
        LogSoftmax(dim=1),
    )
    return model


def get_rpu(rpu: Union[TorchInferenceRPUConfig, InferenceRPUConfig]):
    """Create test rpu config."""
    rpu.forward.out_noise = 0.01
    rpu.forward.noise_management = NoiseManagementType.NONE
    rpu.forward.bound_management = BoundManagementType.NONE
    rpu.pre_post.input_range.enable = True
    return rpu


class Sampler:
    """Example of a sampler used for calibration."""

    def __init__(self, inp_shape):
        self.inp_shape = inp_shape
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        x = randn(self.inp_shape)
        self.idx += 1
        if self.idx > 10:
            raise StopIteration
        return (x,), {}


def fix_random(seed):
    """Fix random seed."""
    random.seed(seed)
    manual_seed(seed)
    np.random.seed(seed)


@mark.parametrize("rpu_cls", [TorchInferenceRPUConfig, InferenceRPUConfig])
@mark.parametrize("strategy", STRATEGIES)
def test_determinism(rpu_cls, strategy):
    """Test whether the calibration is deterministic."""
    dataloader = Sampler(inp_shape=(10, 1, 28, 28))
    model = create_analog_network(get_rpu(rpu_cls()))
    fix_random(0)
    calibrate_input_ranges(model, strategy, dataloader)
    ir_dict = {}
    for tile_name, tile in model.named_analog_tiles():
        if tile.input_range is not None:
            ir_dict[tile_name] = tile.input_range.data
    dataloader = Sampler(inp_shape=(10, 1, 28, 28))
    fix_random(0)
    calibrate_input_ranges(model, strategy, dataloader)
    for tile_name, tile in model.named_analog_tiles():
        if tile.input_range is not None:
            assert allclose(ir_dict[tile_name], tile.input_range.data, atol=1e-5)


@mark.parametrize("rpu_cls", [TorchInferenceRPUConfig, InferenceRPUConfig])
@mark.parametrize("strategy", STRATEGIES)
def test_state_before_and_after(rpu_cls, strategy):
    """Test the correct state of the tile before and after calibration."""
    dataloader = Sampler(inp_shape=(10, 1, 28, 28))
    rpu_config = get_rpu(rpu_cls())
    model = create_analog_network(deepcopy(rpu_config))
    calibrate_input_ranges(model, strategy, dataloader)
    assert model.training
    assert (
        next(model.analog_tiles()).rpu_config.forward.noise_management == NoiseManagementType.NONE
    )
    assert next(model.analog_tiles()).rpu_config.forward.is_perfect == rpu_config.forward.is_perfect

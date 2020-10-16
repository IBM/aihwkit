# -*- coding: utf-8 -*-

# (C) Copyright 2020 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Configurations for resistive processing units."""

from dataclasses import dataclass, field
from typing import ClassVar, Type

from aihwkit.simulator.configs.devices import (
    FloatingPointDevice, ConstantStepDevice, PulsedDevice,
    UnitCellDevice, IdealDevice
)

from aihwkit.simulator.configs.utils import (
    BackwardIOParameters, IOParameters, UpdateParameters, PulseType,
    tile_parameters_to_bindings
)
from aihwkit.simulator.rpu_base import devices
from aihwkit.simulator.noise_models import (
    BaseNoiseModel, PCMLikeNoiseModel,
    BaseDriftCompensation, GlobalDriftCompensation
)


@dataclass
class FloatingPointRPUConfig:
    """Configuration for a floating point resistive processing unit."""

    device: FloatingPointDevice = field(default_factory=FloatingPointDevice)
    """Parameters that modify the behavior of the pulsed device."""


@dataclass
class SingleRPUConfig:
    """Configuration for an analog (pulsed device) resistive processing unit."""

    bindings_class: ClassVar[Type] = devices.AnalogTileParameter

    device: PulsedDevice = field(default_factory=ConstantStepDevice)
    """Parameters that modify the behavior of the pulsed device."""

    forward: IOParameters = field(default_factory=IOParameters)
    """Input-output parameter setting for the forward direction."""

    backward: BackwardIOParameters = field(
        default_factory=BackwardIOParameters)
    """Input-output parameter setting for the backward direction."""

    update: UpdateParameters = field(default_factory=UpdateParameters)
    """Parameter for the update behavior."""

    def as_bindings(self) -> devices.AnalogTileParameter:
        """Return a representation of this instance as a simulator bindings object."""
        return tile_parameters_to_bindings(self)


@dataclass
class UnitCellRPUConfig:
    """Configuration for an analog (unit cell) resistive processing unit."""

    bindings_class: ClassVar[Type] = devices.AnalogTileParameter

    device: UnitCellDevice = field(default_factory=UnitCellDevice)
    """Parameters that modify the behavior of the pulsed device."""

    forward: IOParameters = field(default_factory=IOParameters)
    """Input-output parameter setting for the forward direction."""

    backward: BackwardIOParameters = field(
        default_factory=BackwardIOParameters)
    """Input-output parameter setting for the backward direction."""

    update: UpdateParameters = field(default_factory=UpdateParameters)
    """Parameter for the update behavior."""

    def as_bindings(self) -> devices.AnalogTileParameter:
        """Return a representation of this instance as a simulator bindings object."""
        return tile_parameters_to_bindings(self)


@dataclass
class InferenceRPUConfig:
    """Configuration for an analog tile that is used only for inference.

    Training is done in *hardware-aware* manner, thus using only the
    non-idealities of the forward-pass, but backward and update passes
    are ideal.

    During inference, statistical models of programming, drift
    and read noise can be used.

    """

    bindings_class: ClassVar[Type] = devices.AnalogTileParameter

    forward: IOParameters = field(default_factory=IOParameters)
    """Input-output parameter setting for the forward direction."""

    noise_model: BaseNoiseModel = field(default_factory=PCMLikeNoiseModel)
    """Statistical noise model to be used during (realistic) inference."""

    drift_compensation: BaseDriftCompensation = field(default_factory=GlobalDriftCompensation)
    """For compensating the drift during inference only."""

    device: IdealDevice = field(default_factory=IdealDevice)
    """Ideal device."""

    def as_bindings(self) -> devices.AnalogTileParameter:
        """Return a representation of this instance as a simulator bindings object."""

        # backward/update/device technically read-only properties, so we set
        # them here instead
        params_dic = {'forward': self.forward,
                      'backward': BackwardIOParameters(is_perfect=True),
                      'update': UpdateParameters(pulse_type=PulseType.NONE),
                      'bindings_class': self.bindings_class
                      }
        return tile_parameters_to_bindings(params_dic)

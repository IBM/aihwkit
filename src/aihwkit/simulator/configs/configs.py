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
    ConstantStepDevice, FloatingPointDevice, IdealDevice, PulsedDevice,
    UnitCell
)
from aihwkit.simulator.configs.helpers import (
    _PrintableMixin, tile_parameters_to_bindings
)
from aihwkit.simulator.configs.utils import (
    IOParameters, PulseType, UpdateParameters, WeightClipParameter,
    WeightModifierParameter
)
from aihwkit.simulator.noise_models import (
    BaseDriftCompensation, BaseNoiseModel, GlobalDriftCompensation,
    PCMLikeNoiseModel
)
from aihwkit.simulator.rpu_base import devices


@dataclass
class FloatingPointRPUConfig(_PrintableMixin):
    """Configuration for a floating point resistive processing unit."""

    device: FloatingPointDevice = field(default_factory=FloatingPointDevice)
    """Parameters that modify the behavior of the pulsed device."""


@dataclass
class SingleRPUConfig(_PrintableMixin):
    """Configuration for an analog (pulsed device) resistive processing unit."""

    bindings_class: ClassVar[Type] = devices.AnalogTileParameter

    device: PulsedDevice = field(default_factory=ConstantStepDevice)
    """Parameters that modify the behavior of the pulsed device."""

    forward: IOParameters = field(default_factory=IOParameters)
    """Input-output parameter setting for the forward direction."""

    backward: IOParameters = field(default_factory=IOParameters)
    """Input-output parameter setting for the backward direction."""

    update: UpdateParameters = field(default_factory=UpdateParameters)
    """Parameter for the update behavior."""

    def as_bindings(self) -> devices.AnalogTileParameter:
        """Return a representation of this instance as a simulator bindings object."""
        return tile_parameters_to_bindings(self)


@dataclass
class UnitCellRPUConfig(_PrintableMixin):
    """Configuration for an analog (unit cell) resistive processing unit."""

    bindings_class: ClassVar[Type] = devices.AnalogTileParameter

    device: UnitCell = field(default_factory=UnitCell)
    """Parameters that modify the behavior of the pulsed device."""

    forward: IOParameters = field(default_factory=IOParameters)
    """Input-output parameter setting for the forward direction."""

    backward: IOParameters = field(default_factory=IOParameters)
    """Input-output parameter setting for the backward direction."""

    update: UpdateParameters = field(default_factory=UpdateParameters)
    """Parameter for the update behavior."""

    def as_bindings(self) -> devices.AnalogTileParameter:
        """Return a representation of this instance as a simulator bindings object."""
        return tile_parameters_to_bindings(self)


@dataclass
class InferenceRPUConfig(_PrintableMixin):
    """Configuration for an analog tile that is used only for inference.

    Training is done in *hardware-aware* manner, thus using only the
    non-idealities of the forward-pass, but backward and update passes
    are ideal.

    During inference, statistical models of programming, drift
    and read noise can be used.
    """
    # pylint: disable=too-many-instance-attributes

    bindings_class: ClassVar[Type] = devices.AnalogTileParameter

    forward: IOParameters = field(default_factory=IOParameters)
    """Input-output parameter setting for the forward direction."""

    noise_model: BaseNoiseModel = field(default_factory=PCMLikeNoiseModel)
    """Statistical noise model to be used during (realistic) inference."""

    drift_compensation: BaseDriftCompensation = field(default_factory=GlobalDriftCompensation)
    """For compensating the drift during inference only."""

    clip: WeightClipParameter = field(default_factory=WeightClipParameter)
    """Parameters for weight clip."""

    modifier: WeightModifierParameter = field(default_factory=WeightModifierParameter)
    """Parameters for weight modifier."""

    # The following fields are not included in `__init__`, and should be
    # treated as read-only.

    device: IdealDevice = field(default_factory=IdealDevice,
                                init=False)
    """Parameters that modify the behavior of the pulsed device: ideal device."""

    backward: IOParameters = field(
        default_factory=lambda: IOParameters(is_perfect=True),
        init=False
    )
    """Input-output parameter setting for the backward direction: perfect."""

    update: UpdateParameters = field(
        default_factory=lambda: UpdateParameters(pulse_type=PulseType.NONE),
        init=False
    )
    """Parameter for the update behavior: ``NONE`` pulse type."""

    def as_bindings(self) -> devices.AnalogTileParameter:
        """Return a representation of this instance as a simulator bindings object."""
        return tile_parameters_to_bindings(self)

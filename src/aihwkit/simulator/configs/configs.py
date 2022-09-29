# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
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
from typing import ClassVar, Type, Optional, Union, TYPE_CHECKING

from aihwkit.simulator.configs.devices import (
    ConstantStepDevice, FloatingPointDevice, IdealDevice, PulsedDevice,
    UnitCell, DigitalRankUpdateCell
)
from aihwkit.simulator.configs.helpers import (
    _PrintableMixin, tile_parameters_to_bindings
)
from aihwkit.simulator.configs.utils import (
    IOParameters, PulseType, UpdateParameters, WeightClipParameter,
    WeightModifierParameter, WeightRemapParameter, MappingParameter
)
from aihwkit.inference import (
    BaseDriftCompensation, BaseNoiseModel, GlobalDriftCompensation,
    PCMLikeNoiseModel
)
from aihwkit.simulator.rpu_base import devices
from aihwkit.simulator.tiles import AnalogTile, FloatingPointTile, InferenceTile

if TYPE_CHECKING:
    from aihwkit.nn.modules.linear import AnalogLinear
    from aihwkit.nn.modules.linear_mapped import AnalogLinearMapped


@dataclass
class MapableRPU(_PrintableMixin):
    """Defines the mapping parameters and utility factories"""

    mapping: MappingParameter = field(default_factory=MappingParameter)
    """Parameter related to mapping weights to tiles for supporting modules."""

    def get_linear(self) -> Union[Type['AnalogLinear'], Type['AnalogLinearMapped']]:
        """Returns a AnalogLinear module as specified """
        # pylint: disable=import-outside-toplevel
        # need to import here to avoid circular imports
        from aihwkit.nn.modules.linear import AnalogLinear
        from aihwkit.nn.modules.linear_mapped import AnalogLinearMapped

        if self.mapping.max_input_size > 0 or self.mapping.max_output_size > 0:
            return AnalogLinearMapped
        return AnalogLinear


@dataclass
class FloatingPointRPUConfig(MapableRPU, _PrintableMixin):
    """Configuration for a floating point resistive processing unit."""

    tile_class: ClassVar[Type] = FloatingPointTile
    """Tile class that correspond to this RPUConfig."""

    device: FloatingPointDevice = field(default_factory=FloatingPointDevice)
    """Parameter that modify the behavior of the pulsed device."""


@dataclass
class SingleRPUConfig(MapableRPU, _PrintableMixin):
    """Configuration for an analog (pulsed device) resistive processing unit."""

    tile_class: ClassVar[Type] = AnalogTile
    """Tile class that correspond to this RPUConfig."""

    bindings_class: ClassVar[Type] = devices.AnalogTileParameter

    device: PulsedDevice = field(default_factory=ConstantStepDevice)
    """Parameter that modify the behavior of the pulsed device."""

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
class UnitCellRPUConfig(MapableRPU, _PrintableMixin):
    """Configuration for an analog (unit cell) resistive processing unit."""

    tile_class: ClassVar[Type] = AnalogTile
    """Tile class that correspond to this RPUConfig."""

    bindings_class: ClassVar[Type] = devices.AnalogTileParameter

    device: UnitCell = field(default_factory=UnitCell)
    """Parameter that modify the behavior of the pulsed device."""

    forward: IOParameters = field(default_factory=IOParameters)
    """Input-output parameter setting for the forward direction."""

    backward: IOParameters = field(default_factory=IOParameters)
    """Input-output parameter setting for the backward direction."""

    update: UpdateParameters = field(default_factory=UpdateParameters)
    """Parameter for the parallel analog update behavior."""

    def as_bindings(self) -> devices.AnalogTileParameter:
        """Return a representation of this instance as a simulator bindings object."""
        return tile_parameters_to_bindings(self)


@dataclass
class InferenceRPUConfig(MapableRPU, _PrintableMixin):
    """Configuration for an analog tile that is used only for inference.

    Training is done in *hardware-aware* manner, thus using only the
    non-idealities of the forward-pass, but backward and update passes
    are ideal.

    During inference, statistical models of programming, drift
    and read noise can be used.
    """
    # pylint: disable=too-many-instance-attributes

    tile_class: ClassVar[Type] = InferenceTile
    """Tile class that correspond to this RPUConfig."""

    bindings_class: ClassVar[Type] = devices.AnalogTileParameter

    forward: IOParameters = field(default_factory=IOParameters)
    """Input-output parameter setting for the forward direction.

    This parameters govern the hardware definitions specifying analog
    MVM non-idealities.

    Note:

        This forward pass is applied equally in training and
        inference. In addition, materials effects such as drift and
        programming noise can be enabled during inference by
        specifying the ``noise_model``

    """

    noise_model: BaseNoiseModel = field(default_factory=PCMLikeNoiseModel)
    """Statistical noise model to be used during (realistic) inference.

    This noise models establishes a phenomenological model of the
    material which is applied to the weights during inference only, when
    ``program_analog_weights`` or ``drift_analog_weights`` is called.

    """

    drift_compensation: Optional[BaseDriftCompensation] = field(
        default_factory=GlobalDriftCompensation)
    """For compensating the drift during inference only."""

    clip: WeightClipParameter = field(default_factory=WeightClipParameter)
    """Parameter for weight clip.

    If a clipping type is set, the weights are clipped according to
    the type specified.

    Caution:

        The clipping type is set to ``None`` by default, setting
        parameters of the clipping will not be taken into account, if
        the clipping type is not specified.
    """

    remap: WeightRemapParameter = field(default_factory=WeightRemapParameter)
    """Parameter for remapping.

    Remapping can be enabled by specifying a remap ``type``. If
    enabled, it ensures that the weights are mapped maximally into the
    conductance units during training. It will be called after each mini-batch.
    """

    modifier: WeightModifierParameter = field(default_factory=WeightModifierParameter)

    """Parameter for weight modifier.

    If a modifier type is set, it is called once per mini-match in the
    ``post_update_step`` and modifies the weight in forward and
    backward direction for the next mini-batch during training, but
    updates hidden reference weights. In eval mode, the reference
    weights are used instead for forward.

    The modifier is used to do hardware-aware training, so that the
    model becomes more noise robust during inference (e.g. when the
    ``noise_model`` is employed).
    """

    # The following fields are not included in `__init__`, and should be
    # treated as read-only.

    device: IdealDevice = field(default_factory=IdealDevice,
                                init=False)
    """Parameter that modify the behavior of the pulsed device: ideal device."""

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


@dataclass
class DigitalRankUpdateRPUConfig(MapableRPU, _PrintableMixin):
    """Configuration for an analog (unit cell) resistive processing unit
    where the rank update is done in digital.

    Note that for forward and backward, an analog crossbar is still
    used, and during update the digitally computed rank update is
    transferred to the analog crossbar using pulses.
    """

    tile_class: ClassVar[Type] = AnalogTile
    """Tile class that correspond to this RPUConfig."""

    bindings_class: ClassVar[Type] = devices.AnalogTileParameter

    device: DigitalRankUpdateCell = field(default_factory=DigitalRankUpdateCell)
    """Parameter that modify the behavior of the pulsed device."""

    forward: IOParameters = field(default_factory=IOParameters)
    """Input-output parameter setting for the forward direction."""

    backward: IOParameters = field(default_factory=IOParameters)
    """Input-output parameter setting for the backward direction."""

    update: UpdateParameters = field(default_factory=UpdateParameters)
    """Parameter for the analog part of the update, that is the transfer
    from the digital buffer to the devices."""

    def as_bindings(self) -> devices.AnalogTileParameter:
        """Return a representation of this instance as a simulator bindings object."""
        return tile_parameters_to_bindings(self)

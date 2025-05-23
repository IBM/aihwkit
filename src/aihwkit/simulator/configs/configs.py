# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Configurations for resistive processing units."""

# pylint: disable=too-few-public-methods

from dataclasses import dataclass, field
from typing import ClassVar, Type, Optional, Union, Any

from aihwkit.simulator.parameters.pre_post import (
    PrePostProcessingRPU,
    PrePostProcessingParameterQuant,
)
from aihwkit.simulator.parameters.mapping import MappableRPU
from aihwkit.simulator.parameters.helpers import tile_parameters_to_bindings
from aihwkit.simulator.parameters.quantization import ActivationQuantConfig
from aihwkit.simulator.parameters.runtime import RuntimeParameter

from aihwkit.simulator.configs.devices import (
    ConstantStepDevice,
    FloatingPointDevice,
    IdealDevice,
    PulsedDevice,
)
from aihwkit.simulator.configs.compounds import DigitalRankUpdateCell, UnitCell, TransferCompound
from aihwkit.simulator.parameters import (
    IOParameters,
    IOParametersIRDropT,
    PulseType,
    UpdateParameters,
    WeightClipParameter,
    WeightModifierParameter,
    WeightRemapParameter,
)
from aihwkit.inference import (
    BaseDriftCompensation,
    BaseNoiseModel,
    GlobalDriftCompensation,
    PCMLikeNoiseModel,
)

from aihwkit.simulator.tiles import (
    AnalogTile,
    FloatingPointTile,
    InferenceTile,
    TorchInferenceTile,
    QuantizedTorchInferenceTile,
)

from aihwkit.simulator.tiles.torch_tile import TorchSimulatorTile
from aihwkit.simulator.tiles.torch_tile_irdrop_t import TorchSimulatorTileIRDropT
from aihwkit.simulator.tiles.array import TileModuleArray, QuantizedTileModuleArray


@dataclass
class FloatingPointRPUConfig(MappableRPU, PrePostProcessingRPU):
    """Configuration for a floating point resistive processing unit."""

    tile_class: Type = FloatingPointTile
    """Tile class that corresponds to this RPUConfig."""

    tile_array_class: Type = TileModuleArray
    """Tile class used for mapped logical tile arrays."""

    device: FloatingPointDevice = field(default_factory=FloatingPointDevice)
    """Parameter that modify the behavior of the pulsed device."""


@dataclass
class IOManagedRPUConfig(MappableRPU, PrePostProcessingRPU):
    """Configuration for an analog (pulsed device) resistive processing unit."""

    bindings_class: ClassVar[Optional[Union[Type, str]]] = "AnalogTileParameter"
    bindings_module: ClassVar[Optional[str]] = "devices"

    forward: IOParameters = field(
        default_factory=IOParameters, metadata=dict(bindings_include=True)
    )
    """Input-output parameter setting for the forward direction."""

    backward: IOParameters = field(
        default_factory=IOParameters, metadata=dict(bindings_include=True)
    )
    """Input-output parameter setting for the backward direction."""

    update: UpdateParameters = field(
        default_factory=UpdateParameters, metadata=dict(bindings_include=True)
    )
    """Parameter for the update behavior."""

    def as_bindings(self) -> Any:
        """Return a representation of this instance as a simulator bindings object."""
        if not hasattr(self, "runtime"):
            # legacy
            self.runtime = RuntimeParameter()
        return tile_parameters_to_bindings(self, self.runtime.data_type)


@dataclass
class SingleRPUConfig(IOManagedRPUConfig):
    """Configuration for an analog (pulsed device) resistive processing unit."""

    tile_class: Type = AnalogTile
    """Tile class that corresponds to this RPUConfig."""

    tile_array_class: Type = TileModuleArray
    """Tile class used for mapped logical tile arrays."""

    device: PulsedDevice = field(default_factory=ConstantStepDevice)
    """Parameter that modify the behavior of the pulsed device."""


@dataclass
class UnitCellRPUConfig(IOManagedRPUConfig):
    """Configuration for an analog (unit cell) resistive processing unit."""

    tile_class: Type = AnalogTile
    """Tile class that corresponds to this RPUConfig."""

    tile_array_class: Type = TileModuleArray
    """Tile class used for mapped logical tile arrays."""

    device: Union[UnitCell, TransferCompound] = field(default_factory=UnitCell)
    """Parameter that modify the behavior of the pulsed device."""


@dataclass
class DigitalRankUpdateRPUConfig(IOManagedRPUConfig):
    """Configuration for an analog (unit cell) resistive processing unit
    where the rank update is done in digital.

    Note that for forward and backward, an analog crossbar is still
    used, and during update the digitally computed rank update is
    transferred to the analog crossbar using pulses.
    """

    tile_class: Type = AnalogTile
    """Tile class that corresponds to this RPUConfig."""

    tile_array_class: Type = TileModuleArray
    """Tile class used for mapped logical tile arrays."""

    device: DigitalRankUpdateCell = field(default_factory=DigitalRankUpdateCell)
    """Parameter that modify the behavior of the pulsed device."""


@dataclass
class InferenceRPUConfig(IOManagedRPUConfig):
    """Configuration for an analog tile that is used only for inference.

    Training is done in *hardware-aware* manner, thus using only the
    non-idealities of the forward-pass, but backward and update passes
    are ideal.

    During inference, statistical models of programming, drift
    and read noise can be used.
    """

    # pylint: disable=too-many-instance-attributes

    tile_class: Type = InferenceTile
    """Tile class that corresponds to this RPUConfig."""

    tile_array_class: Type = TileModuleArray
    """Tile class used for mapped logical tile arrays."""

    forward: IOParameters = field(
        default_factory=IOParameters, metadata=dict(bindings_include=True)
    )
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
        default_factory=GlobalDriftCompensation
    )
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

    device: IdealDevice = field(default_factory=IdealDevice, init=False)
    """Parameter that modify the behavior of the pulsed device: ideal device."""

    backward: IOParameters = field(
        default_factory=lambda: IOParameters(is_perfect=True),
        init=False,
        metadata=dict(bindings_include=True),
    )
    """Input-output parameter setting for the backward direction: perfect."""

    update: UpdateParameters = field(
        default_factory=lambda: UpdateParameters(pulse_type=PulseType.NONE),
        init=False,
        metadata=dict(bindings_include=True),
    )
    """Parameter for the update behavior: ``NONE`` pulse type."""

    def compatible_with(self, tile_class_name: str) -> bool:
        if tile_class_name in ["TorchInferenceTile"]:
            return True
        return tile_class_name == self.tile_class.__name__


@dataclass
class TorchInferenceRPUConfig(InferenceRPUConfig):
    """TorchInference configuration.

    This configuration defaults to a tile module implementation that
    supported a subset of functions of the ``InferenceRPUConfig`` but
    uses native torch instead of the RPUCuda library for simulating
    the analog MVM.

    The advantage is that autograd is more fully supported and
    hardware aware training is more flexible to be modified. However,
    some nonidealities are not supported.

    Note:

        For features that are not supported a ``NotImplementedError`` or a
        ``TorchTileConfigError`` is raised.
    """

    simulator_tile_class: Type = TorchSimulatorTile

    tile_class: Type = TorchInferenceTile
    """Tile class that corresponds to this RPUConfig."""

    tile_array_class: Type = TileModuleArray
    """Tile class used for mapped logical tile arrays."""


@dataclass
class QuantizedTorchInferenceRPUConfig(TorchInferenceRPUConfig):
    """Extends the TorchInference configuration with quantized functionality
    in the periphery of the tile and the final output"""

    tile_class: Type = QuantizedTorchInferenceTile
    tile_array_class: Type = QuantizedTileModuleArray
    """Tile and tile array classes used to simulate quantization
    on the output and the periphery of the tiles"""

    act_quant_config: Optional[ActivationQuantConfig] = None
    """Activation quantization configuration for the output of the tiles,
    including the output of the array, if multiple tiles are used"""

    pre_post: PrePostProcessingParameterQuant = field(
        default_factory=PrePostProcessingParameterQuant
    )
    """PrePostProcessing Parameter containing the periphery quantization
    options, that are only supported in QuantizedTorchInferenceTile
    and QuantizedTileModuleArray"""


@dataclass
class TorchInferenceRPUConfigIRDropT(TorchInferenceRPUConfig):
    """Inference configuration using time-dependent IR drop.

    This configuration defaults to a tile module implementation that
    supported a subset of functions of the ``InferenceRPUConfig`` but
    uses native torch instead of the RPUCuda library for simulating
    the analog MVM.

    The advantage is that autograd is more fully supported and
    hardware aware training is more flexible to be modified. However,
    some nonidealities are not supported.

    Note:

        For features that are not supported a ``NotImplementedError`` or a
        ``TorchTileConfigError`` is raised.
    """

    simulator_tile_class: Type = TorchSimulatorTileIRDropT

    forward: IOParametersIRDropT = field(default_factory=IOParametersIRDropT)
    """Input-output parameter setting for the forward direction.

    This parameters govern the hardware definitions specifying analog
    MVM non-idealities.

    Note:

        This forward pass is applied equally in training and
        inference. In addition, materials effects such as drift and
        programming noise can be enabled during inference by
        specifying the ``noise_model``
    """

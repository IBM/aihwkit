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

# pylint: disable=too-many-lines

"""RPU configurations presets for resistive processing units."""

from dataclasses import dataclass, field

from aihwkit.simulator.configs.configs import (
    SingleRPUConfig,
    UnitCellRPUConfig,
    DigitalRankUpdateRPUConfig,
)
from aihwkit.simulator.configs.devices import PulsedDevice
from aihwkit.simulator.configs.compounds import (
    TransferCompound,
    UnitCell,
    VectorUnitCell,
    DynamicTransferCompound,
    DigitalRankUpdateCell,
    MixedPrecisionCompound,
    ChoppedTransferCompound,
)
from aihwkit.simulator.parameters.enums import (
    VectorUnitCellUpdatePolicy,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.simulator.parameters.training import UpdateParameters
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.presets.devices import (
    CapacitorPresetDevice,
    EcRamPresetDevice,
    EcRamMOPresetDevice,
    IdealizedPresetDevice,
    ReRamESPresetDevice,
    ReRamSBPresetDevice,
    GokmenVlasovPresetDevice,
)
from aihwkit.simulator.presets.compounds import PCMPresetUnitCell
from aihwkit.simulator.presets.utils import PresetIOParameters, PresetUpdateParameters


# Single device configs.


@dataclass
class ReRamESPreset(SingleRPUConfig):
    """Preset configuration using a single ReRam device (based on ExpStep
    model, see :class:`~aihwkit.simulator.presets.devices.ReRamESPresetDevice`).

    This preset uses standard SGD with fully parallel update on analog
    with stochastic pulses.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: PulsedDevice = field(default_factory=ReRamESPresetDevice)
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class ReRamSBPreset(SingleRPUConfig):
    """Preset configuration using a single ReRam device (based on
    SoftBounds model, see
    :class:`~aihwkit.simulator.presets.devices.ReRamSBPresetDevice`).

    This preset uses standard SGD with fully parallel update on analog
    with stochastic pulses.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: PulsedDevice = field(default_factory=ReRamSBPresetDevice)
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class CapacitorPreset(SingleRPUConfig):
    """Preset configuration using a single capacitor device, see
    :class:`~aihwkit.simulator.presets.devices.CapacitorPresetDevice`.

    This preset uses standard SGD with fully parallel update on analog
    with stochastic pulses.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: PulsedDevice = field(default_factory=CapacitorPresetDevice)
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class EcRamPreset(SingleRPUConfig):
    """Preset configuration using a single Lithium-based EcRAM device, see
    :class:`~aihwkit.simulator.presets.devices.EcRamPresetDevice`.

    This preset uses standard SGD with fully parallel update on analog
    with stochastic pulses.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: PulsedDevice = field(default_factory=EcRamPresetDevice)
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class EcRamMOPreset(SingleRPUConfig):
    """Preset configuration using a single metal-oxide EcRAM device, see
    :class:`~aihwkit.simulator.presets.devices.EcRamMOPresetDevice`.

    This preset uses standard SGD with fully parallel update on analog
    with stochastic pulses.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: PulsedDevice = field(default_factory=EcRamMOPresetDevice)
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class IdealizedPreset(SingleRPUConfig):
    """Preset configuration using a single idealized device, see
    :class:`~aihwkit.simulator.presets.devices.IdealizedPresetDevice`.

    This preset uses standard SGD with fully parallel update on analog
    with stochastic pulses.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: PulsedDevice = field(default_factory=IdealizedPresetDevice)
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class GokmenVlasovPreset(SingleRPUConfig):
    """Preset configuration using a single device with constant update
    step size, see
    :class:`~aihwkit.simulator.presets.devices.GokmenVlasovPresetDevice`.

    This preset uses standard SGD with fully parallel update on analog
    with stochastic pulses.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: PulsedDevice = field(default_factory=GokmenVlasovPresetDevice)
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class PCMPreset(UnitCellRPUConfig):
    """Preset configuration using a single pair of PCM devicec with refresh, see
    :class:`~aihwkit.simulator.presets.devices.PCMPresetUnitCell`.

    This preset uses standard SGD with fully parallel update on analog
    with stochastic pulses.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(default_factory=PCMPresetUnitCell)
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


# 2-device configs.


@dataclass
class ReRamES2Preset(UnitCellRPUConfig):
    """Preset configuration using two ReRam devices per cross-point
    (:class:`~aihwkit.simulator.presets.devices.ReRamESPresetDevice`),
    where both are updated with random selection policy for update.

    See :class:`~aihwkit.simulator.configs.devices.VectorUnitCell` for
    more details on multiple devices per cross-points.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: VectorUnitCell(
            unit_cell_devices=[ReRamESPresetDevice(), ReRamESPresetDevice()],
            update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class ReRamSB2Preset(UnitCellRPUConfig):
    """Preset configuration using two ReRam devices per cross-point
    (:class:`~aihwkit.simulator.presets.devices.ReRamSBPresetDevice`),
    where both are updated with random selection policy for update.

    See :class:`~aihwkit.simulator.configs.devices.VectorUnitCell` for
    more details on multiple devices per cross-points.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: VectorUnitCell(
            unit_cell_devices=[ReRamSBPresetDevice(), ReRamSBPresetDevice()],
            update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class Capacitor2Preset(UnitCellRPUConfig):
    """Preset configuration using two Capacitor devices per cross-point
    (:class:`~aihwkit.simulator.presets.devices.CapacitorPresetDevice`),
    where both are updated with random selection policy for update.

    See :class:`~aihwkit.simulator.configs.devices.VectorUnitCell` for
    more details on multiple devices per cross-points.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: VectorUnitCell(
            unit_cell_devices=[CapacitorPresetDevice(), CapacitorPresetDevice()],
            update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class EcRam2Preset(UnitCellRPUConfig):
    """Preset configuration using two Lithium-based EcRam devices per cross-point
    (:class:`~aihwkit.simulator.presets.devices.EcRamPresetDevice`),
    where both are updated with random selection policy for update.

    See :class:`~aihwkit.simulator.configs.devices.VectorUnitCell` for
    more details on multiple devices per cross-points.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: VectorUnitCell(
            unit_cell_devices=[EcRamPresetDevice(), EcRamPresetDevice()],
            update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class EcRamMO2Preset(UnitCellRPUConfig):
    """Preset configuration using two metal-oxide EcRam devices per cross-point
    (:class:`~aihwkit.simulator.presets.devices.EcRamMOPresetDevice`),
    where both are updated with random selection policy for update.

    See :class:`~aihwkit.simulator.configs.devices.VectorUnitCell` for
    more details on multiple devices per cross-points.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: VectorUnitCell(
            unit_cell_devices=[EcRamMOPresetDevice(), EcRamMOPresetDevice()],
            update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class Idealized2Preset(UnitCellRPUConfig):
    """Preset configuration using two Idealized devices per cross-point
    (:class:`~aihwkit.simulator.presets.devices.IdealizedPresetDevice`),
    where both are updated with random selection policy for update.

    See :class:`~aihwkit.simulator.configs.devices.VectorUnitCell` for
    more details on multiple devices per cross-points.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: VectorUnitCell(
            unit_cell_devices=[IdealizedPresetDevice(), IdealizedPresetDevice()],
            update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


# 4-device configs.


@dataclass
class ReRamES4Preset(UnitCellRPUConfig):
    """Preset configuration using four ReRam devices per cross-point
    (:class:`~aihwkit.simulator.presets.devices.ReRamESPresetDevice`),
    where both are updated with random selection policy for update.

    See :class:`~aihwkit.simulator.configs.devices.VectorUnitCell` for
    more details on multiple devices per cross-points.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: VectorUnitCell(
            unit_cell_devices=[
                ReRamESPresetDevice(),
                ReRamESPresetDevice(),
                ReRamESPresetDevice(),
                ReRamESPresetDevice(),
            ],
            update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class ReRamSB4Preset(UnitCellRPUConfig):
    """Preset configuration using four ReRam devices per cross-point
    (:class:`~aihwkit.simulator.presets.devices.ReRamSBPresetDevice`),
    where both are updated with random selection policy for update.

    See :class:`~aihwkit.simulator.configs.devices.VectorUnitCell` for
    more details on multiple devices per cross-points.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: VectorUnitCell(
            unit_cell_devices=[
                ReRamSBPresetDevice(),
                ReRamSBPresetDevice(),
                ReRamSBPresetDevice(),
                ReRamSBPresetDevice(),
            ],
            update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class Capacitor4Preset(UnitCellRPUConfig):
    """Preset configuration using four Capacitor devices per cross-point
    (:class:`~aihwkit.simulator.presets.devices.CapacitorPresetDevice`),
    where both are updated with random selection policy for update.

    See :class:`~aihwkit.simulator.configs.devices.VectorUnitCell` for
    more details on multiple devices per cross-points.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: VectorUnitCell(
            unit_cell_devices=[
                CapacitorPresetDevice(),
                CapacitorPresetDevice(),
                CapacitorPresetDevice(),
                CapacitorPresetDevice(),
            ],
            update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class EcRam4Preset(UnitCellRPUConfig):
    """Preset configuration using four Lithium-based EcRam devices per cross-point
    (:class:`~aihwkit.simulator.presets.devices.EcRamPresetDevice`),
    where both are updated with random selection policy for update.

    See :class:`~aihwkit.simulator.configs.devices.VectorUnitCell` for
    more details on multiple devices per cross-points.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: VectorUnitCell(
            unit_cell_devices=[
                EcRamPresetDevice(),
                EcRamPresetDevice(),
                EcRamPresetDevice(),
                EcRamPresetDevice(),
            ],
            update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class EcRamMO4Preset(UnitCellRPUConfig):
    """Preset configuration using four metal-oxide EcRam devices per cross-point
    (:class:`~aihwkit.simulator.presets.devices.EcRamMOPresetDevice`),
    where both are updated with random selection policy for update.

    See :class:`~aihwkit.simulator.configs.devices.VectorUnitCell` for
    more details on multiple devices per cross-points.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: VectorUnitCell(
            unit_cell_devices=[
                EcRamMOPresetDevice(),
                EcRamMOPresetDevice(),
                EcRamMOPresetDevice(),
                EcRamMOPresetDevice(),
            ],
            update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class Idealized4Preset(UnitCellRPUConfig):
    """Preset configuration using four Idealized devices per cross-point
    (:class:`~aihwkit.simulator.presets.devices.IdealizedPresetDevice`),
    where both are updated with random selection policy for update.

    See :class:`~aihwkit.simulator.configs.devices.VectorUnitCell` for
    more details on multiple devices per cross-points.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: VectorUnitCell(
            unit_cell_devices=[
                IdealizedPresetDevice(),
                IdealizedPresetDevice(),
                IdealizedPresetDevice(),
                IdealizedPresetDevice(),
            ],
            update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


# Tiki-taka configs.


@dataclass
class TikiTakaReRamESPreset(UnitCellRPUConfig):
    """Configuration using Tiki-taka with
    :class:`~aihwkit.simulator.presets.devices.ReRamESPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.TransferCompound`
    for details on Tiki-taka-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: TransferCompound(
            unit_cell_devices=[ReRamESPresetDevice(), ReRamESPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(),
            units_in_mbatch=True,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class TikiTakaReRamSBPreset(UnitCellRPUConfig):
    """Configuration using Tiki-taka with
    :class:`~aihwkit.simulator.presets.devices.ReRamSBPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.TransferCompound`
    for details on Tiki-taka-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: TransferCompound(
            unit_cell_devices=[
                ReRamSBPresetDevice(subtract_symmetry_point=True),
                ReRamSBPresetDevice(subtract_symmetry_point=True),
            ],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(),
            transfer_every=1.0,
            units_in_mbatch=True,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class TikiTakaCapacitorPreset(UnitCellRPUConfig):
    """Configuration using Tiki-taka with
    :class:`~aihwkit.simulator.presets.devices.CapacitorPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.TransferCompound`
    for details on Tiki-taka-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: TransferCompound(
            unit_cell_devices=[CapacitorPresetDevice(), CapacitorPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(),
            units_in_mbatch=True,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class TikiTakaEcRamPreset(UnitCellRPUConfig):
    """Configuration using Tiki-taka with
    :class:`~aihwkit.simulator.presets.devices.EcRamPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.TransferCompound`
    for details on Tiki-taka-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: TransferCompound(
            unit_cell_devices=[EcRamPresetDevice(), EcRamPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(),
            units_in_mbatch=True,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class TikiTakaEcRamMOPreset(UnitCellRPUConfig):
    """Configuration using Tiki-taka with
    :class:`~aihwkit.simulator.presets.devices.EcRamMOPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.TransferCompound`
    for details on Tiki-taka-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: TransferCompound(
            unit_cell_devices=[EcRamMOPresetDevice(), EcRamMOPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(),
            units_in_mbatch=True,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class TikiTakaIdealizedPreset(UnitCellRPUConfig):
    """Configuration using Tiki-taka with
    :class:`~aihwkit.simulator.presets.devices.IdealizedPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.TransferCompound`
    for details on Tiki-taka-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: TransferCompound(
            unit_cell_devices=[IdealizedPresetDevice(), IdealizedPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(),
            units_in_mbatch=True,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


# TTv2 configs.


@dataclass
class TTv2ReRamESPreset(UnitCellRPUConfig):
    """Configuration using TTv2 with
    :class:`~aihwkit.simulator.presets.devices.ReRamESPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.ChoppedTransferCompound`
    for details on TTv2-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: ChoppedTransferCompound(
            unit_cell_devices=[ReRamESPresetDevice(), ReRamESPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False
            ),
            units_in_mbatch=False,
            in_chop_prob=0.0,
            fast_lr=0.5,
            auto_scale=True,
            auto_granularity=1000,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=lambda: PresetUpdateParameters(desired_bl=31))


@dataclass
class TTv2ReRamSBPreset(UnitCellRPUConfig):
    """Configuration using TTv2 with
    :class:`~aihwkit.simulator.presets.devices.ReRamSBPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.ChoppedTransferCompound`
    for details on TTv2-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: ChoppedTransferCompound(
            unit_cell_devices=[
                ReRamSBPresetDevice(subtract_symmetry_point=True),
                ReRamSBPresetDevice(subtract_symmetry_point=True),
            ],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False
            ),
            units_in_mbatch=False,
            in_chop_prob=0.0,
            fast_lr=0.5,
            auto_scale=True,
            auto_granularity=1000,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=lambda: PresetUpdateParameters(desired_bl=31))


@dataclass
class TTv2CapacitorPreset(UnitCellRPUConfig):
    """Configuration using TTv2 with
    :class:`~aihwkit.simulator.presets.devices.CapacitorPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.ChoppedTransferCompound`
    for details on TTv2-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: ChoppedTransferCompound(
            unit_cell_devices=[CapacitorPresetDevice(), CapacitorPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False
            ),
            units_in_mbatch=False,
            in_chop_prob=0.0,
            fast_lr=0.1,
            auto_scale=True,
            auto_granularity=1000,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=lambda: PresetUpdateParameters(desired_bl=31))


@dataclass
class TTv2EcRamPreset(UnitCellRPUConfig):
    """Configuration using TTv2 with
    :class:`~aihwkit.simulator.presets.devices.EcRamPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.ChoppedTransferCompound`
    for details on TTv2-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: ChoppedTransferCompound(
            unit_cell_devices=[EcRamPresetDevice(), EcRamPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False
            ),
            units_in_mbatch=False,
            in_chop_prob=0.0,
            fast_lr=0.1,
            auto_scale=True,
            auto_granularity=500,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=lambda: PresetUpdateParameters(desired_bl=100))


@dataclass
class TTv2EcRamMOPreset(UnitCellRPUConfig):
    """Configuration using TTv2 with
    :class:`~aihwkit.simulator.presets.devices.EcRamMOPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.ChoppedTransferCompound`
    for details on TTv2-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: ChoppedTransferCompound(
            unit_cell_devices=[EcRamMOPresetDevice(), EcRamMOPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False
            ),
            units_in_mbatch=False,
            in_chop_prob=0.0,
            fast_lr=0.1,
            auto_scale=True,
            auto_granularity=500,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=lambda: PresetUpdateParameters(desired_bl=100))


@dataclass
class TTv2IdealizedPreset(UnitCellRPUConfig):
    """Configuration using TTv2 with
    :class:`~aihwkit.simulator.presets.devices.IdealizedPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.ChoppedTransferCompound`
    for details on TTv2-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: ChoppedTransferCompound(
            unit_cell_devices=[IdealizedPresetDevice(), IdealizedPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False
            ),
            units_in_mbatch=False,
            in_chop_prob=0.0,
            fast_lr=0.1,
            auto_scale=True,
            auto_granularity=500,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=lambda: PresetUpdateParameters(desired_bl=100))


# Chopped-TTv2 configs.


@dataclass
class ChoppedTTv2ReRamESPreset(UnitCellRPUConfig):
    """Configuration using ChoppedTTv2 with
    :class:`~aihwkit.simulator.presets.devices.ReRamESPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.ChoppedTransferCompound`
    for details on TTv2-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: ChoppedTransferCompound(
            unit_cell_devices=[ReRamESPresetDevice(), ReRamESPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False
            ),
            units_in_mbatch=False,
            in_chop_prob=0.01,
            fast_lr=0.5,
            auto_scale=True,
            auto_granularity=1000,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=lambda: PresetUpdateParameters(desired_bl=31))


@dataclass
class ChoppedTTv2ReRamSBPreset(UnitCellRPUConfig):
    """Configuration using ChoppedTTv2 with
    :class:`~aihwkit.simulator.presets.devices.ReRamSBPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.ChoppedTransferCompound`
    for details on ChoppedTTv2-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: ChoppedTransferCompound(
            unit_cell_devices=[
                ReRamSBPresetDevice(subtract_symmetry_point=True),
                ReRamSBPresetDevice(subtract_symmetry_point=True),
            ],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False
            ),
            units_in_mbatch=False,
            in_chop_prob=0.01,
            fast_lr=0.5,
            auto_scale=True,
            auto_granularity=1000,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=lambda: PresetUpdateParameters(desired_bl=31))


@dataclass
class ChoppedTTv2CapacitorPreset(UnitCellRPUConfig):
    """Configuration using ChoppedTTv2 with
    :class:`~aihwkit.simulator.presets.devices.CapacitorPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.ChoppedTransferCompound`
    for details on ChoppedTTv2-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: ChoppedTransferCompound(
            unit_cell_devices=[CapacitorPresetDevice(), CapacitorPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False
            ),
            units_in_mbatch=False,
            in_chop_prob=0.01,
            fast_lr=0.1,
            auto_scale=True,
            auto_granularity=1000,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=lambda: PresetUpdateParameters(desired_bl=31))


@dataclass
class ChoppedTTv2EcRamPreset(UnitCellRPUConfig):
    """Configuration using ChoppedTTv2 with
    :class:`~aihwkit.simulator.presets.devices.EcRamPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.ChoppedTransferCompound`
    for details on TTv2-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: ChoppedTransferCompound(
            unit_cell_devices=[EcRamPresetDevice(), EcRamPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False
            ),
            auto_granularity=500,
            in_chop_prob=0.01,
            units_in_mbatch=False,
            auto_scale=True,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=lambda: PresetUpdateParameters(desired_bl=100))


@dataclass
class ChoppedTTv2EcRamMOPreset(UnitCellRPUConfig):
    """Configuration using ChoppedTTv2 with
    :class:`~aihwkit.simulator.presets.devices.EcRamMOPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.ChoppedTransferCompound`
    for details on TTv2-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: ChoppedTransferCompound(
            unit_cell_devices=[EcRamMOPresetDevice(), EcRamMOPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False
            ),
            units_in_mbatch=False,
            in_chop_prob=0.001,
            fast_lr=0.1,
            auto_scale=True,
            auto_granularity=500,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=lambda: PresetUpdateParameters(desired_bl=100))


@dataclass
class ChoppedTTv2IdealizedPreset(UnitCellRPUConfig):
    """Configuration using ChoppedTTv2 with
    :class:`~aihwkit.simulator.presets.devices.IdealizedPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.ChoppedTransferCompound`
    for details on TTv2-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: ChoppedTransferCompound(
            unit_cell_devices=[IdealizedPresetDevice(), IdealizedPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False
            ),
            units_in_mbatch=False,
            in_chop_prob=0.001,
            fast_lr=0.1,
            auto_scale=True,
            auto_granularity=500,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=lambda: PresetUpdateParameters(desired_bl=100))


# AGAD configs.


@dataclass
class AGADReRamESPreset(UnitCellRPUConfig):
    """Configuration using AGAD with
    :class:`~aihwkit.simulator.presets.devices.ReRamESPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.DynamicTransferCompound`
    for details on TTv2-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: DynamicTransferCompound(
            unit_cell_devices=[ReRamESPresetDevice(), ReRamESPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False
            ),
            fast_lr=0.5,
            auto_granularity=1000,
            tail_weightening=5.0,
            in_chop_prob=0.02,
            auto_scale=True,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=lambda: PresetUpdateParameters(desired_bl=31))


@dataclass
class AGADReRamSBPreset(UnitCellRPUConfig):
    """Configuration using AGAD with
    :class:`~aihwkit.simulator.presets.devices.ReRamSBPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.DynamicTransferCompound`
    for details on AGAD-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: DynamicTransferCompound(
            unit_cell_devices=[
                ReRamSBPresetDevice(subtract_symmetry_point=True),
                ReRamSBPresetDevice(subtract_symmetry_point=True),
            ],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False
            ),
            fast_lr=0.5,
            auto_granularity=1000,
            tail_weightening=5.0,
            in_chop_prob=0.02,
            auto_scale=True,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=lambda: PresetUpdateParameters(desired_bl=31))


@dataclass
class AGADCapacitorPreset(UnitCellRPUConfig):
    """Configuration using AGAD with
    :class:`~aihwkit.simulator.presets.devices.CapacitorPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.DynamicTransferCompound`
    for details on AGAD-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: DynamicTransferCompound(
            unit_cell_devices=[CapacitorPresetDevice(), CapacitorPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False
            ),
            units_in_mbatch=False,
            fast_lr=0.1,
            auto_granularity=1000,
            tail_weightening=20.0,
            in_chop_prob=0.01,
            auto_scale=True,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=lambda: PresetUpdateParameters(desired_bl=31))


@dataclass
class AGADEcRamPreset(UnitCellRPUConfig):
    """Configuration using AGAD with
    :class:`~aihwkit.simulator.presets.devices.EcRamPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.DynamicTransferCompound`
    for details on TTv2-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: DynamicTransferCompound(
            unit_cell_devices=[EcRamPresetDevice(), EcRamPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False
            ),
            fast_lr=0.1,
            auto_granularity=750,
            tail_weightening=50.0,
            in_chop_prob=0.005,
            units_in_mbatch=False,
            auto_scale=True,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=lambda: PresetUpdateParameters(desired_bl=31))


@dataclass
class AGADEcRamMOPreset(UnitCellRPUConfig):
    """Configuration using AGAD with
    :class:`~aihwkit.simulator.presets.devices.EcRamMOPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.DynamicTransferCompound`
    for details on TTv2-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: DynamicTransferCompound(
            unit_cell_devices=[EcRamMOPresetDevice(), EcRamMOPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False
            ),
            fast_lr=0.1,
            auto_granularity=500,
            tail_weightening=50.0,
            in_chop_prob=0.005,
            units_in_mbatch=False,
            auto_scale=True,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=lambda: PresetUpdateParameters(desired_bl=100))


@dataclass
class AGADIdealizedPreset(UnitCellRPUConfig):
    """Configuration using AGAD with
    :class:`~aihwkit.simulator.presets.devices.IdealizedPresetDevice`.

    See :class:`~aihwkit.simulator.configs.devices.DynamicTransferCompound`
    for details on TTv2-like optimizers.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: UnitCell = field(
        default_factory=lambda: DynamicTransferCompound(
            unit_cell_devices=[IdealizedPresetDevice(), IdealizedPresetDevice()],
            transfer_forward=PresetIOParameters(
                noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE
            ),
            transfer_update=PresetUpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False
            ),
            fast_lr=0.1,
            auto_granularity=500,
            tail_weightening=50.0,
            in_chop_prob=0.005,
            units_in_mbatch=False,
            auto_scale=True,
        )
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=lambda: PresetUpdateParameters(desired_bl=100))


# Mixed precision presets


@dataclass
class MixedPrecisionReRamESPreset(DigitalRankUpdateRPUConfig):
    """Configuration using Mixed-precision with
    class:`~aihwkit.simulator.presets.devices.ReRamESPresetDevice`

    See
    class:`~aihwkit.simulator.configs.devices.MixedPrecisionCompound`
    for details on the mixed precision optimizer.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: DigitalRankUpdateCell = field(
        default_factory=lambda: MixedPrecisionCompound(device=ReRamESPresetDevice())
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class MixedPrecisionReRamSBPreset(DigitalRankUpdateRPUConfig):
    """Configuration using Mixed-precision with
    class:`~aihwkit.simulator.presets.devices.ReRamSBPresetDevice`.

    See class:`~aihwkit.simulator.configs.devices.MixedPrecisionCompound`
    for details on the mixed precision optimizer.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: DigitalRankUpdateCell = field(
        default_factory=lambda: MixedPrecisionCompound(device=ReRamSBPresetDevice())
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class MixedPrecisionCapacitorPreset(DigitalRankUpdateRPUConfig):
    """Configuration using Mixed-precision with
    class:`~aihwkit.simulator.presets.devices.CapacitorPresetDevice`.

    See class:`~aihwkit.simulator.configs.devices.MixedPrecisionCompound`
    for details on the mixed precision optimizer.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: DigitalRankUpdateCell = field(
        default_factory=lambda: MixedPrecisionCompound(device=CapacitorPresetDevice())
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class MixedPrecisionEcRamPreset(DigitalRankUpdateRPUConfig):
    """Configuration using Mixed-precision with
    class:`~aihwkit.simulator.presets.devices.EcRamPresetDevice`.

    See class:`~aihwkit.simulator.configs.devices.MixedPrecisionCompound`
    for details on the mixed precision optimizer.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: DigitalRankUpdateCell = field(
        default_factory=lambda: MixedPrecisionCompound(device=EcRamPresetDevice())
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class MixedPrecisionEcRamMOPreset(DigitalRankUpdateRPUConfig):
    """Configuration using Mixed-precision with
    class:`~aihwkit.simulator.presets.devices.EcRamMOPresetDevice`.

    See class:`~aihwkit.simulator.configs.devices.MixedPrecisionCompound`
    for details on the mixed precision optimizer.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: DigitalRankUpdateCell = field(
        default_factory=lambda: MixedPrecisionCompound(device=EcRamMOPresetDevice())
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class MixedPrecisionIdealizedPreset(DigitalRankUpdateRPUConfig):
    """Configuration using Mixed-precision with
    class:`~aihwkit.simulator.presets.devices.IdealizedPresetDevice`.

    See class:`~aihwkit.simulator.configs.devices.MixedPrecisionCompound`
    for details on the mixed precision optimizer.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: DigitalRankUpdateCell = field(
        default_factory=lambda: MixedPrecisionCompound(device=IdealizedPresetDevice())
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class MixedPrecisionGokmenVlasovPreset(DigitalRankUpdateRPUConfig):
    """Configuration using Mixed-precision with
    class:`~aihwkit.simulator.presets.devices.GokmenVlasovPresetDevice`.

    See class:`~aihwkit.simulator.configs.devices.MixedPrecisionCompound`
    for details on the mixed precision optimizer.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: DigitalRankUpdateCell = field(
        default_factory=lambda: MixedPrecisionCompound(device=GokmenVlasovPresetDevice())
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class MixedPrecisionPCMPreset(DigitalRankUpdateRPUConfig):
    """Configuration using Mixed-precision with
    class:`~aihwkit.simulator.presets.devices.PCMPresetDevice`.

    See class:`~aihwkit.simulator.configs.devices.MixedPrecisionCompound`
    for details on the mixed precision optimizer.

    The default peripheral hardware
    (:class:`~aihwkit.simulator.presets.utils.PresetIOParameters`) and
    analog update
    (:class:`~aihwkit.simulator.presets.utils.PresetUpdateParameters`)
    configuration is used otherwise.
    """

    device: DigitalRankUpdateCell = field(
        default_factory=lambda: MixedPrecisionCompound(device=PCMPresetUnitCell())
    )
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)

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

"""RPU configurations presets for resistive processing units."""

from dataclasses import dataclass, field

from aihwkit.simulator.configs.configs import (
    SingleRPUConfig, UnitCellRPUConfig, DigitalRankUpdateRPUConfig
)
from aihwkit.simulator.configs.devices import (
    PulsedDevice, TransferCompound, UnitCell, VectorUnitCell,
    DigitalRankUpdateCell, MixedPrecisionCompound
)
from aihwkit.simulator.configs.utils import (
    IOParameters, UpdateParameters, VectorUnitCellUpdatePolicy
)
from aihwkit.simulator.presets.devices import (
    CapacitorPresetDevice, EcRamPresetDevice, IdealizedPresetDevice,
    ReRamESPresetDevice, ReRamSBPresetDevice, GokmenVlasovPresetDevice
)
from aihwkit.simulator.presets.utils import (
    PresetIOParameters, PresetUpdateParameters
)


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
    """Preset configuration using a single EcRAM device, see
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

    device: UnitCell = field(default_factory=lambda: VectorUnitCell(
        unit_cell_devices=[ReRamESPresetDevice(), ReRamESPresetDevice()],
        update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM
    ))
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

    device: UnitCell = field(default_factory=lambda: VectorUnitCell(
        unit_cell_devices=[ReRamSBPresetDevice(), ReRamSBPresetDevice()],
        update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM
    ))
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

    device: UnitCell = field(default_factory=lambda: VectorUnitCell(
        unit_cell_devices=[CapacitorPresetDevice(), CapacitorPresetDevice()],
        update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM
    ))
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class EcRam2Preset(UnitCellRPUConfig):
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

    device: UnitCell = field(default_factory=lambda: VectorUnitCell(
        unit_cell_devices=[EcRamPresetDevice(), EcRamPresetDevice()],
        update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM
    ))
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

    device: UnitCell = field(default_factory=lambda: VectorUnitCell(
        unit_cell_devices=[IdealizedPresetDevice(), IdealizedPresetDevice()],
        update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM
    ))
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

    device: UnitCell = field(default_factory=lambda: VectorUnitCell(
        unit_cell_devices=[ReRamESPresetDevice(), ReRamESPresetDevice(),
                           ReRamESPresetDevice(), ReRamESPresetDevice()],
        update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM
    ))
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

    device: UnitCell = field(default_factory=lambda: VectorUnitCell(
        unit_cell_devices=[ReRamSBPresetDevice(), ReRamSBPresetDevice(),
                           ReRamSBPresetDevice(), ReRamSBPresetDevice()],
        update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM
    ))
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

    device: UnitCell = field(default_factory=lambda: VectorUnitCell(
        unit_cell_devices=[CapacitorPresetDevice(), CapacitorPresetDevice(),
                           CapacitorPresetDevice(), CapacitorPresetDevice()],
        update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM
    ))
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


@dataclass
class EcRam4Preset(UnitCellRPUConfig):
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

    device: UnitCell = field(default_factory=lambda: VectorUnitCell(
        unit_cell_devices=[EcRamPresetDevice(), EcRamPresetDevice(),
                           EcRamPresetDevice(), EcRamPresetDevice()],
        update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM
    ))
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

    device: UnitCell = field(default_factory=lambda: VectorUnitCell(
        unit_cell_devices=[IdealizedPresetDevice(), IdealizedPresetDevice(),
                           IdealizedPresetDevice(), IdealizedPresetDevice()],
        update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM
    ))
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
            transfer_forward=PresetIOParameters(),
            transfer_update=PresetUpdateParameters(),
            transfer_every=1.0,
            units_in_mbatch=True,
            ))
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
            unit_cell_devices=[ReRamSBPresetDevice(), ReRamSBPresetDevice()],
            transfer_forward=PresetIOParameters(),
            transfer_update=PresetUpdateParameters(),
            transfer_every=1.0,
            units_in_mbatch=True,
            ))
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
            transfer_forward=PresetIOParameters(),
            transfer_update=PresetUpdateParameters(),
            transfer_every=1.0,
            units_in_mbatch=True,
            ))
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
            transfer_forward=PresetIOParameters(),
            transfer_update=PresetUpdateParameters(),
            transfer_every=1.0,
            units_in_mbatch=True,
            ))
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
            transfer_forward=PresetIOParameters(),
            transfer_update=PresetUpdateParameters(),
            transfer_every=1.0,
            units_in_mbatch=True,
            ))
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)


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
        default_factory=lambda: MixedPrecisionCompound(
            device=ReRamESPresetDevice(),
        ))
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
        default_factory=lambda: MixedPrecisionCompound(
            device=ReRamSBPresetDevice(),
        ))
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
        default_factory=lambda: MixedPrecisionCompound(
            device=CapacitorPresetDevice(),
        ))
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
        default_factory=lambda: MixedPrecisionCompound(
            device=EcRamPresetDevice(),
        ))
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
        default_factory=lambda: MixedPrecisionCompound(
            device=IdealizedPresetDevice(),
        ))
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
        default_factory=lambda: MixedPrecisionCompound(
            device=GokmenVlasovPresetDevice(),
        ))
    forward: IOParameters = field(default_factory=PresetIOParameters)
    backward: IOParameters = field(default_factory=PresetIOParameters)
    update: UpdateParameters = field(default_factory=PresetUpdateParameters)

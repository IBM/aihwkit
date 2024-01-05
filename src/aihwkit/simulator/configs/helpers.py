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

"""Helper for generating presets."""

from typing import Union, Type, Callable, Any
from copy import deepcopy

from aihwkit.exceptions import ArgumentError
from aihwkit.simulator.configs.configs import (
    UnitCellRPUConfig,
    SingleRPUConfig,
    DigitalRankUpdateRPUConfig,
)
from aihwkit.simulator.configs.devices import PulsedDevice
from aihwkit.simulator.configs.compounds import (
    TransferCompound,
    ChoppedTransferCompound,
    DynamicTransferCompound,
    MixedPrecisionCompound,
    VectorUnitCell,
)
from aihwkit.simulator.parameters.training import UpdateParameters
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.parameters.enums import (
    VectorUnitCellUpdatePolicy,
    NoiseManagementType,
    BoundManagementType,
)


def build_config(
    algorithm: str,
    device: Union[Type[PulsedDevice], PulsedDevice, Callable],
    io_parameters: Union[Type[IOParameters], IOParameters, Callable] = IOParameters,
    up_parameters: Union[Type[UpdateParameters], UpdateParameters, Callable] = UpdateParameters,
    n_devices: int = 1,
    construction_seed: int = 0,
    **kwargs: Any,
) -> Union[UnitCellRPUConfig, SingleRPUConfig, DigitalRankUpdateRPUConfig]:
    """Generate a RPU configuration for analog training using a
    specific device model and a given training algoithm.

    Args:
        algorithm: The type of the training algorithm. Valid choices are:

            "sgd": Random pulsed (naive) SGD on analog crossbars.  See
                `Gokmen & Vlasov, Front. Neurosci. 2016`_ for details.

            "mp", "mixed-precision": Mixed-precision analog, where the
                gradient is computed in digital and only the forward
                abd backward pass is in analog. Uses
                :class:`~aihwkit.simulator.configs.compounds.MixedPrecisionCompound`. See
                also `Nandakumar et al. Front. in Neurosci. (2020)`_
                for details.

            "tiki-taka", "ttv1", "tt": Tiki-taka I algorithm. Uses
                :class:`~aihwkit.simulator.configs.compounds.TransferCompound`.
                See `Gokmen & Haensch, Front. Neurosci. 2020`_ for
                details.

            "ttv2": second version of the Tiki-taka algorithm
                (TTv2). Uses
                :class:`~aihwkit.simulator.configs.compounds.ChoppedTransferCompound`
                with chopper probabilty set to 0. See `Gokmen,
                Front. Artif. Intell. 2021`_ for details.

            "chopped-ttv2", "ttv3", "c-ttv2": Chopped version of TTv2
                algorithm. Uses
                :class:`~aihwkit.simulator.configs.compounds.ChoppedTransferCompound`.
                See `Rasch et al., ArXiv 2023`_ for details.

            "agad", "ttv4": Analog gradient accumulation with dynamic
                reference computation. Uses
                :class:`~aihwkit.simulator.configs.compounds.StatsticalransferCompound`. See
                `Rasch et al., ArXiv 2023`_ for details.

        device: Device configuration of the analog devices. Can be the
            class or the actual device. All available device will have the
            same configuration.

        io_parameters: IOParameters class (or actual instance) that
            are used for forward / backward and transfer. Default is
            :class:`~aihwkit.simulator.config.IOParameters`.

        up_parameters: UpdateParameters class (or actual instance) that are used for update and
            transfer update. Default is
            :class:`~aihwkit.simulator.config.UpdateParameters`.

        n_devices: In case of SGD, how many device pairs are used in the unit cell.

            Note:
                This option is only applied for ``algorithm="sgd"``
                and ignored for all other algorithm choices.

        construction_seed: Seed of the construction

        kwargs: Other RPUConfig fields to assign explicitely (e.g. ``mapping``).

    Returns:
        RPU config according to the algorithm and device settings.

    Raises:
        ArgumentError: in case algorithm is not known

    .. _`Gokmen & Vlasov, Front. Neurosci. 2016`: \
    https://www.frontiersin.org/articles/10.3389/fnins.2016.00333/full
    .. _`Gokmen & Haensch, Front. Neurosci. 2020`: \
    https://www.frontiersin.org/articles/10.3389/fnins.2020.00103/full
    .. _`Gokmen, Front. Artif. Intell. 2021`: \
    https://www.frontiersin.org/articles/10.3389/frai.2021.699148/full
    .. _`Rasch et al., ArXiv 2023`: \
    https://arxiv.org/abs/2303.04721
    .. _`Nandakumar et al. Front. in Neurosci. (2020)`: \
    https://doi.org/10.3389/fnins.2020.00406

    """
    # pylint: disable=too-many-statements, too-many-return-statements

    if isinstance(device, PulsedDevice):
        device_to_use = device

        def device_fun(**kwargs: Any) -> PulsedDevice:
            dev = deepcopy(device_to_use)
            dev.__dict__.update(**kwargs)
            return dev

        device = device_fun

    if isinstance(io_parameters, IOParameters):
        io_pars_to_use = io_parameters

        def io_pars_fun(**kwargs: Any) -> IOParameters:
            io_pars = deepcopy(io_pars_to_use)
            io_pars.__dict__.update(**kwargs)
            return io_pars

        io_parameters = io_pars_fun

    if isinstance(up_parameters, UpdateParameters):
        up_pars_to_use = up_parameters

        def up_pars_fun(**kwargs: Any) -> UpdateParameters:
            up_pars = deepcopy(up_pars_to_use)
            up_pars.__dict__.update(**kwargs)
            return up_pars

        up_parameters = up_pars_fun

    if algorithm.lower() in ["sgd"]:
        if n_devices == 1:
            return SingleRPUConfig(
                device=device(construction_seed=construction_seed),
                forward=io_parameters(),
                backward=io_parameters(),
                update=up_parameters(),
                **kwargs,
            )
        return UnitCellRPUConfig(
            device=VectorUnitCell(
                unit_cell_devices=[device() for _ in range(n_devices)],
                update_policy=VectorUnitCellUpdatePolicy.SINGLE_RANDOM,
                construction_seed=construction_seed,
            ),
            forward=io_parameters(),
            backward=io_parameters(),
            update=up_parameters(),
            **kwargs,
        )

    if algorithm.lower() in ["tiki-taka", "tt", "ttv1"]:
        return UnitCellRPUConfig(
            device=TransferCompound(
                unit_cell_devices=[device(), device()],
                transfer_forward=io_parameters(
                    noise_management=NoiseManagementType.NONE,
                    bound_management=BoundManagementType.NONE,
                ),
                transfer_update=up_parameters(),
                units_in_mbatch=True,
                construction_seed=construction_seed,
            ),
            forward=io_parameters(),
            backward=io_parameters(),
            update=up_parameters(),
            **kwargs,
        )
    if algorithm.lower() in ["ttv2"]:
        return UnitCellRPUConfig(
            device=ChoppedTransferCompound(
                unit_cell_devices=[device(), device()],
                transfer_forward=io_parameters(
                    noise_management=NoiseManagementType.NONE,
                    bound_management=BoundManagementType.NONE,
                ),
                transfer_update=up_parameters(
                    desired_bl=1, update_bl_management=False, update_management=False
                ),
                in_chop_prob=0.0,
                units_in_mbatch=False,
                auto_scale=False,
                construction_seed=construction_seed,
            ),
            forward=io_parameters(),
            backward=io_parameters(),
            update=up_parameters(desired_bl=5),
            **kwargs,
        )
    if algorithm.lower() in ["chopped-ttv2", "ttv3", "c-ttv2"]:
        return UnitCellRPUConfig(
            device=ChoppedTransferCompound(
                unit_cell_devices=[device(), device()],
                transfer_forward=io_parameters(
                    noise_management=NoiseManagementType.NONE,
                    bound_management=BoundManagementType.NONE,
                ),
                transfer_update=up_parameters(
                    desired_bl=1, update_bl_management=False, update_management=False
                ),
                units_in_mbatch=False,
                fast_lr=0.1,
                auto_scale=True,
                construction_seed=construction_seed,
            ),
            forward=io_parameters(),
            backward=io_parameters(),
            update=up_parameters(desired_bl=5),
            **kwargs,
        )
    if algorithm.lower() in ["agad", "ttv4"]:
        return UnitCellRPUConfig(
            device=DynamicTransferCompound(
                unit_cell_devices=[device(), device()],
                transfer_forward=io_parameters(),
                transfer_update=up_parameters(
                    desired_bl=1, update_bl_management=True, update_management=True
                ),
                auto_scale=True,
                fast_lr=0.1,
                units_in_mbatch=False,
                construction_seed=construction_seed,
            ),
            forward=io_parameters(),
            backward=io_parameters(),
            update=up_parameters(desired_bl=5),
            **kwargs,
        )
    if algorithm.lower() in ["mp", "mixed-precision"]:
        return DigitalRankUpdateRPUConfig(
            device=MixedPrecisionCompound(device=device(construction_seed=construction_seed)),
            forward=io_parameters(),
            backward=io_parameters(),
            update=up_parameters(),
            **kwargs,
        )

    raise ArgumentError("Algorithm {} is not known".format(algorithm))

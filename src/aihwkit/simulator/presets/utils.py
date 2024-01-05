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

"""Utils for configurations presets for resistive processing units."""

# pylint: disable=too-many-instance-attributes

from dataclasses import dataclass

from aihwkit.simulator.parameters import (
    BoundManagementType,
    IOParameters,
    NoiseManagementType,
    PulseType,
    UpdateParameters,
    WeightNoiseType,
)


@dataclass
class PresetIOParameters(IOParameters):
    r"""Preset for the forward and backward pass parameters.

    This defines the default (peripheral) hardware configurations
    used for most presets.

    Currently, we assume 7 bit DAC (without stochastic rounding) and 9
    bit ADC (including the sign) and a fixed dynamic range ratio. The
    dynamic range ratio is defined by how many fully-on inputs would
    saturate the output ADC when all weights are set maximal
    conductances. This value is set to 20 here, as the weight range is
    normalized to :math:`-1,\ldots 1`.

    Moreover, the output noise (additive Gaussian) is set to 0.1,
    which is on the order of 1 LSB of the ADC.

    By default, we turned additional weight noise off, however, some
    presets might turn it on as required by the device specification.

    Finally, we assume by default that the device is run with bound
    management (see
    :class:`~aihwkit.simulator.parameters.enums.BoundManagementType`) and
    noise management (see
    :class:`~aihwkit.simulator.parameters.enums.NoiseManagementType`)
    turned on to `ITERATIVE` and `ABS_MAX`, respectively.
    """

    bound_management: BoundManagementType = BoundManagementType.ITERATIVE
    noise_management: NoiseManagementType = NoiseManagementType.ABS_MAX

    inp_res: float = 1.0 / (2**7 - 2)  # 7 bit DAC.
    inp_sto_round: bool = False

    out_bound: float = 20.0
    out_noise: float = 0.1
    out_res: float = 1.0 / (2**9 - 2)  # 9 bit ADC.

    # No read noise by default.
    w_noise: float = 0.0
    w_noise_type: WeightNoiseType = WeightNoiseType.NONE


@dataclass
class PresetUpdateParameters(UpdateParameters):
    """Preset for the general update behavior.

    This defines the default update configurations used for most
    presets. Presets might override this default behavior to implement
    other analog SGD optimizers.

    Parallel analog update is the default. We assume stochastic pulse
    to do the parallel update in analog, as described in `Gokmen &
    Vlasov, Front. Neurosci. 2016`_.

    Moreover, we assume that the pulse length is dynamically adjusted
    with a maximal pulse length of 31 pulses.

    .. _`Gokmen & Vlasov, Front. Neurosci. 2016`: \
        https://www.frontiersin.org/articles/10.3389/fnins.2016.00333/full
    """

    desired_bl: int = 31  # Less than 32 preferable (faster implementation).
    pulse_type: PulseType = PulseType.STOCHASTIC_COMPRESSED
    update_bl_management: bool = True  # Dynamically adjusts pulse train length (max 31).
    update_management: bool = True


@dataclass
class StandardIOParameters(IOParameters):
    r"""Preset for the forward and backward pass parameters.

    Preset that is more aligned with the the forward pass of
    :class:`~aihwkit.simulator.presets.configs.StandardHWATrainingPreset`,
    as it assumes the same DAC/ ADC resolution, output bound and
    output noise (see also `Rasch et al. ArXiv 2023`_ for a discussion)

    However, here, noise and bound mangement is turned on by default,
    and IR-drop as well as short-term weight noise is set to 0 by
    default.

    .. _`Rasch et al. ArXiv 2023`: https://arxiv.org/abs/2302.08469
    """

    bound_management: BoundManagementType = BoundManagementType.ITERATIVE
    noise_management: NoiseManagementType = NoiseManagementType.ABS_MAX

    inp_res: float = 1.0 / (2**8 - 2)  # 8 bit DAC.
    inp_sto_round: bool = False

    out_bound: float = 10.0
    out_noise: float = 0.04
    out_res: float = 1.0 / (2**8 - 2)  # 8 bit ADC.

    # No read noise by default.
    w_noise: float = 0.0
    w_noise_type: WeightNoiseType = WeightNoiseType.NONE

# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Device configurations presets for resistive processing units."""

# pylint: disable=too-many-instance-attributes

from dataclasses import dataclass
from aihwkit.simulator.configs.devices import (
    ConstantStepDevice,
    ExpStepDevice,
    LinearStepDevice,
    SoftBoundsReferenceDevice,
)


@dataclass
class ReRamESPresetDevice(ExpStepDevice):
    """Preset configuration for a single RRAM analog resistive processing
    unit based on exp. step device.

    Fit of the model :class:`ExpStepDevice` to  `Gong & al., Nat. Commun., 2018`_.

    .. _`Gong & al., Nat. Commun., 2018`: https://www.nature.com/articles/s41467-018-04485-1
    """

    # pylint: disable=invalid-name

    dw_min: float = 0.00135
    up_down: float = 0.259359

    w_max: float = 1.0
    w_min: float = -1.0

    a: float = -0.5
    b: float = -0.5
    gamma_up: float = 5.0
    gamma_down: float = 5.0
    A_up: float = -1.18445
    A_down: float = -0.081404

    # Device-to-device var.
    dw_min_dtod: float = 0.2  # a little reduced compared to SB because of non-linearity
    up_down_dtod: float = 0.05

    w_max_dtod: float = 0.3
    w_min_dtod: float = 0.3

    # Cycle-to_cycle.
    dw_min_std: float = 5.0
    write_noise_std: float = 75.0


@dataclass
class ReRamSBPresetDevice(SoftBoundsReferenceDevice):
    """Preset configuration for a single ReRAM analog resistive processing
    unit based on soft bounds device.

    Loose fit of the model :class:`SoftBoundsDevice` to  `Gong & al., Nat. Commun., 2018`_.

    Note:
        Here it is assumed that the devices have been calibrated to the symmetry
        point by subtracting a reference device (which is in this case not
        explicitly modeled). For a more accurate fit see :class:`ReRamESPresetDevice`.

    .. _`Gong & al., Nat. Commun., 2018`: https://www.nature.com/articles/s41467-018-04485-1
    """

    dw_min: float = 0.002
    up_down: float = 0.0

    w_max: float = 1.25
    w_min: float = -0.75

    mult_noise: bool = False

    # Device-to-device var.
    dw_min_dtod: float = 0.3
    up_down_dtod: float = 0.01  # assumes symmetry point corrected.

    w_max_dtod: float = 0.3 / 1.25
    w_min_dtod: float = 0.3 / 0.75

    # Cycle-to_cycle.
    dw_min_std: float = 3.75
    write_noise_std: float = 56


@dataclass
class CapacitorPresetDevice(LinearStepDevice):
    """Preset configuration for a single capacitor resistive processing
    unit based on linear step device.

    Fit of the model :class:`LinearStepDevice` to  `Li et al., VLSI, 2018`_

    Here some capacitor leakage is assumed as well.

    Caution:
        Capacitor leakage is applied only once per mini-batch and this
        the size of the leakage has to be adapted by the user as it
        depends not only on the size of the leak of the physical
        capacitor but also on the assumptions how much physical time
        is required for a full forward and backward cycle through the
        network (which depends on whether one assumes pipelining or not).

        The parameter ``lifetime`` needs to be adjusted accordingly.

    .. _`Li et al., VLSI, 2018`: https://ieeexplore.ieee.org/abstract/document/8510648
    """

    dw_min: float = 0.005
    up_down: float = 0.0

    w_max: float = 1.0
    w_min: float = -1.0

    mult_noise: bool = False

    gamma_up: float = 0.05  # gamma_up = -slope_up * w_max / dw_min_up
    gamma_down: float = 0.05

    # Device-to-device var.
    dw_min_dtod: float = 0.1
    up_down_dtod: float = 0.06

    w_max_dtod: float = 0.07
    w_min_dtod: float = 0.07

    gamma_up_dtod: float = 0.01
    gamma_down_dtod: float = 0.01

    # Cycle-to_cycle.
    dw_min_std: float = 0.3
    write_noise_std: float = 0.0

    # Slope does not depend on bound.
    mean_bound_reference: bool = True

    # Leakage (in mini-batches).
    lifetime: float = 1.0e6
    lifetime_dtod: float = 0.3


@dataclass
class EcRamPresetDevice(LinearStepDevice):
    """Preset configuration for a single Lithium-based ECRAM resistive processing
    unit based on linear step device.

    Fit of the model :class:`LinearStepDevice` to  `Tang & al., IEDM, 2018`_

    The range is shifted, so that the symmetry point is at zero.

    .. _`Tang & al., IEDM, 2018`: https://ieeexplore.ieee.org/document/8614551
    """

    dw_min: float = 0.002
    up_down: float = 0.0  # assumed shifted onto range

    w_max: float = 1.1724
    w_min: float = -0.8276

    mult_noise: bool = True

    gamma_up: float = 0.1153
    gamma_down: float = 0.5085

    # Device-to-device var.
    dw_min_dtod: float = 0.1
    up_down_dtod: float = 0.01

    w_max_dtod: float = 0.05
    w_min_dtod: float = 0.05

    gamma_up_dtod: float = 0.05
    gamma_down_dtod: float = 0.05

    # Cycle-to_cycle.
    dw_min_std: float = 0.3
    write_noise_std: float = 0.0

    # Slope does not depend on the actual bound.
    mean_bound_reference: bool = True


@dataclass
class EcRamMOPresetDevice(LinearStepDevice):
    """Preset configuration for a single metal-oxide ECRAM resistive processing
    unit based on linear step device.

    Based on data from `Kim et al. IEDM, 2019`_

    .. _`Kim et al. IEDM, 2019`:  https://ieeexplore.ieee.org/document/8993463
    """

    dw_min: float = 2.8214e-4
    up_down: float = 0.0  # assumed shifted onto range

    w_max: float = 1.1714
    w_min: float = -0.8286

    mult_noise: bool = True

    gamma_up: float = 0.4152
    gamma_down: float = 0.7342

    # Device-to-device var.
    dw_min_dtod: float = 0.1
    up_down_dtod: float = 0.01

    w_max_dtod: float = 0.05
    w_min_dtod: float = 0.05

    gamma_up_dtod: float = 0.05
    gamma_down_dtod: float = 0.05

    # Cycle-to_cycle.
    dw_min_std: float = 2.0
    write_noise_std: float = 0.0

    # Slope does not depend on the actual bound.
    mean_bound_reference: bool = True


@dataclass
class IdealizedPresetDevice(ConstantStepDevice):
    """Preset configuration using an idealized device using
    :class:`ConstantStepDevice`.

    (On average) perfectly symmetric device with 10000 steps.

    Definitions are from the specifications listed in `Gokmen &
    Vlasov, Front. Neurosci. 2016`_ (which includes a number of
    device-to-device and cycle-to-cycle variations, see
    :class:`PulsedDevice`), however, setting the device-to-device
    asymmetry term to zero and increasing the number of states by
    roughly 8 (to 10000 states).

    This is the same device used for
    `Rasch, Gokmen & Haensch, IEEE Design & Test, 2019`_.

    .. _`Gokmen & Vlasov, Front. Neurosci. 2016`: \
        https://www.frontiersin.org/articles/10.3389/fnins.2016.00333/full
    .. _`Rasch, Gokmen & Haensch, IEEE Design & Test, 2019`: https://arxiv.org/abs/1906.02698
    """

    dw_min: float = 0.0002  # Factor 5 smaller steps.
    dw_min_dtod: float = 0.3
    dw_min_std: float = 0.3

    up_down: float = 0.0
    up_down_dtod: float = 0.0  # set to zero, since idealized

    w_max: float = 1.0  # Increased range.
    w_min: float = -1.0

    # Device-to-device of range.
    w_max_dtod: float = 0.3
    w_min_dtod: float = 0.3


@dataclass
class GokmenVlasovPresetDevice(ConstantStepDevice):
    """Preset configuration using :class:`ConstantStepDevice`.

    Definitions are (largely) from the specifications listed in `Gokmen &
    Vlasov, Front. Neurosci. 2016`_ (which includes a number of device-to-device
    and cycle-to-cycle variations, see :class:`PulsedDevice`).

    Note, however, that we use some of the algorithmic optimization of the
    follow-up papers as well and here scale everything to the weight range
    ``-1..1``.

    .. _`Gokmen & Vlasov, Front. Neurosci. 2016`: \
        https://www.frontiersin.org/articles/10.3389/fnins.2016.00333/full
    """

    dw_min: float = 0.0016  # to keep 1200 states
    dw_min_dtod: float = 0.3
    dw_min_std: float = 0.3

    up_down: float = 0.0
    up_down_dtod: float = 0.01

    w_max: float = 1.0  # Increased range, parameter adjusted
    w_min: float = -1.0

    # Device-to-device variation of range
    w_max_dtod: float = 0.3
    w_min_dtod: float = 0.3


@dataclass
class PCMPresetDevice(ExpStepDevice):
    """Preset configuration for a single Phase change memory (PCM) analog
    resistive processing unit based on exponential step device model.

    A PCM device based on :math:`Ge_2Sb_2Te_5` described in
    `Nandakumar et al., Front. Neurosci. 2020`_ by using the
    exponential device model with complex cycle-to-cycle noise (see
    :class:`~aihwkit.simulator.configs.device.ExpStepDevice`)

    Note:
        This is an uni-directional device and thus can only be used as
        a plus-minus pair with refresh (see
        :class:`~PCMPresetUnitCell` which is using this device
        combined as a pair based on
        :class:`~aihwkit.simulator.configs.device.OneSidedUnitCell`).

        When the device is reset, device-to-device and cycle-to-cycle
        variation is assumed.

    .. _`Nandakumar et al., Front. Neurosci. 2020`: \
        https://www.frontiersin.org/articles/10.3389/fnins.2020.00406/full
    """

    # pylint: disable=invalid-name

    dw_min: float = 0.01
    up_down: float = 0.0

    w_max: float = 2.0
    w_min: float = 0.0

    a: float = -1.0  # scales with w_max
    b: float = 0.0
    gamma_up: float = 2.5
    gamma_down: float = 2.5
    A_up: float = -27.235
    A_down: float = -2.235

    # Device-to-device var.
    dw_min_dtod: float = 0.2
    up_down_dtod: float = 0.05

    w_max_dtod: float = 0.1
    w_min_dtod: float = 0.0

    # Cycle-to_cycle.
    dw_min_std: float = 0.6
    dw_min_std_add: float = 0.042  # dw_min_std/0.6*0.5/20
    dw_min_std_slope: float = 0.108  # dw_min_std/0.6*1.3/20

    write_noise_std: float = 0.0

    # reset behavior
    reset: float = 0.01
    reset_dtod: float = 0.02


@dataclass
class ReRamArrayOMPresetDevice(SoftBoundsReferenceDevice):
    r"""Preset configuration for a single ReRAM analog resistive processing
    unit based on soft bounds reference device.

    This parameter setting was obtained from ReRAM device array measurements
    as described in `Gong & Rasch et al., IEDM., 2022`_.

    This setting is a fit to the "Optimized Material" ReRAM device
    array in the article.

    Here the :class:`SoftBoundsReferenceDevice` is used as device
    model class, so that the symmetry point can be easily
    subtracted. The subtraction is by default on, assuming 5 \% of
    :math:`w_\max` error (adjustable with ``reference_std``).

    Note:

        Here the weight range is compliance adjusted as described in
        the article.

        The corrupt device probability (which is 13.5 \% for the array
        measured) is by default set to zero. It can be set with the
        ``corrupt_devices_prob`` parameter.

    .. _`Gong & Rasch et al., IEDM., 2022`: \
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10019569

    """

    enforce_consistency: bool = True
    dw_min_dtod_log_normal: bool = True

    dw_min: float = 0.0949
    up_down: float = 0.0

    w_max: float = 1.0  # 1.4839
    w_min: float = -1.0  # -0.6192

    mult_noise: bool = False

    # Device-to-device var.
    dw_min_dtod: float = 0.7829
    up_down_dtod: float = 0.01

    w_max_dtod: float = 0.3499
    w_min_dtod: float = 0.5695

    # Cycle-to_cycle.
    dw_min_std: float = 0.4158
    write_noise_std: float = 1.4113

    corrupt_devices_range: float = 0.0100
    corrupt_devices_prob: float = 0.0  # 0.1348

    subtract_symmetry_point: bool = True
    reference_std: float = 0.05


@dataclass
class ReRamArrayHfO2PresetDevice(SoftBoundsReferenceDevice):
    r"""Preset configuration for a single ReRAM analog resistive processing
    unit based on soft bounds reference device.

    This parameter setting was obtained from ReRAM device array measurements
    as described in `Gong & Rasch et al., IEDM., 2022`_.

    This setting is a fit to the "Baseline HfO2" ReRAM device
    array in the article.

    Here the :class:`SoftBoundsReferenceDevice` is used as device
    model class, so that the symmetry point can be easily
    subtracted. The subtraction is by default on, assuming 5 \% of
    :math:`w_\max` error (adjustable with ``reference_std``).

    Note:

        Here the weight range is compliance adjusted as described in
        the article.

        The corrupt device probability (which is 10 \% for the array
        measured) is by default set to zero. It can be set with the
        ``corrupt_devices_prob`` parameter.

    .. _`Gong & Rasch et al., IEDM., 2022`: \
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10019569

    """

    enforce_consistency: bool = True
    dw_min_dtod_log_normal: bool = True

    dw_min: float = 0.4622
    up_down: float = 0.0

    w_max: float = 1.0  # 1.1490
    w_min: float = -1.0  # -1.0284

    mult_noise: bool = False

    # Device-to-device var.
    dw_min_dtod: float = 0.7125
    up_down_dtod: float = 0.01

    w_max_dtod: float = 0.4295
    w_min_dtod: float = 0.5990

    # Cycle-to_cycle.
    dw_min_std: float = 0.2174
    write_noise_std: float = 0.5841

    corrupt_devices_range: float = 0.0100
    corrupt_devices_prob: float = 0.0  # 0.0977

    subtract_symmetry_point: bool = True
    reference_std: float = 0.05

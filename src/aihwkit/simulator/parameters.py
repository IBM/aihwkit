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

# pylint: disable=too-many-instance-attributes

"""Parameters for resistive devices and tiles."""

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Type

from aihwkit.simulator.rpu_base import devices


# Helper enums.

class BoundManagementType(Enum):
    """Bound management type.

    In the case ``Iterative`` the MAC is iteratively recomputed with
    inputs iteratively halved, when the output bound was hit.
    """

    NONE = 'None'
    """No bound management."""

    ITERATIVE = 'Iterative'
    r"""Iteratively recomputes input scale set to :math:`\alpha\leftarrow\alpha/2`."""


class NoiseManagementType(Enum):
    r"""Noise management type.

    Noise management determines a factor :math:`\alpha` how the input is reduced:

    .. math:: \mathbf{y} = \alpha\;F_\text{analog-mac}\left(\mathbf{x}/\alpha\right)
    """

    NONE = 'None'
    """No noise management."""

    ABS_MAX = 'AbsMax'
    r"""Use :math:`\alpha\equiv\max{|\mathbf{x}|}`."""

    MAX = 'Max'
    r"""Use :math:`\alpha\equiv\max{\mathbf{x}}`."""

    CONSTANT = 'Constant'
    r"""A constant value (given by parameter ``nm_thres``)."""


class OutputWeightNoiseType(Enum):
    r"""Output weight noise type.

    The weight noise is applied for each MAC computation, while not
    touching the actual weight matrix but referring it to the output.

    .. math:: y_i = \sum_j w_{ij}+\xi_{ij}
    """

    NONE = 'None'
    """No weight noise."""

    ADDITIVE_CONSTANT = 'AdditiveConstant'
    r"""
    The :math:`\xi\sim{\cal N}(0,\sigma)` thus all are Gaussian distributed.
    :math:`\sigma` is determined by ``w_noise``.
    """


class PulseType(Enum):
    """Pulse type."""

    NONE = 'None'
    """Floating point update instead of pulses."""

    STOCHASTIC_COMPRESSED = 'StochasticCompressed'
    """Generates actual stochastic bit lines. Plus and minus pulses are taken in the same pass."""

    STOCHASTIC = 'Stochastic'
    """Two passes for plus and minus (only CPU)."""

    NONE_WITH_DEVICE = 'NoneWithDevice'
    """Floating point like ``None``, but with analog devices (e.g. weight clipping)."""

    MEAN_COUNT = 'MeanCount'
    """Coincidence based in prob (:math:`p_a p_b`)."""


# Specialized parameters.

@dataclass
class AnalogTileInputOutputParameters:
    """Parameters that modify the IO behavior."""

    bindings_class: ClassVar[Type] = devices.AnalogTileInputOutputParameter

    bm_test_negative_bound: bool = True

    bound_management: BoundManagementType = BoundManagementType.ITERATIVE
    """Type of bound management, see :class:`BoundManagementType`."""

    inp_bound: float = 1.0
    """Input bound and ranges for the digital-to-analog converter (DAC)."""

    inp_noise: float = 0.0
    r"""Std deviation of Gaussian input noise (:math:`\sigma_\text{inp}`),
    i.e. noisiness of the analog input (at the stage after DAC and
    before the multiplication)."""

    inp_res: float = 1 / (2**7 - 2)
    r"""Number of discretization steps for DAC (:math:`\le0` means infinite steps)
    or resolution (1/steps)."""

    inp_sto_round: bool = False
    """Whether to enable stochastic rounding of DAC."""

    is_perfect: bool = False
    """Short-cut to compute a perfect forward pass. If ``True``, it assumes an
    ideal forward pass (e.g. no bound, ADC etc...). Will disregard all other
    settings in this case."""

    max_bm_factor: int = 1000
    """Maximal bound management factor. If this factor is reached then the
    iterative process is stopped."""

    max_bm_res: float = 0.25
    """Another way to limit the maximal number of iterations of the bound
    management. The max effective resolution number of the inputs, e.g.
    use :math:`1/4` for 2 bits."""

    nm_thres: float = 0.0
    r"""Constant noise management value for ``type`` ``Constant``.

    In other cases, this is a upper threshold :math:`\theta` above
    which the noise management factor is saturated. E.g. for
    `AbsMax`:

    .. math::
       :nowrap:

       \begin{equation*} \alpha=\begin{cases}\max_i|x_i|, &
       \text{if} \max_i|x_i|<\theta \\ \theta, &
       \text{otherwise}\end{cases} \end{equation*}

    Caution:
        If ``nm_thres`` is set (and type is not ``Constant``), the
        noise management will clip some large input values, in
        favor of having a better SNR for smaller input values.
    """

    noise_management: NoiseManagementType = NoiseManagementType.ABS_MAX
    """Type of noise management, see :class:`NoiseManagementType`."""

    out_bound: float = 12.0
    """Output bound and ranges for analog-to-digital converter (ADC)."""

    out_noise: float = 0.06
    r"""Std deviation of Gaussian output noise (:math:`\sigma_\text{out}`),
    i.e. noisiness of device summation at the output."""

    out_res: float = 1 / (2**9 - 2)
    """Number of discretization steps for ADC (:math:`<=0` means infinite steps)
    or resolution (1/steps)."""

    out_scale: float = 1.0
    """Additional fixed scalar factor."""

    out_sto_round: bool = False
    """Whether to enable stochastic rounding of ADC."""

    w_noise: float = 0.0
    r"""Scale of output referred weight noise (:math:`\sigma_w`) for a given
    ``w_noise_type``."""

    w_noise_type: OutputWeightNoiseType = OutputWeightNoiseType.NONE
    """Type as specified in :class:`OutputWeightNoiseType`.

    Note:

     This noise us applied each time anew as it is referred to
     the output. It will not change the conductance values of
     the weight matrix. For the latter one can apply
     :meth:`diffuse_weights`.
    """


@dataclass
class AnalogTileBackwardInputOutputParameters(AnalogTileInputOutputParameters):
    """Parameters that modify the backward IO behavior.

    This class contains the same parameters as
    ``AnalogTileInputOutputParameters``, specializing the default value of
    ``bound_management`` (as backward does not support bound management).
    """

    bound_management: BoundManagementType = BoundManagementType.NONE
    """Type of noise management, see :class:`NoiseManagementType`."""


@dataclass
class AnalogTileUpdateParameters:
    """Parameter that modify the update behaviour of a pulsed device."""

    bindings_class: ClassVar[Type] = devices.AnalogTileUpdateParameter

    desired_bl: int = 31
    """Desired length of the pulse trains. For update BL management, it is the
    maximal pulse train length."""

    fixed_bl: bool = True
    """Whether to fix the length of the pulse trains (however, see ``update_bl_management``).

    In case of ``True`` (where ``dw_min`` is the mean minimal weight change step size) it is::

        BL = desired_BL
        A = B =  sqrt(learning_rate / (dw_min * BL));

    In case of ``False``::

        if dw_min * desired_BL < learning_rate:
            A = B = 1;
            BL = ceil(learning_rate / dw_min;
        else:
            # same as for fixed_BL=True
    """

    pulse_type: PulseType = PulseType.STOCHASTIC_COMPRESSED
    """Switching between different pulse types. See :class:`PulseTypeMap` for details.

    Important:

       Pulsing can also be turned off in which case
       the update is done as if in floating point and all
       other update related parameter are ignored.
   """

    res: float = 0
    """Number of discretization steps of the probability in ``0..1``.
    Use -1 for turning discretization off. Can be :math:`1/n_\text{steps}` as well.
    """

    sto_round: bool = False
    """Whether to enable stochastic rounding."""

    update_bl_management: bool = True
    """Whether to enable dynamical adjustment of ``A``,``B``,and ``BL``::

        BL = ceil(learning_rate * abs(x_j) * abs(d_i) / dw_min);
        BL  = min(BL,desired_BL);
        A = B = sqrt(learning_rate / (dw_min * BL));
    """

    update_management: bool = True
    r"""After the above setting an additional scaling (always on when using
    `update_bl_management``) is applied to account for the different input strengths.
    If

    .. math:: \gamma \equiv \max_i |x_i| / \max_j |d_j|

    is the ratio between the two maximal inputs, then ``A`` is
    additionally scaled by :math:`\gamma` and ``B`` is scaled by
    :math:`1/\gamma`.
    """


# Basic parameters.

@dataclass
class FloatingPointTileParameters:
    """Parameters that modify the behaviour of a simple device."""

    bindings_class: ClassVar[Type] = devices.FloatingPointTileParameter

    diffusion: float = 0.0
    """Standard deviation of diffusion process."""

    lifetime: float = 0.0
    r"""One over `decay_rate`, ie :math:`1/r_\text{decay}`."""


@dataclass()
class AnalogTileParameters:
    """Parameters that modify the behavior of the analog tile."""

    bindings_class: ClassVar[Type] = devices.AnalogTileParameter

    forward_io: AnalogTileInputOutputParameters = field(
        default_factory=AnalogTileInputOutputParameters)
    """Input-output parameter setting for the forward direction."""

    backward_io: AnalogTileInputOutputParameters = field(
        default_factory=AnalogTileInputOutputParameters)
    """Input-output parameter setting for the backward direction."""

    update: AnalogTileUpdateParameters = field(
        default_factory=AnalogTileUpdateParameters)
    """Parameter for the update behavior."""


# Device parameters.

@dataclass
class PulsedResistiveDeviceParameters(FloatingPointTileParameters):
    """Parameters that modify the behaviour of a pulsed device."""

    bindings_class: ClassVar[Type] = devices.PulsedResistiveDeviceParameter

    corrupt_devices_prob: float = 0.0
    """Probability for devices to be corrupt (weights fixed to random value
    with hard bounds, that is min and max bounds are set to equal)."""

    corrupt_devices_range: int = 1000
    """Range around zero for establishing corrupt devices."""

    diffusion_dtod: float = 0.0
    """Device-to device variation of diffusion rate in relative units."""

    dw_min: float = 0.001
    """Mean of the minimal update step sizes across devices and directions."""

    dw_min_dtod: float = 0.3
    """Device-to-device std deviation of dw_min (in relative units to ``dw_min``)."""

    dw_min_std: float = 0.3
    r"""Cycle-to-cycle variation size of the update step (related to
    :math:`\sigma_\text{c-to-c}` above) in relative units to ``dw_min``.

    Note:
        Many spread (device-to-device variation) parameters are
        given in relative units. For instance e.g. a setting of
        ``dw_min_std`` of 0.1 would mean 10% spread around the
        mean and thus a resulting standard deviation
        (:math:`\sigma_\text{c-to-c}`) of ``dw_min`` *
        ``dw_min_std``.
    """

    enforce_consistency: bool = True
    """Whether to enforce during initialization that max weight bounds cannot
    be smaller than min weight bounds, and up direction step size is positive
    and down negative. Switches the opposite values if encountered during
    init."""

    lifetime_dtod: float = 0.0
    """Device-to-device variation in the decay rate (in relative units)."""

    perfect_bias: bool = False
    """No up-down differences and device-to-device variability in the bounds
    for the devices in the bias row."""

    reset: float = 0.01
    """The reset values and spread per cross-point ``ij`` when using reset functionality
    of the device."""

    reset_dtod: float = 0.0
    """See ``reset``."""

    reset_std: float = 0.01
    """See ``reset``."""

    up_down: float = 0.0
    r"""Up and down direction step sizes can be systematically different and also
    vary across devices.
    :math:`\Delta w_{ij}^d` is set during RPU initialization (for each cross-point `ij`):

    .. math::

        \Delta w_{ij}^d = d\; \Delta w_\text{min}\, \left(
        1 + d \beta_{ij} + \sigma_\text{d-to-d}\xi\right)

    where \xi is again a standard Gaussian. :math:`\beta_{ij}`
    is the directional up `versus` down bias.  At initialization
    ``up_down_dtod`` and ``up_down`` defines this bias term:

    .. math::

        \beta_{ij} = \beta_\text{up-down} + \xi
        \sigma_\text{up-down-dtod}

    where \xi is again a standard Gaussian number and
    :math:`\beta_\text{up-down}` corresponds to
    ``up_down``. Note that ``up_down_dtod`` is again given in
    relative units to ``dw_min``.
    """

    up_down_dtod: float = 0.01
    """See ``up_down``."""

    w_max: float = 0.6
    """See ``w_min``."""

    w_max_dtod: float = 0.3
    """See ``w_min_dtod``."""

    w_min: float = -0.6
    """Mean of hard bounds across device cross-point `ij`. The parameters
    ``w_min`` and ``w_max`` are used to set the min/max bounds independently.

    Note:
        For this abstract device, we assume that weights can have
        positive and negative values and are symmetrically around
        zero. In physical circuit terms, this might be implemented
        as a difference of two resistive elements.
    """

    w_min_dtod: float = 0.3
    """Device-to-device variation of the hard bounds, of min and max value,
    respectively. All are given in relative units to ``w_min``, or ``w_max``,
    respectively."""


@dataclass
class ConstantStepResistiveDeviceParameters(PulsedResistiveDeviceParameters):
    """Parameters that modify the behaviour of a ConstantStep device."""

    bindings_class: ClassVar[Type] = devices.ConstantStepResistiveDeviceParameter

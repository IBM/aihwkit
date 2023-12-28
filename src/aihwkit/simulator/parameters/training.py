# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=too-many-instance-attributes

"""Forward / backward / update related parameters for resistive processing units."""

from dataclasses import dataclass
from typing import ClassVar, Type, Optional, Union

from .helpers import _PrintableMixin
from .enums import (
    BoundManagementType,
    NoiseManagementType,
    WeightNoiseType,
    PulseType,
    AnalogMVType,
)


@dataclass
class IOParameters(_PrintableMixin):
    """Parameter that define the analog-matvec (forward / backward) and
    peripheral digital input-output behavior.

    Here one can enable analog-digital conversion, dynamic input
    scaling, and define the properties of the analog-matvec
    computations, such as noise and non-idealities (e.g. IR-drop).
    """

    bindings_class: ClassVar[Optional[Union[str, Type]]] = "AnalogTileInputOutputParameter"
    bindings_module: ClassVar[str] = "devices"

    is_perfect: bool = False
    """Short-cut to compute a perfect forward pass.

    If ``True``, it assumes an ideal forward pass (e.g. no bound, ADC etc...).
    Will disregard all other settings in this case.
    """

    mv_type: AnalogMVType = AnalogMVType.ONE_PASS
    """Selects the type of analog mat-vec computation. See
    :class:`AnalogMVType` for details. """

    inp_bound: float = 1.0
    """Input bound and ranges for the digital-to-analog converter (DAC)."""

    inp_noise: float = 0.0
    r"""Std deviation of Gaussian input noise (:math:`\sigma_\text{inp}`).

    i.e. noisiness of the analog input (at the stage after DAC and
    before the multiplication).
    """

    inp_res: float = 1 / (2**7 - 2)
    r"""Number of discretization steps for DAC (:math:`\le0` means infinite steps)
    or resolution (1/steps)."""

    inp_sto_round: bool = False
    """Whether to enable stochastic rounding of DAC."""

    inp_asymmetry: float = 0.0
    """Input asymmetry :math:`a_\text{input}`.

    Input of the negative input pass is scaled by :math:`(1 - a_\text{input})`.

    Note:
        This setting has only effect in case of and
        :class:`AnalogMVType` that uses separate passes for positive
        and negative inputs.
    """

    out_bound: float = 12.0
    """Output bound and ranges for analog-to-digital converter (ADC)."""

    out_noise: float = 0.06
    r"""Output noise strength at each output of a tile.

    This sets the std-deviation of the Gaussian output noise
    (:math:`\sigma_\text{out}`) at each output, i.e. noisiness of
    device summation at the output.
    """

    out_noise_std: float = 0.0
    r"""Systematic output-to-output variation of the output noise strength.

    In fraction of the ``out_noise`` parameter, that is 0.3 would mean a
    30\% variation of the output noise std deviation given by ``out_noise``.

    Note:

        This variation is drawn at instantiation and kept fixed
        thereafter. It can be adjusted, however, with
        ``analog_tile.set_forward/backward_parameter({'out_noise_values':
        x})`` for each analog tiles (if implemented).

    Caution:

      This is *not* simply the output noise std.-dev, but the
      systematic variation of the noise strength across outputs. Use
      ``out_noise`` to set the former.
    """

    out_res: float = 1 / (2**9 - 2)
    """Number of discretization steps for ADC or resolution.

    Number of discretization steps for ADC (:math:`<=0` means infinite steps)
    or resolution (1/steps).
    """

    out_sto_round: bool = False
    """Whether to enable stochastic rounding of ADC."""

    out_scale: float = 1.0
    """Additional fixed scalar factor."""

    out_asymmetry: float = 0.0
    """Output asymmetry :math:`a_\text{output}`.

    Output of the negative input pass is scaled by :math:`(1 - a_\text{output})`.

    Note:
        This setting has only effect in case of and
        :class:`AnalogMVType` that uses separate passes for positive
        and negative inputs.
    """

    bound_management: BoundManagementType = BoundManagementType.ITERATIVE

    """Type of bound management, see :class:`BoundManagementType`.

    Caution:
        Bound management is **only** available for the forward pass. It
        will be ignored when used for the backward pass.
    """

    noise_management: NoiseManagementType = NoiseManagementType.ABS_MAX
    """Type of noise management, see :class:`NoiseManagementType`."""

    w_noise: float = 0.0
    r"""Scale of output referred weight noise (:math:`\sigma_w`) for a given
    ``w_noise_type``."""

    w_noise_type: WeightNoiseType = WeightNoiseType.NONE
    """Type as specified in :class:`OutputWeightNoiseType`.

    Note:
        This noise us applied each time anew as it is referred to
        the output. It will not change the conductance values of
        the weight matrix. For the latter one can apply
        :meth:`diffuse_weights`.
    """

    ir_drop: float = 0.0
    """Scale of IR drop along the inputs (rows of the weight matrix).

    The IR-drop is calculated assuming that the first input is
    farthest away from the output channel. The expected drop is
    approximating the steady-state voltage distributions and depends
    on the input current.
    """

    ir_drop_g_ratio: float = 1.0 / 0.35 / 5e-6
    """Physical ratio of wire conductance from one cell to the next to
    physical max conductance of a device.

    Default is compute with 5mS maximal conductance set state and 0.35
    Ohm wire resistance.
    """

    out_nonlinearity: float = 0.0
    """S-shaped non-linearity applied to the analog output.

    Output non-linearity applies an S-shaped non-linearity to the
    analog output (before the ADC), i.e. :math:`\frac{y_i}{1 +
    n_i*|y_i|}` where :math:`n_i` is drawn at the instantiation time
    by::

        out_nonlinearity / out_bound * (1 + out_nonlinearity_std * rand)
    """

    out_nonlinearity_std: float = 0.0
    """ Output-to-output non linearity variation. """

    slope_calibration: float = 0.0
    """Models a calibration process of the output non-linearity (and
    r-series).

    This is the relative value in the output range where the slope of
    the non-linearity should have slope 1. E.g. 0.5 would be at half-out
    range.
    """

    v_offset_std: float = 0.0
    """Voltage offset variation.

    The output is multiplied by a systematic factor set for each
    output line at time of instantiation, e.g. :math:`(1 - v_i)` for
    the coding device and :math:`(1 + v_i)` for the reference device
    (assuming differential reads).

    """

    v_offset_w_min: float = -1.0
    """ Voltage offset for an implicit reference unit. """

    r_series: float = 0.0
    """Series resistance in fraction of the total output current."""

    w_read_asymmetry_dtod: float = 0.0
    """Device polarity read dependence.

    The negative inputs perceive a slightly different weight (e.g. pcm
    polarity dependence). Each device has a different factor, and the
    spread of this device-to-device variability can be set with
    ``w_read_asymmetry_dtod``. A weight (given negative input) will be
    then scaled by :math:`1 - f_{ij}` where :math:`f_{ij}` is drawn
    from a Gaussian distribution (with zero mean and standard
    deviation ``w_read_asymmetry_dtod``).
    """

    max_bm_factor: int = 1000
    """Maximal bound management factor.

    If this factor is reached then the iterative process is stopped.
    """

    max_bm_res: float = 0.25
    """Limit the maximal number of iterations of the bound management.

    Another way to limit the maximal number of iterations of the bound
    management. The max effective resolution number of the inputs, e.g. use
    :math:`1/4` for 2 bits.
    """

    bm_test_negative_bound: bool = True

    nm_thres: float = 0.0
    r"""Constant noise management value for ``type`` ``Constant``.

    In other cases, this is a upper threshold :math:`\theta` above which the
    noise management factor is saturated. E.g. for `AbsMax`:

    .. math::
        :nowrap:

        \begin{equation*} \alpha=\begin{cases}\max_i|x_i|, &
        \text{if} \max_i|x_i|<\theta \\ \theta, &
        \text{otherwise}\end{cases} \end{equation*}

    Caution:
        If ``nm_thres`` is set (and type is not ``Constant``), the noise
        management will clip some large input values, in favor of having a
        better SNR for smaller input values.
    """


@dataclass
class UpdateParameters(_PrintableMixin):
    """Parameter that modify the update behaviour of a pulsed device."""

    bindings_class: ClassVar[Optional[Union[str, Type]]] = "AnalogTileUpdateParameter"
    bindings_module: ClassVar[str] = "devices"

    desired_bl: int = 31
    """Desired length of the pulse trains.

    For update BL management, it is the maximal pulse train length.
    """

    fixed_bl: bool = True
    """Whether to fix the length of the pulse trains.

    See also ``update_bl_management``.

    In case of ``True`` (where ``dw_min`` is the mean minimal weight change
    step size) it is::

        BL = desired_BL
        A = B =  sqrt(learning_rate / (dw_min * BL))

    In case of ``False``::

        if dw_min * desired_BL < learning_rate:
            A = B = 1
            BL = ceil(learning_rate / dw_min
        else:
            # same as for fixed_BL=True
    """

    pulse_type: PulseType = PulseType.STOCHASTIC_COMPRESSED
    """Switching between different pulse types.

    See also :class:`PulseTypeMap` for details.

    Important:
        Pulsing can also be turned off in which case the update is done as if
        in floating point and all other update related parameter are ignored.
    """

    res: float = 0
    """Resolution of the update probability for the stochastic bit line
    generation.

    Resolution ie. bin width in ``0..1``) of the update probability for the
    stochastic bit line generation. Use -1 for turning discretization off. Can
    be given as number of steps as well.
    """

    x_res_implicit: float = 0
    """Resolution of each quantization step for the inputs ``x``.

    Resolution (ie. bin width) of each quantization step for the inputs ``x``
    in case of ``DeterministicImplicit`` pulse trains. See
    :class:`PulseTypeMap` for details.
    """

    d_res_implicit: float = 0
    """Resolution of each quantization step for the error ``d``.

    Resolution (ie. bin width) of each quantization step for the error ``d``
    in case of `DeterministicImplicit` pulse trains. See
    :class:`PulseTypeMap` for details.
    """

    d_sparsity: bool = False
    """Whether to compute gradient sparsity.
    """

    sto_round: bool = False
    """Whether to enable stochastic rounding."""

    update_bl_management: bool = True
    """Whether to enable dynamical adjustment of ``A``,``B``,and ``BL``::

        BL = ceil(learning_rate * abs(x_j) * abs(d_i) / weight_granularity);
        BL  = min(BL,desired_BL);
        A = B = sqrt(learning_rate / (weight_granularity * BL));

    The ``weight_granularity`` is usually equal to ``dw_min``.
    """

    update_management: bool = True
    r"""Whether to apply additional scaling.

    After the above setting an additional scaling (always on when using
    `update_bl_management``) is applied to account for the different input
    strengths.
    If

    .. math:: \gamma \equiv \max_i |x_i| / (\alpha \max_j |d_j|)

    is the ratio between the two maximal inputs, then ``A`` is additionally
    scaled by :math:`\gamma` and ``B`` is scaled by :math:`1/\gamma`.

    The gradient scale :math:`\alpha` can be set with ``um_grad_scale``
    """

    um_grad_scale: float = 1.0
    r"""Scales the gradient for the update management.

    The factor :math:`\alpha` for the ``update_management``. If
    smaller than 1 it means that the gradient will be earlier clipped
    when learning rate is too large (ie. exceeding the maximal
    pulse number times the weight granularity). If 1, both d and x inputs
    are clipped for the same learning rate.
    """

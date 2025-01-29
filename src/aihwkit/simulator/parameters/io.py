# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=too-many-instance-attributes

"""Forward / backward / update related parameters for resistive processing units."""

from dataclasses import dataclass
from typing import ClassVar, Type, Optional, Union, List

from .helpers import _PrintableMixin
from .enums import BoundManagementType, NoiseManagementType, WeightNoiseType, AnalogMVType


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
class IOParametersIRDropT(IOParameters):
    """Parameter that define the analog-matvec (forward / backward) and
    peripheral digital input-output behavior.

    Here one can enable analog-digital conversion, dynamic input
    scaling, and define the properties of the analog-matvec
    computations, such as noise and non-idealities (e.g. IR-drop).
    """

    bindings_ignore: ClassVar[List] = [
        "ir_drop_rs",
        "ir_drop_segments",
        "ir_drop_time_step_resolution_scale",
        "ir_drop_v_read",
        "split_mode_bit_shift",
    ]

    ir_drop: float = 1.0
    """Scales the physical wire resistance. To turn off the ir_drop,
    set `ir_drop=0.0`.

    Note:

        IR-drop is only enabled during testing. It is automatically
        turned off during HWA training.
    """

    ir_drop_rs: float = 0.15
    """Physical wire resistance [Ohms] from one unit cell to the next
    unit cell.
    """

    ir_drop_segments: int = 8
    """Number of segments into which to break column simulation when
    appromixating IR-drop. Ideal calculation uses max_input_size
    meaning that the number of segments will be equivalent to the
    number of unit cells.  This is more computationally expensive,
    however. Approximating IR-drop using eight segments is usually
    suffiecient and an appropriate trade-off between speed and
    IR-drop accuracy.
    """

    ir_drop_time_step_resolution_scale: float = 1.0
    """Time slice resolution to be used when calculating the
    transient current responses for IR-drop. Default PWM/DAC
    uses 8-bits (with one sign bit), so 128 time slices is
    adequate. These time slices are automatically determined,
    during conventional mode operation and adjusted (reduced)
    during split PWM and bit-wise PWM operation. The inp_res
    parameter to determine the appropriate number of IR-drop
    time slices as inp_res determines the max number of discrete
    time steps used for describing input activations. Reducing
    ir_drop_time_resolution_scale to 0.5, for instance, will
    reduce the number of time slices computed (and reduce
    accuracy), but increase the speed of IR drop calculations.
    Values above 1.0 will increase resolution beyond the PWM/DAC
    change rate and therefore offer no benefit (just increased
    memory usage and increased computational time). Bit slicing
    mode will ignore this parameter as it will lead to faulty
    IR drop calculations if the resolution is below the minimum
    resolution of the PWM/DAC.
    """

    ir_drop_v_read: float = 0.4005
    """ADC read voltage used in calculating IR-drop current
    in conjunction with Thevenin equivalent circuit approximation
    of MAC tile.
    """

    split_mode_bit_shift = 3
    """Number of bits in in LSB (remainder) that is fed to MVM
    tile in first phase of split mode PWM/DAC operation. Split mode
    PWM/DAC operation increases throughput / energy efficiency
    of MVM tile hardware while potentially sacrificing some
    analog MVM accuracy."""

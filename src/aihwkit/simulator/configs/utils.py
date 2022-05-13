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

# pylint: disable=too-many-instance-attributes

"""Utility parameters for resistive processing units."""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Type

from aihwkit.simulator.configs.helpers import _PrintableMixin
from aihwkit.simulator.rpu_base import devices, tiles


# Helper enums.

class BoundManagementType(Enum):
    """Bound management type.

    In the case ``Iterative`` the MAC is iteratively recomputed with
    inputs iteratively halved, when the output bound was hit.

    Caution:
        Bound management is **only** available for the forward pass. It
        will be ignored when used for the backward pass.
    """

    NONE = 'None'
    """No bound management."""

    ITERATIVE = 'Iterative'
    r"""Iteratively recomputes input scale set to :math:`\alpha\leftarrow\alpha/2`.

    It iteratively recomputes the bounds up to limit of passes (given by
    ``max_bm_factor`` or ``max_bm_res``).
    """

    ITERATIVE_WORST_CASE = 'IterativeWorstCase'
    """Worst case bound management.

    Uses ``AbsMax`` noise management for the first pass and only when output
    bound is hit, the ``AbsMaxNPSum`` for the second. Thus, at most 2 passes
    are computed.
    """

    SHIFT = 'Shift'
    """Shift bound management.

    Shifts the output by adding the difference ``output_bound - max_output`` to
    the analog output value. This is only useful to increase the dynamic range
    before the softmax, where the max can be safely.

    Note:
        Shifting needs hardware implementations.
    """


class NoiseManagementType(Enum):
    r"""Noise management type.

    Noise management determines a factor :math:`\alpha` how the input is reduced:

    .. math:: \mathbf{y} = \alpha\;F_\text{analog-mac}\left(\mathbf{x}/\alpha\right)
    """

    NONE = 'None'
    """No noise management."""

    ABS_MAX = 'AbsMax'
    r"""Use :math:`\alpha\equiv\max{|\mathbf{x}|}`."""

    ABS_MAX_NP_SUM = 'AbsMaxNPSum'
    """Assume weight value is constant and given by ``nm_assumed_wmax``.

    Takes a worst case scenario of the weight matrix to calculate the input
    scale to ensure that output is not clipping. Assumed weight value is
    constant and given by ``nm_assumed_wmax``.
    """

    MAX = 'Max'
    r"""Use :math:`\alpha\equiv\max{\mathbf{x}}`."""

    CONSTANT = 'Constant'
    r"""A constant value (given by parameter ``nm_thres``)."""

    AVERAGE_ABS_MAX = 'AverageAbsMax'
    """Moment-based scale input scale estimation.

    Computes the average abs max over the mini-batch and applies ``nm_decay``
    to update the value with the history.

    Note:
        ``nm_decay`` is ``1-momentum`` and always given in mini-batches.
        However, the CUDA implementation does not discount values within
        mini-batches, whereas the CPU implementation does.
    """


class WeightNoiseType(Enum):
    r"""Output weight noise type.

    The weight noise is applied for each MAC computation, while not
    touching the actual weight matrix but referring it to the output.

    .. math:: y_i = \sum_j w_{ij}+\xi_{ij}
    """

    NONE = 'None'
    """No weight noise."""

    ADDITIVE_CONSTANT = 'AdditiveConstant'
    r"""The :math:`\xi\sim{\cal N}(0,\sigma)` thus all are Gaussian distributed.

    :math:`\sigma` is determined by ``w_noise``.
    """

    PCM_READ = 'PCMRead'
    """Output-referred PCM-like read noise.

    Output-referred PCM-like read noise that scales with the amount of current
    generated for each output line and thus scales with both conductance values
    and input strength.

    The same general for is taken as for PCM-like statistical model of the 1/f
    noise during inference, see
    :class:`aihwkit.inference.noise.pcm.PCMLikeNoiseModel`.
    """


class PulseType(Enum):
    """Pulse type."""

    NONE = 'None'
    """Floating point update instead of pulses."""

    STOCHASTIC_COMPRESSED = 'StochasticCompressed'
    """Generates actual stochastic bit lines.

    Plus and minus pulses are taken in the same pass.
    """

    STOCHASTIC = 'Stochastic'
    """Two passes for plus and minus (only CPU)."""

    NONE_WITH_DEVICE = 'NoneWithDevice'
    """Floating point like ``None``, but with analog devices (e.g. weight
    clipping)."""

    MEAN_COUNT = 'MeanCount'
    """Coincidence based in prob (:math:`p_a p_b`)."""

    DETERMINISTIC_IMPLICIT = 'DeterministicImplicit'
    r"""Coincidences are computed in deterministic manner.

    Coincidences are calculated by :math:`b_l x_q d_q` where ``BL`` is the
    desired bit length (possibly subject to dynamic adjustments using
    ``update_bl_management``) and :math:`x_q` and :math:`d_q` are the quantized
    input and error values, respectively, normalized to the range
    :math:`0,\ldots,1`. It can be shown that explicit bit lines exist that
    generate these coincidences.
    """


class WeightModifierType(Enum):
    """Weight modifier type."""

    COPY = 'Copy'
    """Just copy, however, could also drop."""

    DISCRETIZE = 'Discretize'
    """Quantize the weights."""

    MULT_NORMAL = 'MultNormal'
    """Multiplicative Gaussian noise."""

    ADD_NORMAL = 'AddNormal'
    """Additive Gaussian noise."""

    DISCRETIZE_ADD_NORMAL = 'DiscretizeAddNormal'
    """First discretize and then additive Gaussian noise."""

    DOREFA = 'DoReFa'
    """DoReFa discretization."""

    POLY = 'Poly'
    r"""2nd order Polynomial noise model (in terms of the weight value).

    In detail, for the duration of a mini-batch, each weight will be
    added a Gaussian random number with the standard deviation of
    :math:`\sigma_\text{wnoise} (c_0 + c_1 w_{ij}/\omega + c_2
    w_{ij}^2/\omega^2` where :math:`omega` is either the actual max
    weight (if ``rel_to_actual_wmax`` is set) or the value
    ``assumed_wmax``.
    """


class WeightClipType(Enum):
    """Weight clipper type."""

    NONE = 'None'
    """None."""

    FIXED_VALUE = 'FixedValue'
    """Clip to fixed value give, symmetrical around zero."""

    LAYER_GAUSSIAN = 'LayerGaussian'
    """Calculates the second moment of the whole weight matrix and clips
    at ``sigma`` times the result symmetrically around zero."""

    AVERAGE_CHANNEL_MAX = 'AverageChannelMax'
    """Calculates the abs max of each output channel (row of the weight
    matrix) and takes the average as clipping value for all."""


class VectorUnitCellUpdatePolicy(Enum):
    """Vector unit cell update policy."""

    ALL = 'All'
    """All devices updated simultaneously."""

    SINGLE_FIXED = 'SingleFixed'
    """Device index is not changed. Can be set initially and/or updated on
    the fly."""

    SINGLE_SEQUENTIAL = 'SingleSequential'
    """Each device one at a time in sequence."""

    SINGLE_RANDOM = 'SingleRandom'
    """A single device is selected by random choice each mini-batch."""


# Specialized parameters.

@dataclass
class IOParameters(_PrintableMixin):
    """Parameter that modify the IO behavior."""

    bindings_class: ClassVar[Type] = devices.AnalogTileInputOutputParameter

    bm_test_negative_bound: bool = True

    bound_management: BoundManagementType = BoundManagementType.ITERATIVE
    """Type of bound management, see :class:`BoundManagementType`.

    Caution:
        Bound management is **only** available for the forward pass. It
        will be ignored when used for the backward pass.
    """

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

    is_perfect: bool = False
    """Short-cut to compute a perfect forward pass.

    If ``True``, it assumes an ideal forward pass (e.g. no bound, ADC etc...).
    Will disregard all other settings in this case.
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

    noise_management: NoiseManagementType = NoiseManagementType.ABS_MAX
    """Type of noise management, see :class:`NoiseManagementType`."""

    out_bound: float = 12.0
    """Output bound and ranges for analog-to-digital converter (ADC)."""

    out_noise: float = 0.06
    r"""Std deviation of Gaussian output noise (:math:`\sigma_\text{out}`).

    i.e. noisiness of device summation at the output.
    """

    out_res: float = 1 / (2**9 - 2)
    """Number of discretization steps for ADC or resolution.

    Number of discretization steps for ADC (:math:`<=0` means infinite steps)
    or resolution (1/steps).
    """

    out_scale: float = 1.0
    """Additional fixed scalar factor."""

    out_sto_round: bool = False
    """Whether to enable stochastic rounding of ADC."""

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

    ir_drop_g_ratio: float = 1/(0.35*5e-6)
    """Physical ratio of wire conductance from one cell to the next to
    physical max conductance of a device.

    Default is compute with 5mS maximal conductance set state and 0.35
    Ohm wire resistance.
    """


@dataclass
class UpdateParameters(_PrintableMixin):

    """Parameter that modify the update behaviour of a pulsed device."""

    bindings_class: ClassVar[Type] = devices.AnalogTileUpdateParameter

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
        A = B =  sqrt(learning_rate / (dw_min * BL));

    In case of ``False``::

        if dw_min * desired_BL < learning_rate:
            A = B = 1;
            BL = ceil(learning_rate / dw_min;
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

    .. math:: \gamma \equiv \max_i |x_i| / \max_j |d_j|

    is the ratio between the two maximal inputs, then ``A`` is additionally
    scaled by :math:`\gamma` and ``B`` is scaled by :math:`1/\gamma`.
    """


@dataclass
class WeightModifierParameter(_PrintableMixin):
    """Parameter that modify the forward/backward weights during hardware-aware training."""

    bindings_class: ClassVar[Type] = tiles.WeightModifierParameter

    std_dev: float = 0.0
    """Standard deviation of the added noise to the weight matrix.

    This parameter affects the modifier types ``AddNormal``, ``MultNormal`` and
    ``DiscretizeAddNormal``.

    Note:
        If the parameter ``rel_to_actual_wmax`` is set then the ``std_dev`` is
        computed in relative terms to the abs max of the given weight matrix,
        otherwise it in relative terms to the assumed max, which is set by
        ``assumed_wmax``.
    """

    res: float = 0.0
    r"""Resolution of the discretization.

    The invert of ``res`` gives the number of equal sized steps in
    :math:`-a_\text{max}\ldots,a_\text{max}` where the
    :math:`a_\text{max}` is either given by the abs max (if
    ``rel_to_actual_wmax`` is set) or ``assumed_wmax`` otherwise.

    ``res`` is only used in the modifier types ``DoReFa``, ``Discretize``, and
    ``DiscretizeAddNormal``.
    """

    sto_round: bool = False
    """Whether the discretization is done with stochastic rounding enabled.

    ``sto_round`` is only used in the modifier types ``DoReFa``,
    ``Discretize``, and ``DiscretizeAddNormal``.
    """

    dorefa_clip: float = 0.6
    """Parameter for DoReFa."""

    pdrop: float = 0.0
    """Drop connect probability.

    Drop connect sets weights to zero with the given probability. This
    implements drop connect.

    Important:
        Drop connect can be used with any other modifier type in combination.
    """

    enable_during_test: bool = False
    """Whether to use the last modified weight matrix during testing.

    Caution:
        This will **not** remove drop connect or any other noise
        during evaluation, and thus should only used with care.
    """

    rel_to_actual_wmax: bool = True
    """Whether to calculate the abs max of the weight and apply noise relative
    to this number.

    If set to False, ``assumed_wmax`` is taken as relative units.
    """

    assumed_wmax: float = 1.0
    """Assumed weight value that is mapped to the maximal conductance.

    This is typically 1.0. This parameter will be ignored if
    ``rel_to_actual_wmax`` is set.
    """

    copy_last_column: bool = False
    """Whether to not apply noise to the last column (which usually contains
    the bias values)."""

    coeff0: float = 0.26348 / 25.0
    coeff1: float = 0.0768
    coeff2: float = -0.001877 * 25.0
    """Coefficients for the ``POLY`` weight modifier type.

    See :class:`WeightModifierType` for details.
    """

    type: WeightModifierType = WeightModifierType.COPY
    """Type of the weight modification."""


@dataclass
class WeightClipParameter(_PrintableMixin):
    """Parameter that clip the weights during hardware-aware training."""

    bindings_class: ClassVar[Type] = tiles.WeightClipParameter

    fixed_value: float = -1.0
    """Clipping value in case of ``FixedValue`` type.

    Caution:

        If ``fixed_value > 0`` it will be also applied during other
        clipping types.

    """

    sigma: float = 2.5
    """Sigma value for clipping for the ``LayerGaussian`` type."""

    type: WeightClipType = WeightClipType.NONE
    """Type of clipping."""


@dataclass
class SimpleDriftParameter(_PrintableMixin):
    r"""Parameter for a simple power law drift.

    The drift as a simple power law drift without device-to-device
    variation or conductance dependence.

    It computes:
    .. math::

        w_{ij}*\left(\frac{t + \Delta t}{t_0}\right)^(-\nu)
    """

    bindings_class: ClassVar[Type] = devices.DriftParameter

    nu: float = 0.0
    r"""Average drift :math:`\nu` value.

    Need to non-zero to actually use the drift.
    """

    t_0: float = 1.0
    """Time between write and first read.

    Usually assumed in milliseconds, however, it really determines the time
    units of ``time_since_last_call`` when calling the drift.
    """

    reset_tol: float = 1e-7
    """Reset tolerance.

    This should a number smaller than the expected weight change as it is used
    to detect any changes in the weight from the last drift call. Every change
    to the weight above this tolerance will reset the drift time.

    Caution:
        Any write noise or diffusion on the weight might thus
        interfere with the drift.
   """


@dataclass
class DriftParameter(SimpleDriftParameter):
    r"""Parameter for a power law drift.

    The drift is based on the model described by `Oh et al (2019)`_.

    It computes:
    .. math::

        w_{ij}*\left(\frac{t + \Delta t}{t_0}\right)^(-\nu^\text{actual}_{ij})

    where the drift coefficient is drawn once at the beginning and
    might depend on device. It also can depend on the actual weight
    value.

    The actual drift coefficient is computed as:
    .. math::

        \nu_{ij}^\text{actual} =  \nu_{ij} - \nu_k \log \frac{(w_{ij} - w_\text{off}) / r_\text{wg}
        + g_\text{off}}{G_0}  + \nu\sigma_\nu\xi

    here :math:`w_{ij}` is the actual weight and `\nu_{ij}` fixed for
    each device given by the mean :math:`\nu` and the device-to-device
    variation: :math:`\nu_{ij} = \nu + \nu_dtod\nu\xi` and are only
    drawn once at the beginning (tile instantiation).  `\xi` is
    Gaussian noise.

    Note:
        If the weight has changed from the last drift call (determined
        by the ``reset_tol`` parameter), for instance due to update,
        decay or noise, then the drift time :math:`t` will be reset and start
        from new, however, the drift coefficients :math:`\nu_{ij}` are
        *not* changed. On the other hand, if the weights has not
        changed since last call, :math:`t` will accumulate the time.

    Caution:
        Note that the drift coefficient does *not* depend on the initially
        programmed weight value at :math:`t=0` in the current
        implementation (ie G0 is a constant for all devices), but
        instead on the actual weight. In some materials (e.g. phase
        changed materials), that might be not accurate.

    .. _`Oh et al (2019)`: https://ieeexplore.ieee.org/document/8753712
    """

    bindings_class: ClassVar[Type] = devices.DriftParameter

    nu_dtod: float = 0.0
    r"""Device-to-device variation of the :math:`\nu` values."""

    nu_std: float = 0.0
    r"""Cycle-to-cycle variation of :math:`\nu`.

    A more realistic way to add noise of the drift might be using
    ``w_noise_std``.
    """

    wg_ratio: float = 1.0
    """``(w_max-w_min)/(g_max-g_min)`` to convert to physical units."""

    g_offset: float = 0.0
    """``g_min`` to convert to physical units."""

    w_offset: float = 0.0
    """``w(g_min)``, i.e. to what value ``g_min`` is mapped to in w-space."""

    nu_k: float = 0.0
    r"""Variation of math:`nu` with :math:`W`.

    That is :math:`\nu(R) = nu_0 - k \log(G/G_0)`.  See Oh et al. for
    details.
    """

    log_g0: float = 0.0
    """Log g0."""

    w_noise_std: float = 0.0
    """Additional weight noise (Gaussian diffusion) added to the weights
    after the drift is applied."""


@dataclass
class MappingParameter(_PrintableMixin):
    """Parameter related to hardware design and the mapping of logical
    weight matrices to physical tiles.

    Caution:

        Some of these parameters have only an affect for modules that
        support tile mappings.

    """

    digital_bias: bool = True
    """Whether the bias term is handled by the analog tile or kept in
    digital.

    Note:
        Default is having a *digital* bias so that bias values are
        *not* stored onto the analog crossbar. This needs to be
        supported by the chip design. Set to False if the analog bias
        is instead situated on the the crossbar itself (as an extra
        column)

    Note:
        ``digital_bias`` is supported by *all* analog modules.
    """

    weight_scaling_omega: float = 0.0
    """omega_scale is a user defined parameter used to scale the weights
    while remapping these to cover the full range of values allowed"""

    weight_scaling_omega_columnwise: bool = False
    """Whether the weight matrix will be remapped column-wise over
    the maximum device allowed value"""

    learn_out_scaling_alpha: bool = False
    """define the out_scaling_alpha as a learnable parameter
    used to scale the output"""

    max_input_size: int = 512
    """Maximal input size (number of columns) of the weight matrix
    that is handled on a single analog tile.

    If the logical weight matrix size exceeds this size it will be
    split and mapped onto multiple analog tiles.

    Caution:
        Only relevant for ``Mapped`` modules such as
        :class:`aihwkit.nn.modules.linear_mapped.AnalogLinearMapped`.
    """

    max_output_size: int = 512
    """Maximal output size (number of rows) of the weight matrix
    that is handled on a single analog tile.

    If the logical weight matrix size exceeds this size it will be
    split and mapped onto multiple analog tiles.

    Caution:
        Only relevant for ``Mapped`` modules such as
        :class:`aihwkit.nn.modules.linear_mapped.AnalogLinearMapped`.
    """

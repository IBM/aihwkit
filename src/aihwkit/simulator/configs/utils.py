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

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-lines

"""Utility parameters for resistive processing units."""

from dataclasses import dataclass, field
from typing import ClassVar, Type, Union, Any, List, TYPE_CHECKING

from aihwkit.simulator.configs.helpers import _PrintableMixin
from aihwkit.simulator.rpu_base import devices, tiles
from aihwkit.simulator.configs.enums import (
    BoundManagementType, NoiseManagementType, WeightNoiseType, PulseType,
    WeightModifierType, WeightClipType, WeightRemapType,
    AnalogMVType
)

if TYPE_CHECKING:
    from aihwkit.nn.modules.linear import AnalogLinear
    from aihwkit.nn.modules.linear_mapped import AnalogLinearMapped


@dataclass
class IOParameters(_PrintableMixin):
    """Parameter that define the analog-matvec (forward / backward) and
    peripheral digital input-output behavior.

    Here one can enable analog-digital conversion, dynamic input
    scaling, and define the properties of the analog-matvec
    computations, such as noise and non-idealities (e.g. IR-drop).
    """

    bindings_class: ClassVar[Type] = devices.AnalogTileInputOutputParameter

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
    r"""Std deviation of Gaussian output noise (:math:`\sigma_\text{out}`).

    i.e. noisiness of device summation at the output.
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

    coeffs: List[float] = field(
        default_factory=lambda: [0.0105392, 0.0768, -0.046925],
        metadata={'hide_if': [0.0105392, 0.0768, -0.046925]}
    )

    """Coefficients for the ``POLY`` weight modifier type.

    See :class:`WeightModifierType` for details.
    """

    type: WeightModifierType = WeightModifierType.COPY
    """Type of the weight modification."""


@dataclass
class WeightClipParameter(_PrintableMixin):
    """Parameter that clip the weights during hardware-aware training.

    Important:
        A clipping ``type`` has to be set before any of the parameter
        changes take any effect.

    """

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
class WeightRemapParameter(_PrintableMixin):
    """Parameter that remap the weights during hardware-aware training.

    Important:
        A remap ``type`` has to be set before any of the parameter
        changes take any effect.
    """
    bindings_class: ClassVar[Type] = tiles.WeightRemapParameter

    remapped_wmax: float = 1.0
    """Assumed max of weight, ie the value of the weight the maximal
    conductance is mapped to. Typically 1.0.
    """

    max_scale_range: float = 0.0
    """Maximal range of scale values. Use zero to turn any restrictions
    off (default)."""

    max_scale_ref: float = 0.0
    """Reference scale that use used as minimal scale for determining the
    scale range."""

    type: WeightRemapType = WeightRemapType.NONE
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

        Some of these parameters have only an effect for modules that
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
    while remapping these to cover the full range of values allowed.
    """

    weight_scaling_columnwise: bool = False
    """Whether the weight matrix will be remapped column-wise over
    the maximum device allowed value."""

    learn_out_scaling: bool = False
    """Define (additional) out scales that are learnable parameter
    used to scale the output."""

    out_scaling_columnwise: bool = False
    """Whether the learnable out scaling parameter enabled by
    ``learn_out_scaling`` is a scalar (``False``) or learned for
    each output (``True``).
    """

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


@dataclass
class InputRangeParameter(_PrintableMixin):
    """ Parameter related to input range learning """

    enable: bool = False
    """Whether to enable to learn the input range. Note that if enable is
    ``False`` then no clip is applied.

    Note:

        The input bound (``forward.inp_bound``) is assumed to be 1 if
        enabled as the input range already scales the input into to the
        range :math:`(-1, 1)` by dividing the input to the type by
        itself and multiplying the output accordingly.

        Typically, noise and bound management should be set to `NONE`
        for the input range learning as it replaces the dynamic
        managements with a static but learned input bound. However, in
        some exceptional experimental cases one might want to enable
        the management techniques on top of the input range learning,
        so that no error is raised if they are not set to `NONE`.
    """

    init_value: float = 3.0
    """Initial setting of the input range in case of input range learning."""

    init_from_data: int = 100
    """Number of batches to use for initialization from data. Set 0 to turn off."""

    init_std_alpha: float = 3.0
    """Standard deviation multiplier for initialization from data."""

    decay: float = 0.001
    """Decay rate for input range learning."""

    input_min_percentage: float = 0.95
    """Decay is only applied if percentage of non-clipped values is above this value.

    Note:

        The added gradient is (in case of non-clipped input
        percentage ``percentage > input_min_percentage``)::

            grad += decay * input_range
    """

    manage_output_clipping: bool = True
    """Whether to increase the input range when output clipping occurs.

    Caution:

        The output bound is taken from the ``forward.out_bound``
        value, which has to exist. Noise and bound management have to
        be set to NONE if this feature is enabled otherwise a
        ``ConfigError`` is raised.

    """

    output_min_percentage: float = 0.95
    """Increase of the input range is only applied if percentage of
    non-clipped output values is below this value.

    Note:

        The gradient subtracted from the input range is (in case of
        ``output_percentage < output_min_percentage``)::

            grad -= (1.0 - output_percentage) * input_range
    """

    gradient_scale: float = 1.0
    """Scale of the gradient magnitude (learning rate) for the input range learning."""

    gradient_relative: bool = True
    """Whether to make the gradient of the input range learning relative to
    the current range value.
    """

    def supports_manage_output_clipping(self, rpu_config: Any) -> bool:
        """ Checks whether rpu_config supported ``manage_output_clipping``.

        Args:
            rpu_config: RPUConfig to check

        Returns:
            True if supported otherwise False
        """

        if not hasattr(rpu_config, 'forward') or rpu_config.forward.is_perfect:
            return False
        if not isinstance(rpu_config.forward, IOParameters):
            return False
        if rpu_config.forward.noise_management != NoiseManagementType.NONE:
            return False
        if rpu_config.forward.bound_management != BoundManagementType.NONE:
            return False
        return True


@dataclass
class PrePostProcessingParameter(_PrintableMixin):
    """Parameter related to digital input and output processing, such as input clip
    learning.
     """
    input_range: InputRangeParameter = field(default_factory=InputRangeParameter)


@dataclass
class MapableRPU(_PrintableMixin):
    """Defines the mapping parameters and utility factories"""

    mapping: MappingParameter = field(default_factory=MappingParameter)
    """Parameter related to mapping weights to tiles for supporting modules."""

    def get_linear(self) -> Union[Type['AnalogLinear'], Type['AnalogLinearMapped']]:
        """Returns a AnalogLinear module as specified """
        # pylint: disable=import-outside-toplevel
        # need to import here to avoid circular imports
        from aihwkit.nn.modules.linear import AnalogLinear
        from aihwkit.nn.modules.linear_mapped import AnalogLinearMapped

        if self.mapping.max_input_size > 0 or self.mapping.max_output_size > 0:
            return AnalogLinearMapped
        return AnalogLinear


@dataclass
class PrePostProcessingRPU(_PrintableMixin):
    """Defines the pre-post parameters and utility factories"""

    pre_post: PrePostProcessingParameter = field(default_factory=PrePostProcessingParameter)
    """Parameter related digital pre and post processing."""

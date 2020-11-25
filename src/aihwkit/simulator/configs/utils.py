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


class WeightNoiseType(Enum):
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


class WeightModifierType(Enum):
    """Weight modifier type."""

    COPY = 'Copy'
    """Just copy, however, could also drop."""

    DISCRETIZE = 'Discretize'
    """Quantize the weights."""

    MULT_NORMAL = 'MultNormal'
    """Mutiplicative Gaussian noise."""

    ADD_NORMAL = 'AddNormal'
    """Additive Gaussian noise."""

    DISCRETIZE_ADD_NORMAL = 'DiscretizeAddNormal'
    """First discretize and then additive Gaussian noise."""

    DOREFA = 'DoReFa'
    """DoReFa discretization."""


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
    """Parameters that modify the IO behavior."""

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

    w_noise_type: WeightNoiseType = WeightNoiseType.NONE
    """Type as specified in :class:`OutputWeightNoiseType`.

    Note:
        This noise us applied each time anew as it is referred to
        the output. It will not change the conductance values of
        the weight matrix. For the latter one can apply
        :meth:`diffuse_weights`.
    """


@dataclass
class UpdateParameters(_PrintableMixin):
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


@dataclass
class WeightModifierParameter(_PrintableMixin):
    """Parameter that modify the forward/backward weights during hardware-aware training."""

    bindings_class: ClassVar[Type] = tiles.WeightModifierParameter

    std_dev: float = 0.0
    """Standard deviation of the added noise to the weight matrix.

    This parameter affects the modifier types ``AddNormal``,
    ``MultNormal`` and ``DiscretizeAddNormal``.

    Note:
        If the parameter ``rel_to_actual_wmax`` is set then the
        ``std_dev`` is computed in relative terms to the abs max of the
        given weight matrix, otherwise it in relative terms to the
        assumed max, which is set by ``assumed_wmax``.
    """

    res: float = 0.0
    r"""Resolution of the discretization.

    The invert of ``res`` gives the number of equal sized steps in
    :math:`-a_\text{max}\ldots,a_\text{max}` where the
    :math:`a_\text{max}` is either given by the abs max (if
    ``rel_to_actual_wmax`` is set) or ``assumed_wmax`` otherwise.

    ``res`` is only used in the modifier types ``DoReFa``,
    ``Discretize``, and ``DiscretizeAddNormal``.
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

    Drop connect sets weights to zero with the given probability. This implements drop connect.

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
    """Whether to calculate the abs max of the weight and apply noise relative to this number.

    If set to False, ``assumed_wmax`` is taken as relative units.
    """

    assumed_wmax: float = 1.0
    """Assumed weight value that is mapped to the maximal conductance.

    This is typically 1.0. This parameter will be ignored if
    ``rel_to_actual_wmax`` is set.
    """

    type: WeightModifierType = WeightModifierType.COPY
    """Type of the weight modification."""


@dataclass
class WeightClipParameter(_PrintableMixin):
    """Parameter that clip the weights during hardware-aware training."""

    bindings_class: ClassVar[Type] = tiles.WeightClipParameter

    fixed_value: float = 1.0
    """Clipping value in case of ``FixedValue`` type."""

    sigma: float = 2.5
    """Sigma value for clipping for the ``LayerGaussian`` type."""

    type: WeightClipType = WeightClipType.NONE
    """Type of clipping."""

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

"""Utility parameters for resistive processing units."""

# pylint: disable=too-many-instance-attributes

from dataclasses import dataclass, is_dataclass
from enum import Enum
from typing import Any, ClassVar, Type

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
class IOParameters:
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
class BackwardIOParameters(IOParameters):
    """Parameters that modify the backward IO behavior.

    This class contains the same parameters as
    ``AnalogTileInputOutputParameters``, specializing the default value of
    ``bound_management`` (as backward does not support bound management).
    """

    bound_management: BoundManagementType = BoundManagementType.NONE
    """Type of noise management, see :class:`NoiseManagementType`."""


@dataclass
class UpdateParameters:
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


def parameters_to_bindings(params: Any) -> Any:
    """Convert a dataclass parameter into a bindings class."""
    result = params.bindings_class()
    for field, value in params.__dict__.items():
        # Convert enums to the bindings enums.
        if field == 'unit_cell_devices':
            # Exclude `unit_cell_devices`, as it is a special field that is not
            # present in the bindings.
            continue

        if isinstance(value, Enum):
            enum_class = getattr(devices, value.__class__.__name__)
            enum_value = getattr(enum_class, value.value)
            setattr(result, field, enum_value)
        elif is_dataclass(value):
            setattr(result, field, parameters_to_bindings(value))
        else:
            setattr(result, field, value)

    return result


def tile_parameters_to_bindings(params: Any) -> Any:
    """Convert a tile dataclass parameter into a bindings class."""
    field_map = {'forward': 'forward_io',
                 'backward': 'backward_io'}

    result = params['bindings_class']() if isinstance(params, dict) \
        else params.bindings_class()

    params_dic = params if isinstance(params, dict) else params.__dict__

    for field, value in params_dic.items():
        # Get the mapped field name, if needed.
        field = field_map.get(field, field)

        # Convert enums to the bindings enums.
        if field in ['bindings_class', 'device']:
            # Exclude `device`, as it is a special field that is not
            # present in the bindings.
            continue

        if isinstance(value, Enum):
            enum_class = getattr(devices, value.__class__.__name__)
            enum_value = getattr(enum_class, value.value)
            setattr(result, field, enum_value)
        elif is_dataclass(value):
            setattr(result, field, parameters_to_bindings(value))
        else:
            setattr(result, field, value)

    return result

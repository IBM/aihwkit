# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=too-many-instance-attributes

"""Forward / backward / update related parameters for resistive processing units."""

from dataclasses import dataclass
from typing import ClassVar, Type, Optional, Union

from .helpers import _PrintableMixin
from .enums import PulseType


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

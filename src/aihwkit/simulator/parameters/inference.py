# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-lines

"""Inference related parameters for resistive processing units."""

from dataclasses import dataclass, field
from typing import ClassVar, Type, List, Optional, Union

from aihwkit.simulator.parameters.helpers import _PrintableMixin
from aihwkit.simulator.rpu_base import tiles
from aihwkit.simulator.parameters.enums import WeightModifierType, WeightClipType, WeightRemapType


@dataclass
class WeightModifierParameter(_PrintableMixin):
    """Parameter that modify the forward/backward weights during hardware-aware training."""

    bindings_class: ClassVar[Optional[Union[str, Type]]] = "WeightModifierParameter"
    bindings_module: ClassVar[str] = "tiles"

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

    per_batch_sample: bool = False
    """Should we resample noise for each sample in the batch.

    This parameter only affects is used when using the
    ``TorchSimulatorTile``. In case of ``RPUCudaTile`` it will throw
    an error.
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
        metadata={"hide_if": [0.0105392, 0.0768, -0.046925]},
    )
    """Coefficients for the ``POLY`` weight modifier type.

    See :class:`WeightModifierType` for details.
    """

    type: WeightModifierType = field(
        default_factory=lambda: WeightModifierType.NONE, metadata={"always_show": True}
    )
    """Type of the weight modification."""

    g_max: float = 25.0
    r"""PCM_NOISE and PROG_NOISE parameter, :math:`g_\text{max}`
    setting in :math:`\mu S`."""

    pcm_zero_thres: float = 0.0
    """PCM_NOISE parameter """

    pcm_t_inference: float = 0.0
    """PCM_NOISE parameter, time of inference. """

    pcm_prob_at_reset: float = 0.0
    """PCM_NOISE parameter, probability of reset. """

    pcm_prob_at_gmax: float = 0.0
    r"""PCM_NOISE parameter, probability of devices being at :math:`g_\text{max}`. """

    pcm_prob_at_random: float = 0.0
    r"""PCM_NOISE parameter, probability of devices being at random value in the range. """

    pcm_t0: float = 20.0
    r"""PCM_NOISE parameter,  programming conversion time in seconds. """


@dataclass
class WeightClipParameter(_PrintableMixin):
    """Parameter that clip the weights during hardware-aware training.

    Important:
        A clipping ``type`` has to be set before any of the parameter
        changes take any effect.

    """

    bindings_class: ClassVar[Optional[Union[str, Type]]] = tiles.WeightClipParameter

    fixed_value: float = -1.0
    """Clipping value in case of ``FixedValue`` type.

    Caution:

        If ``fixed_value > 0`` it will be also applied during other
        clipping types.

    """

    sigma: float = 2.5
    """Sigma value for clipping for the ``LayerGaussian`` type."""

    type: WeightClipType = field(
        default_factory=lambda: WeightClipType.NONE, metadata={"always_show": True}
    )
    """Type of clipping."""


@dataclass
class WeightRemapParameter(_PrintableMixin):
    """Parameter that remap the weights during hardware-aware training.

    Important:
        A remap ``type`` has to be set before any of the parameter
        changes take any effect.
    """

    bindings_class: ClassVar[Optional[Union[str, Type]]] = tiles.WeightRemapParameter

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

    type: WeightRemapType = field(
        default_factory=lambda: WeightRemapType.NONE, metadata={"always_show": True}
    )
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

    bindings_class: ClassVar[Optional[Union[str, Type]]] = "DriftParameter"
    bindings_module: ClassVar[str] = "devices"

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

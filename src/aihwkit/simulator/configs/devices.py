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

"""Configuration for Analog (Resistive Device) tiles."""

# pylint: disable=too-many-instance-attributes, too-many-lines

from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, List, Type, Union
from warnings import warn
from numpy import exp

from aihwkit.exceptions import ConfigError
from aihwkit.simulator.configs.helpers import (
    _PrintableMixin, parameters_to_bindings
)
from aihwkit.simulator.configs.utils import (
    IOParameters, UpdateParameters, VectorUnitCellUpdatePolicy,
    DriftParameter, SimpleDriftParameter
)
from aihwkit.simulator.rpu_base import devices


@dataclass
class FloatingPointDevice(_PrintableMixin):
    """Floating point reference.

    Implements ideal devices forward/backward/update behavior.
    """

    bindings_class: ClassVar[Type] = devices.FloatingPointTileParameter

    diffusion: float = 0.0
    """Standard deviation of diffusion process."""

    lifetime: float = 0.0
    r"""One over `decay_rate`, ie :math:`1/r_\text{decay}`."""

    drift: SimpleDriftParameter = field(default_factory=SimpleDriftParameter)
    """Parameter governing a power-law drift."""

    def as_bindings(self) -> devices.FloatingPointTileParameter:
        """Return a representation of this instance as a simulator bindings object."""
        return parameters_to_bindings(self)

    def requires_diffusion(self) -> bool:
        """Return whether device has diffusion enabled."""
        return self.diffusion > 0.0

    def requires_decay(self) -> bool:
        """Return whether device has decay enabled."""
        return self.lifetime > 0.0


@dataclass
class PulsedDevice(_PrintableMixin):
    r"""Pulsed update resistive devices.

    Device are used as part of an
    :class:`~aihwkit.simulator.tiles.AnalogTile` to implement the
    `update once` characteristics, i.e. the material response properties
    when a single update pulse is given (a coincidence between row and
    column pulse train happened).

    Common properties of all pulsed devices include:

    **Reset**:

    Resets the weight in cross points to (around) zero with
    cycle-to-cycle and systematic spread around a mean.

    Important:
        Reset with given parameters is only activated when
        :meth:`~aihwkit.simulator.tiles.base.Base.reset_weights` is
        called explicitly by the user.

    **Decay**:

    .. math:: w_{ij} \leftarrow (w_{ij} - b_{ij})\,(1-\alpha_\text{decay}\delta_{ij}) + b_{ij}

    Weight decay is only activated by inserting a specific call to
    :meth:`~aihwkit.simulator.tiles.base.Base.decay_weights`, which is
    done automatically for a tile each mini-batch is decay is
    present. Note that the device ``decay_lifetime`` parameters (1
    over decay rates :math:`\delta_{ij}`) are analog tile specific and
    are thus set and fixed during RPU
    initialization. :math:`\alpha_\text{decay}` is a scaling factor
    that can be given during run-time.

    The bias :math:`b_{ij}` is given by the reset bias and which is
    determined by the parameter ``reset`` (mean value) and
    ``reset_dtod``  (device-to-device variability). Thus

    .. math:: b_{ij} = \mu_\text{reset} \left(1 + \sigma_\text{reset-dtod}\xi\right)

    Note that the reset bias is also applied in case the device is
    reset (see above).


    **Diffusion**:

    Similar to the decay, diffusion is only activated by inserting a
    specific call to
    :meth:`~aihwkit.simulator.tiles.base.Base.diffuse_weights`, which is
    done automatically for a tile each mini-batch is diffusion is
    present. The parameters of the diffusion process are set during
    RPU initialization and are fixed for the remainder.

    .. math:: w_{ij} \leftarrow w_{ij} + \rho_{ij} \, \xi;

    where :math:`xi` is a standard Gaussian variable and :math:`\rho_{ij}` the
    diffusion rate for a cross-point `ij`.

    Note:
        If diffusion happens to move the weight beyond the hard bounds of the
        weight it is ensured to be clipped appropriately.

    **Drift**:

    Optional power-law drift setting, as described in
    :class:`~aihwkit.similar.configs.utils.DriftParameter`.

    Important:
        Similar to reset, drift is *not* applied automatically each
        mini-batch but requires an explicit call to
        :meth:`~aihwkit.simulator.tiles.base.Base.drift_weights` each
        time the drift should be applied.

    """

    bindings_class: ClassVar[Type] = devices.PulsedResistiveDeviceParameter

    construction_seed: int = 0
    """If not equal 0, will set a unique seed for hidden parameters during
    construction."""

    corrupt_devices_prob: float = 0.0
    """Probability for devices to be corrupt (weights fixed to random value
    with hard bounds, that is min and max bounds are set to equal)."""

    corrupt_devices_range: int = 1000
    """Range around zero for establishing corrupt devices."""

    diffusion: float = 0.0
    """Standard deviation of diffusion process."""

    diffusion_dtod: float = 0.0
    """Device-to device variation of diffusion rate in relative units."""

    drift: DriftParameter = field(default_factory=DriftParameter,
                                  metadata={'hide_if': DriftParameter()})
    """Parameter governing a power-law drift."""

    dw_min: float = 0.001
    """Mean of the minimal update step sizes across devices and directions."""

    dw_min_dtod: float = 0.3
    """Device-to-device std deviation of dw_min (in relative units to
    ``dw_min``)."""

    dw_min_std: float = 0.3
    r"""Cycle-to-cycle variation size of the update step (related to
    :math:`\sigma_\text{c-to-c}` above) in relative units to ``dw_min``.

    Note:
        Many spread (device-to-device variation) parameters are given in
        relative units. For instance e.g. a setting of ``dw_min_std`` of 0.1
        would mean 10% spread around the mean and thus a resulting standard
        deviation (:math:`\sigma_\text{c-to-c}`) of ``dw_min`` * ``dw_min_std``.
    """

    enforce_consistency: bool = True
    """Whether to enforce weight bounds consistency during initialization.

    Whether to enforce that max weight bounds cannot be smaller than min
    weight bounds, and up direction step size is positive and down negative.
    Switches the opposite values if encountered during init.
    """

    lifetime: float = 0.0
    r"""One over `decay_rate`, ie :math:`1/r_\text{decay}`."""

    lifetime_dtod: float = 0.0
    """Device-to-device variation in the decay rate (in relative units)."""

    perfect_bias: bool = False
    """No up-down differences and device-to-device variability in the bounds
    for the devices in the bias row."""

    reset: float = 0.0
    """The reset values and spread per cross-point ``ij`` when using reset
    functionality of the device."""

    reset_dtod: float = 0.0
    """See ``reset``."""

    reset_std: float = 0.01
    """See ``reset``."""

    up_down: float = 0.0
    r"""Up and down direction step sizes can be systematically different and
    also vary across devices.

    :math:`\Delta w_{ij}^d` is set during RPU initialization (for each
    cross-point :math:`ij`):

    .. math::

        \Delta w_{ij}^d = d\; \Delta w_\text{min}\, \left(
        1 + d \beta_{ij} + \sigma_\text{d-to-d}\xi\right)

    where :math:`\xi` is again a standard Gaussian. :math:`\beta_{ij}` is the
    directional up `versus` down bias.  At initialization ``up_down_dtod`` and
    ``up_down`` defines this bias term:

    .. math::

        \beta_{ij} = \beta_\text{up-down} + \xi
        \sigma_\text{up-down-dtod}

    where :math:`\xi` is again a standard Gaussian number and
    :math:`\beta_\text{up-down}` corresponds to ``up_down``. Note that
    ``up_down_dtod`` is again given in relative units to ``dw_min``.
    """

    up_down_dtod: float = 0.01
    """See ``up_down``."""

    w_max: float = 0.6
    """See ``w_min``."""

    w_max_dtod: float = 0.3
    """See ``w_min_dtod``."""

    w_min: float = -0.6
    """Mean of hard bounds across device cross-point `ij`.

    The parameters ``w_min`` and ``w_max`` are used to set the min/max bounds
    independently.

    Note:
        For this abstract device, we assume that weights can have
        positive and negative values and are symmetrically around
        zero. In physical circuit terms, this might be implemented
        as a difference of two resistive elements.
    """

    w_min_dtod: float = 0.3
    """Device-to-device variation of the hard bounds.

    Device-to-device variation of the hard bounds, of min and max value,
    respectively. All are given in relative units to ``w_min``, or ``w_max``,
    respectively.
    """

    def as_bindings(self) -> devices.PulsedResistiveDeviceParameter:
        """Return a representation of this instance as a simulator bindings object."""
        return parameters_to_bindings(self)

    def requires_diffusion(self) -> bool:
        """Return whether device has diffusion enabled."""
        return self.diffusion > 0.0

    def requires_decay(self) -> bool:
        """Return whether device has decay enabled."""
        return self.lifetime > 0.0


@dataclass
class UnitCell(_PrintableMixin):
    """Parameters that modify the behaviour of a unit cell."""

    bindings_class: ClassVar[Type] = devices.VectorResistiveDeviceParameter

    unit_cell_devices: List = field(default_factory=list)
    """Devices that compose this unit cell."""

    def as_bindings(self) -> devices.VectorResistiveDeviceParameter:
        """Return a representation of this instance as a simulator bindings object."""
        raise NotImplementedError

    def requires_diffusion(self) -> bool:
        """Return whether device has diffusion enabled."""
        return any(dev.requires_diffusion() for dev in self.unit_cell_devices)

    def requires_decay(self) -> bool:
        """Return whether device has decay enabled."""
        return any(dev.requires_decay() for dev in self.unit_cell_devices)


###############################################################################
# Specific devices based on ``pulsed``.
###############################################################################

@dataclass
class IdealDevice(_PrintableMixin):
    """Ideal update behavior (using floating point), but forward/backward
    might be non-ideal.

    Ideal update behavior (using floating point), however,
    forward/backward might still have a non-ideal ADC or noise added.
    """

    bindings_class: ClassVar[Type] = devices.IdealResistiveDeviceParameter

    construction_seed: int = 0
    """If not ``0``, set a unique seed for hidden parameters during
    construction."""

    diffusion: float = 0.0
    """Standard deviation of diffusion process."""

    lifetime: float = 0.0
    r"""One over `decay_rate`, ie :math:`1/r_\text{decay}`."""

    def as_bindings(self) -> devices.IdealResistiveDeviceParameter:
        """Return a representation of this instance as a simulator bindings object."""
        return parameters_to_bindings(self)

    def requires_diffusion(self) -> bool:
        """Return whether device has diffusion enabled."""
        return self.diffusion > 0.0

    def requires_decay(self) -> bool:
        """Return whether device has decay enabled."""
        return self.lifetime > 0.0


@dataclass
class ConstantStepDevice(PulsedDevice):
    r"""Pulsed update behavioral model: constant step.

    Pulsed update behavioral model, where the update step of
    material is constant throughout the resistive range (up to hard
    bounds).

    In more detail, the update behavior implemented for ``ConstantStep``
    is:

    .. math::

        w_{ij}  &\leftarrow&  w_{ij} - \Delta w_{ij}^d(1 + \sigma_\text{c-to-c}\,\xi)

        w_{ij}  &\leftarrow& \text{clip}(w_{ij},b^\text{min}_{ij},b^\text{max}_{ij})

    where :math:`d` is the direction of the update (product of signs
    of input and error). :math:`\Delta w_{ij}^d` is the update step
    size of the cross-point `ij` in direction :math:`d` (up or down).
    Note that each cross-point has separate update sizes so that
    device-to-device fluctuations and biases in the directions can be
    given.

    Moreover, the clipping bounds of each cross-point `ij`
    (i.e. :math:`b_{ij}^\text{max/min}`) are also different in
    general. The mean and the amount of systematic spread from
    device-to-device can be given as parameters, see below.

    For parameters regarding the devices settings, see e.g.
    :class:`~PulsedDevice`.
    """

    bindings_class: ClassVar[Type] = devices.ConstantStepResistiveDeviceParameter


@dataclass
class LinearStepDevice(PulsedDevice):
    r"""Pulsed update behavioral model: linear step.

    Pulsed update behavioral model, where the update step response
    size of the material is linearly dependent with resistance (up to
    hard bounds).

    This model is based on :class:`~PulsedDevice` and thus
    shares all parameters and functionality. In addition, it only
    implements a more general `update once` function, where the update
    step size can depend linearly on the weight itself.

    For each coincidence the weights is updated once. Here, the
    positive (negative) update step size decreases linearly in the
    following manner (compare to the `update once` for
    :class:`~ConstantStepDevice`):

    .. math::
        :nowrap:

        \begin{eqnarray*}
        w_{ij}  &\leftarrow&  w_{ij} - \Delta w_{ij}^d(\gamma_{ij}^d\;w_{ij}
        + 1 + \sigma_\text{c-to-c}\,\xi)\\
        w_{ij}  &\leftarrow& \text{clip}(w_{ij},b^\text{min}_{ij},b^\text{max}_{ij})
        \end{eqnarray*}

    in case of additive noise.  Optionally, multiplicative noise can
    be chosen in which case the first equation becomes:

    .. math::

        w_{ij}  \leftarrow  w_{ij} - \Delta w_{ij}^d (\gamma_{ij}^d \;w_{ij} + 1)
        (1 + \sigma_\text{c-to-c}\,\xi)

    The cross-point `ij` dependent slope parameter
    :math:`\gamma_{ij}^d` are given during initialization by

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \gamma_{ij}^+ &=& - |\gamma^+ + \gamma_\text{d-to-d}^+ \xi|/b^\text{max}_{ij}\\
        \gamma_{ij}^- &=& - |\gamma^- + \gamma_\text{d-to-d}^- \xi|/b^\text{min}_{ij}
        \end{eqnarray*}

    where the :math:`\xi` are standard Gaussian random variables and
    :math:`b^\text{min}_{ij}` and :math:`b^\text{max}_{ij}` the
    cross-point `ij` specific minimal and maximal weight bounds,
    respectively (see description for :class:`~PulsedDevice`).

    Note:
        If :math:`\gamma=1` and :math:`\gamma_\text{d-to-d}=0` this
        update implements `soft bounds`, since the updates step becomes
        equal to :math:`1/b`.

    Note:
        If :math:`\gamma=0` and :math:`\gamma_\text{d-to-d}=0` and
        additive noise, this update is identical to those described in
        :class:`~PulsedDevice`.
    """

    bindings_class: ClassVar[Type] = devices.LinearStepResistiveDeviceParameter

    gamma_up: float = 0.0
    r"""The value of :math:`\gamma^+`.

    Intuitively, a value of 0.1 means that the update step size in up direction
    at the weight bounds is 10% decreased relative to that origin :math:`w=0`.

    Note:
        In principle one could fix :math:`\gamma=\gamma^-=\gamma^+` since
        up/down variation can be given by ``up_down_dtod``, see
        :class:`~PulsedDevice`.

    Note:
        The hard-bounds are still observed, so that the weight cannot
        grow beyond its bounds.
    """

    gamma_down: float = 0.0
    r"""The value of :math:`\gamma^-`."""

    gamma_up_dtod: float = 0.05
    r"""Device-to-device variation for :math:`\gamma^+`, i.e. the value of
    :math:`\gamma_\text{d-to-d}^+`."""

    gamma_down_dtod: float = 0.05
    r"""Device-to-device variation for :math:`\gamma^-`, i.e. the value of
    :math:`\gamma_\text{d-to-d}^-`."""

    allow_increasing: bool = False
    """Whether to allow increasing of update sizes.

    Whether to allow the situation where update sizes increase towards the
    bound instead of saturating (and thus becoming smaller).
    """

    mean_bound_reference: bool = True
    r"""Whether to use instead of the above:

    .. math::

        \gamma_{ij}^+ &=& - |\gamma^+ + \gamma_\text{d-to-d}^+ \xi|/b^\text{max}

        \gamma_{ij}^- &=& - |\gamma^- + \gamma_\text{d-to-d}^- \xi|/b^\text{min}

    where :math:`b^\text{max}` and :math:`b^\text{max}` are the values given by
    ``w_max`` and ``w_min``, see :class:`~PulsedDevice`.
    """

    mult_noise: bool = True
    """Whether to use multiplicative noise instead of additive cycle-to-cycle
    noise."""

    write_noise_std: float = 0.0
    r"""Whether to use update write noise.

    Whether to use update write noise that is added to the updated
    devices weight, while the update is done on a hidden persistent weight. The
    update write noise is then sampled anew when the device is touched
    again.

    Thus it is:

    .. math::
        w_\text{apparent}{ij} = w_{ij} + \sigma_\text{write_noise} \Delta w_\text{min}\xi

    and the update is done on :math:`w_{ij}` but the forward sees the
    :math:`w_\text{apparent}`.
    """

    reverse_up: bool = False
    """Whether to increase the step size in up direction with increasing
    weights (default decreases).

    Note:
        If set, ``mult_noise`` needs to be also set.
    """

    reverse_down: bool = False
    """Whether to increase the step size in down direction with decreasing
    weights (default decreases).

    Note:
        If set, ``mult_noise`` needs to be also set.

    """

    reverse_offset: float = 0.01
    """Offset to add to the step size for reverse up or down to avoid
    zero step size at weight min or max.
    """


@dataclass
class SoftBoundsDevice(PulsedDevice):
    r"""Pulsed update behavioral model: soft bounds.

    Pulsed update behavioral model, where the update step response size
    of the material is linearly dependent and it goes to zero at the
    bound.

    This model is based on :class:`~LinearStepDevice` with
    parameters set to model soft bounds.
    """

    bindings_class: ClassVar[Type] = devices.SoftBoundsResistiveDeviceParameter

    mult_noise: bool = True
    """Whether to use multiplicative noise instead of additive cycle-to-cycle
    noise."""

    write_noise_std: float = 0.0
    r"""Whether to use update write noise.

    Whether to use update write noise that is added to the updated
    devices weight, while the update is done on a hidden persistent weight. The
    update write noise is then sampled anew when the device is touched
    again.

    Thus it is:

    .. math::
        w_\text{apparent}{ij} = w_{ij} + \sigma_\text{write_noise} \Delta w_\text{min}\xi

    and the update is done on :math:`w_{ij}` but the forward sees the
    :math:`w_\text{apparent}`.
    """

    reverse_up: bool = False
    """Whether to increase the step size in up direction with increasing
    weights (default decreases).

    Note:
        If set, ``mult_noise`` needs to be also set.
    """

    reverse_down: bool = False
    """Whether to increase the step size in down direction with decreasing
    weights (default decreases).

    Note:
        If set, ``mult_noise`` needs to be also set.
    """

    reverse_offset: float = 0.01
    """Offset to add to the step size for reverse up or down to avoid
    zero step size at weight min or max.
    """


@dataclass
class SoftBoundsPmaxDevice(SoftBoundsDevice):
    r"""Pulsed update behavioral model: soft bounds, with a different
    parameterization for easier device fitting to experimental data.

    Under the hood, the same  device behavior as :class:`~SoftboundsDevice`
    This model is based on :class:`~LinearStepDevice` with
    parameters set to model soft bounds.

    It implements pulse response function of the form:

    .. math::

        w(p_\text{up}) = B\left(1 -e^{-\alpha p_\text{up}} \right) + r_\text{min}

        w(p_\text{down}) = - B\left(1 - e^{-\alpha (p_\text{max}
        - p_\text{down})}\right) + r_\text{max}

    where :math:`B=\frac{r_\text{max} -
    r_\text{min}}{1 - e^{-\alpha p_\text{max}}}`.

    Here :math:`p_\text{max}` is the number of pulses that were applied to get
    the device from the minimum conductance (minimum of range,
    :math:`r_\text{min}`) to the maximum (maximum of range,
    :math:`r_\text{max}`).

    Internally the following transformation is used to get the
    original parameter of :class:`SoftboundsDevice`::

        b_factor = (range_max - range_min)/(1 - exp(-p_max * alpha))
        w_min = range_min
        w_max = range_min + b_factor
        dw_min = b_factor * alpha
        up_down = 1 + 2 * range_min / b_factor

    Note:
        Device-to-device and cycle-to-cycle variation are defined as
        before (see :class:`SoftBoundsDevice`, see also
        :class:`PulsedDevice`). That is, for instance `dw_min_dtod`
        will effectively change the slope (in units of ``dw_min`` which
        is ``b_factor * alpha``, see above). Range offset fluctuations
        can be achieved by using ``w_min_dtod`` and ``w_max_dtod``
        which will vary ``w_min`` and ``w_max`` across devices,
        respectively.
    """

    p_max: int = 1000
    """Number of pulses to drive the synapse from ``range_min`` to ``range_max``."""

    alpha: float = 0.001/2
    r"""The slope of the soft bounds model :math:`dw \propto \alpha w` for both
    up and down direction."""

    range_min: float = -1.0
    """Setting of the weight when starting the :math:`P_max` up pulse
    experiment."""

    range_max: float = 1.0
    """Value of the weight for :math:`P_max` number of up pulses."""

    #  these values will be set from the above, so we hide it.
    w_min: float = field(default_factory=lambda: None, metadata={'hide_if': None})  # type: ignore
    w_max: float = field(default_factory=lambda: None, metadata={'hide_if': None})  # type: ignore
    dw_min: float = field(default_factory=lambda: None, metadata={'hide_if': None})  # type: ignore
    up_down: float = field(default_factory=lambda: None, metadata={'hide_if': None})  # type: ignore

    def as_bindings(self) -> devices.PulsedResistiveDeviceParameter:
        """Return a representation of this instance as a simulator bindings object."""
        params = SoftBoundsDevice()
        for key, value in self.__dict__.items():
            if key not in ['range_min', 'range_max', 'alpha', 'p_max']:
                setattr(params, key, value)

        b_factor = (self.range_max - self.range_min)/(1 - exp(-self.p_max * self.alpha))
        params.w_min = self.range_min
        params.w_max = self.range_min + b_factor
        params.dw_min = b_factor * self.alpha
        params.up_down = 1 + 2 * self.range_min / b_factor

        return parameters_to_bindings(params)


@dataclass
class ExpStepDevice(PulsedDevice):
    r"""Exponential update step or CMOS-like update behavior.

    This model is derived from ``PulsedDevice`` and uses all its
    parameters. ``ExpStepDevice`` only implements a new 'update once'
    functionality, where the minimal weight step change with weight is
    fitted by an exponential function as detailed below.

    .. math::

        w_{ij}  \leftarrow  w_{ij} -  \max(y_{ij},0)  \Delta w_{ij}^d
        (1 + \sigma_\text{c-to-c}\,\xi)

    and :math:`y_{ij}` is given as

    .. math::
        z_{ij} = 2 a_\text{es} \frac{w_{ij}}{b^\text{max}_{ij} - b^\text{min}_{ij}}
        + b_\text{es}

        y_{ij} = 1 - A^{(d)} e^{d \gamma^{(d)} z_{ij}}

    where :math:`d` is the direction of the update (+ or -), see also
    :class:`~PulsedDevice` for details.

    All additional parameter (:math:`a_\text{es}`,
    :math:`b_\text{es}`, :math:`\gamma^{(d)}`, :math:`A^{(d)}` ) are
    tile-wise fitting parameters (ie. no device-to-device variation in
    these parameters).  Note that the other parameter involved can be
    still defined with device-to-device variation and (additional)
    up-down bias (see :class:`~PulsedDevice`).

    Note:
        This device also features a more complex cycle-to-cycle noise
        model of the update step, when specifying ``dw_min_std_add``
        and ``dw_min_std_slope``. By default,    The Gaussian noise added to the
        calculated update step size :math:`\Delta q_\text{act}` is
        proportional to

        .. math::
           \sigma_{\text{final}} = \sigma \left( \sigma_\text{add} +
           |\Delta w_\text{actual}| + \sigma_\text{slope} |w_\text{current}|\right)

        where the :math:`\sigma` is given by ``dw_min_std``,
        :math:`\sigma_\text{add}` is given by ``dw_min_std_add``, and
        :math:`\sigma_\text{slope}` is given by ``dw_min_std_slope``.
    """
    # pylint: disable=invalid-name

    bindings_class: ClassVar[Type] = devices.ExpStepResistiveDeviceParameter

    A_up: float = 0.00081
    """Factor ``A`` for the up direction."""

    A_down: float = 0.36833
    """Factor ``A`` for the down direction."""

    gamma_up: float = 12.44625
    """Exponent for the up direction."""

    gamma_down: float = 12.78785
    """Exponent for the down direction."""

    a: float = 0.244
    """Global slope parameter."""

    b: float = 0.2425
    """Global offset parameter."""

    dw_min_std_add: float = 0.0
    """additive cycle-to-cycle noise of the update size (in units of
    ``dw_min_std``, see above)."""

    dw_min_std_slope: float = 0.0
    """ cycle-to-cycle noise of the update size (in units of ``dw_min_std``, see above)."""

    write_noise_std: float = 0.0
    r"""Whether to use update write noise.

    Whether to use update write noise that is added to the updated
    devices weight, while the update is done on a hidden persistent weight. The
    update write noise is then sampled a new when the device is touched
    again.

    Thus it is:

    .. math::
        w_\text{apparent}{ij} = w_{ij} + \sigma_\text{write_noise}\xi

    and the update is done on :math:`w_{ij}` but the forward sees the
    :math:`w_\text{apparent}`.
    """


@dataclass
class PowStepDevice(PulsedDevice):
    r"""Pulsed update behavioral model: power-dependent step.

    Pulsed update behavioral model, where the update step response
    size of the material has a power-dependent with resistance. This
    device model implements (a shifted from of) the `Fusi & Abott
    (2007)`_ synapse model (see also `Frascaroli et al. (2108)`_).

    The model based on :class:`~PulsedDevice` and thus shares most
    parameters and functionality. However, it implements new `update
    once` function, where the update step size depends in the
    following way. If we set :math:`\omega_{ij} =
    \frac{b_{ij}^\text{max} - w_{ij}}{b_{ij}^\text{max} -
    b_{ij}^\text{min}}` the relative distance of the current weight to
    the upper bound, then the update per pulse is for the upwards direction:

    .. math::
        w_{ij}  \leftarrow  w_{ij} + \Delta w_{ij}^+\,(\omega_{ij})^{\gamma_{ij}^+}
        \left(1 + \sigma_\text{c-to-c}\,\xi\right)

    and in downwards direction:

    .. math::
        w_{ij}  \leftarrow  w_{ij} + \Delta w_{ij}^-\,(1 - \omega_{ij})^{\gamma_{ij}^-}
        \left(1 + \sigma_\text{c-to-c}\,\xi\right)

    Similar to :math:`\Delta w_{ij}^d` the exponent :math:`\gamma_{ij}` can be
    defined with device-to-device variation and bias in up and down
    direction:

    .. math::

        \gamma_{ij}^d = \gamma\, \left(1 + d\, \beta_{ij}
        + \sigma_\text{pow-gamma-d-to-d}\xi\right)

    where :math:`\xi` is again a standard Gaussian. :math:`\beta_{ij}`
    is the directional up `versus` down bias.  At initialization
    ``pow_up_down_dtod`` and ``pow_up_down`` defines this bias term:

    .. math::

        \beta_{ij} = \beta_\text{pow-up-down} + \xi\sigma_\text{pow-up-down-dtod}

    where :math:`\xi` is again a standard Gaussian number and
    :math:`\beta_\text{pow-up-down}` corresponds to ``pow_up_down``.

    Note:
        The ``pow_gamma_dtod`` and ``pow_up_down_dtod``
        device-to-device variation parameters are given in relative
        units to ``pow_gamma``.

    Note:
        :math:`\Delta w_{ij}^d` is defined as for the
        :class:`~PulsedDevice`, however, for this device, the update step
        size will *not* be given by :math:`\Delta w_{ij}` at
        :math:`w_{ij}=0` as for most other devices models

    ..  _Fusi & Abott (2007): https://www.nature.com/articles/nn1859
    ..  _Frascaroli et al. (2108): https://www.nature.com/articles/s41598-018-25376-x
    """

    bindings_class: ClassVar[Type] = devices.PowStepResistiveDeviceParameter

    pow_gamma: float = 1.0
    r"""The value of :math:`\gamma` as explained above.

    Note:
        :math:`\gamma` reduces essentially to the
        :class:`SoftBoundsDevice` (if no device-to-device variation of
        gamma is used additionally). However, the
        :class:`SoftBoundsDevice` will be much faster, as it does not
        need to compute the slow `pow` function.
    """

    pow_gamma_dtod: float = 0.1
    r"""Device-to-device variation for ``pow_gamma``.

    i.e. the value of :math:`\gamma_\text{pow-gamma-d-to-d}` given in relative
    units to ``pow_gamma``.
    """

    pow_up_down: float = 0.0
    r"""The up versus down bias of the :math:`\gamma` as described above.

    It is :math:`\gamma^+ = \gamma (1 + \beta_\text{pow-up-down})` and
    :math:`\gamma^- = \gamma (1 - \beta_\text{pow-up-down})` .
    """

    pow_up_down_dtod: float = 0.0
    r"""Device-to-device variation in the up versus down bias of
    :math:`\gamma` as descibed above.

    In units of ``pow_gamma``.
    """

    write_noise_std: float = 0.0
    r"""Whether to use update write noise.

    Whether to use update write noise that is added to the updated
    devices weight, while the update is done on a hidden persistent weight. The
    update write noise is then sampled a new when the device is touched
    again.

    Thus it is:

    .. math::
        w_\text{apparent}{ij} = w_{ij} + \sigma_\text{write_noise}\xi

    and the update is done on :math:`w_{ij}` but the forward sees the
    :math:`w_\text{apparent}`.
    """


@dataclass
class PiecewiseStepDevice(PulsedDevice):
    r"""Piece-wise interpolated device update characteristics.

    This model is derived from :class:`~PulsedDevice` and uses all its
    parameters. :class:`~PiecewiseStepDevice` implements a new
    functionality where the device's update response curve is given
    explicitly on nodes over the weight range. The device will
    automatically interpolate the update step size using the given node
    values.

    In detail, the update in down direction of the device is given as:

    .. math::
        w_{ij} \leftarrow w_{ij} + \Delta w_{ij}^d \left((1 - q) v^d(i_w)
        + q \, v^d(i_w + 1)\right) (1 + \sigma_\text{c-to-c}\,\xi)

    where :math:`i_w` is the index of the given vector :math:`v^d`
    (``piecewise_down``) where the current weight value would fall
    into if scaled to the current min and max values of that device
    (first and last value are set at weight min and max, respectively,
    the other values are equally distributed in the range).

    The scalar :math:`q` is the relative position of the weight in the
    current segment between the two selected nodes :math:`v^d(i_w)`
    and :math:`v^d(i_w + 1)`.

    The update in up direction is computed analogously using the
    ``piecewise_up`` vector.

    Note:

        The piecewise up and down vectors need to have the same number
        of elements.

    Note:

        In case of GPUs the maximal number of nodes in the vectors
        is limited to below 64 due to performance reasons.

    """

    bindings_class: ClassVar[Type] = devices.PiecewiseStepResistiveDeviceParameter

    piecewise_up: List[float] = field(default_factory=lambda: [1])
    r"""Array of values that characterize the update steps in upwards direction.

    The values are equally spaced in ``w_min`` and `w_max`` (which
    could vary from device-to-device), where the first and the last
    value is set at the boundary. The update will be computed by
    linear interpolation of the adjacent values, depending on where
    the weight is currently within the range.

    The values are given as relative numbers: the final update size
    will be computed by multiplying the value with the current
    ``dw_min`` of the device.

    E.g.  ``[1.5, 1, 1.5]`` and ``dw_min=0.1`` means that the update
    (in up direction) is ``dw_min`` around zero weight value and
    linearly increasing to ``1.5 * dw_min`` for larger or smaller
    weight values.

    """

    piecewise_down: List[float] = field(default_factory=lambda: [1])
    r"""Array of values that characterize the update steps in downwards direction.

    Analogous to ``piecewise_up`` but for the downwards direction.
    """

    write_noise_std: float = 0.0
    r"""Whether to use update write noise.

    Whether to use update write noise that is added to the updated
    devices weight, while the update is done on a hidden persistent weight. The
    update write noise is then sampled a new when the device is touched
    again.

    Thus it is:

    .. math::
        w_\text{apparent}{ij} = w_{ij} + \sigma_\text{write_noise}\xi

    and the update is done on :math:`w_{ij}` but the forward sees the
    :math:`w_\text{apparent}`.
    """


###############################################################################
# Specific devices based on ``unit cell``.
###############################################################################

@dataclass
class VectorUnitCell(UnitCell):
    """Abstract resistive device that combines multiple pulsed resistive
    devices in a single 'unit cell'.

    For instance, a vector device can consist of 2 resistive devices
    where the sum of the two resistive values are coded for each
    weight of a cross point.
    """

    bindings_class: ClassVar[Type] = devices.VectorResistiveDeviceParameter

    update_policy: VectorUnitCellUpdatePolicy = VectorUnitCellUpdatePolicy.ALL
    """The update policy of which if the devices will be receiving the update
    of a mini-batch."""

    first_update_idx: int = 0
    """Device that receives the first mini-batch.

    Useful only for ``VectorUnitCellUpdatePolicy.SINGLE_FIXED``.
    """

    gamma_vec: List[float] = field(default_factory=list, metadata={'hide_if': []})
    """Weighting of the unit cell devices to reduce to final weight.

    User-defined weightening can be given as a list if factors. If not
    given, each device index of the unit cell is weighted by equal
    amounts (:math:`1/n`).
    """

    def as_bindings(self) -> devices.VectorResistiveDeviceParameter:
        """Return a representation of this instance as a simulator bindings object."""
        vector_parameters = parameters_to_bindings(self)

        if not isinstance(self.unit_cell_devices, list):
            raise ConfigError('unit_cell_devices should be a list of devices')

        for param in self.unit_cell_devices:
            device_parameters = param.as_bindings()
            if not vector_parameters.append_parameter(device_parameters):
                raise ConfigError('Could not add unit cell device parameter')

        return vector_parameters


@dataclass
class ReferenceUnitCell(UnitCell):
    """Abstract device model takes two arbitrary device per cross-point and
    implements an device with reference pair.

    The update will only be on the 0-th device whereas the other will
    stay fixed. The resulting effective weight is the difference of
    the two.

    Note:
        Exactly 2 devices are used, if more are given the are
        discarded, if less, the same device will be used twice.

    Note:
        The reference device weights will all zero on default. To set
        the reference device with a particular value one can select the
        device update index::

            analog_tile.set_hidden_update_index(1)
            analog_tile.set_weights(W)
            analog_tile.set_hidden_update_index(0) # set back to 0 for the following updates
    """

    bindings_class: ClassVar[Type] = devices.VectorResistiveDeviceParameter

    update_policy: VectorUnitCellUpdatePolicy = VectorUnitCellUpdatePolicy.SINGLE_FIXED
    """The update policy of which if the devices will be receiving the
    update of a mini-batch.

    Caution:
        This parameter should be kept to SINGLE_FIXED for this device.
    """

    first_update_idx: int = 0
    """Device that receives the update."""

    gamma_vec: List[float] = field(default_factory=lambda: [1., -1.],
                                   metadata={'hide_if': [1., -1.]})
    """Weighting of the unit cell devices to reduce to final weight.

    Note:
        While user-defined weighting can be given it is suggested to keep it to
        the default ``[1, -1]`` to implement the reference device subtraction.
    """

    def as_bindings(self) -> devices.VectorResistiveDeviceParameter:
        """Return a representation of this instance as a simulator bindings object."""
        vector_parameters = parameters_to_bindings(self)

        if not isinstance(self.unit_cell_devices, list):
            raise ConfigError('unit_cell_devices should be a list of devices')

        if len(self.unit_cell_devices) > 2:
            self.unit_cell_devices = self.unit_cell_devices[:2]
        elif len(self.unit_cell_devices) == 1:
            self.unit_cell_devices = [self.unit_cell_devices[0],
                                      deepcopy(self.unit_cell_devices[0])]
        elif len(self.unit_cell_devices) != 2:
            raise ConfigError('ReferenceUnitCell expects two unit_cell_devices')

        for param in self.unit_cell_devices:
            device_parameters = param.as_bindings()
            if not vector_parameters.append_parameter(device_parameters):
                raise ConfigError('Could not add unit cell device parameter')

        return vector_parameters


@dataclass
class OneSidedUnitCell(UnitCell):
    """Abstract device model takes an arbitrary device per crosspoint and
    implements an explicit plus-minus device pair with one sided update.

    One device will receive all positive updated and the other all
    negative updates. Since the devices will quickly saturate, the
    device implements a refresh strategy.

    With fixed frequency per update call (``refresh_every``, in units
    of single vector updates) a refresh is performed. During the
    refresh, each column will be read using a forward pass (parameters
    are specified with ``refresh_forward``) to read out the positive and
    negative device weights.

    Whether a weight needs refreshing is determined by the following
    criterion: The larger weight (normalized by the tile-wise fixed
    w_max setting) is tested against the upper threshold. If larger
    than the upper threshold, and the normalized lower weight is
    larger than the lower threshold, then a reset and rewriting will
    be performed.

    Note that this abstract device needs single devices that are
    derived from :class:`~PulsedDevice`. The reset properties (bias
    and cycle-to-cycle noise) can be thus adjusted (see
    :class:`~PulsedDevice`).

    The rewriting of the computed difference is only done onto one of
    the two devices using the update properties defined in
    ``refresh_update``.

    Note:
        This device will take only the first ``unit_cell_device`` to
        generate two devices. Both positive and negative device will
        thus have the same (reversed) parameters, e.g. the specified
        ``w_min``, will become the w_max of the negative device.
    """

    bindings_class: ClassVar[Type] = devices.OneSidedResistiveDeviceParameter

    refresh_every: int = 0
    """How often a refresh is performed (in units of the number of vector
    updates).

    Note:
        If a refresh is done, full reads of both positive and negative
        devices are performed. Additionally, if single devices deemed
        to be refreshed, an (open-loop) re-write is done (once per
        column). Thus, refresh might have considerable runtime
        impacts.
    """

    units_in_mbatch: bool = True
    """If set, the ``refresh_every`` counter is given in ``m_batch``
    which is the re-use factor. Smaller numbers are not possible.

    Caution:
        For CUDA devices, refresh is always done in  ``m_batch`` (ie
        the number of re-use per layer for a mini-batch). Smaller
        numbers will have no effect.
    """

    refresh_upper_thres: float = 0.75
    """Upper threshold for determining the refresh, see above."""

    refresh_lower_thres: float = 0.25
    """Lower threshold for determining the refresh, see above."""

    refresh_forward: IOParameters = field(
        default_factory=IOParameters)
    """Input-output parameters that define the read during a refresh event.

    :class:`~aihwkit.simulator.config.utils.AnalogTileInputOutputParameters`
    that define the read (forward) of an refresh event. For instance
    the amount of noise or whether refresh is done using a ADC/DAC
    etc.
    """

    refresh_update: UpdateParameters = field(default_factory=UpdateParameters)
    """Update parameters that define the type of update used for each refresh
    event.

    Update parameters
    :class:`~aihwkit.simulator.config.utils.AnalogTileUpdateParameters`
    that define the type of update used for each refresh event.
    """

    copy_inverted: bool = False
    """Whether the use the "down" update behavior of the first device for
    the negative updates instead of the positive half of the second
    device."""

    def as_bindings(self) -> devices.OneSidedResistiveDeviceParameter:
        """Return a representation of this instance as a simulator
        bindings object."""
        if not isinstance(self.unit_cell_devices, list):
            raise ConfigError('unit_cell_devices should be a list of devices')

        onesided_parameters = parameters_to_bindings(self)
        device_parameter0 = self.unit_cell_devices[0].as_bindings()

        if len(self.unit_cell_devices) == 0 or len(self.unit_cell_devices) > 2:
            raise ConfigError('Need 1 or 2 unit_cell_devices')

        if len(self.unit_cell_devices) == 1:
            device_parameter1 = device_parameter0
        else:
            device_parameter1 = self.unit_cell_devices[1].as_bindings()

        # need to be exactly 2 and same parameters
        if not onesided_parameters.append_parameter(device_parameter0):
            raise ConfigError('Could not add unit cell device parameter')

        if not onesided_parameters.append_parameter(device_parameter1):
            raise ConfigError('Could not add unit cell device parameter ' +
                              '(both devices need to be of the same type)')

        return onesided_parameters


@dataclass
class DifferenceUnitCell(OneSidedUnitCell):
    """Deprecated alias to ``OneSidedUnitCell``."""

    def __post__init__(self) -> None:
        warn('The DifferenceUnitCell class is deprecated. Please use '
             'OneSidedUnitCell instead.',
             DeprecationWarning)


@dataclass
class TransferCompound(UnitCell):
    r"""Abstract device model that takes 2 or more devices and
    implements a transfer-based learning rule.

    It uses a (partly) hidden weight (where the SGD update is
    accumulated), which then is transferred partly and occasionally to
    the visible weight. This can implement an analog friendly variant
    of stochastic gradient descent (Tiki-taka), as described in
    `Gokmen & Haensch (2020)`_.

    The hidden weight is always the first in the list of
    ``unit_cell_devices`` given, and the transfer is done from left to
    right. The first of the ``unit_cell_devices`` can have different
    HW specifications from the rest, but the others need to be of
    identical specs. In detail, when specifying the list of devices
    only the first two will actually be used and the rest discarded
    and instead replaced by the second device specification. In this
    manner, the *fast* crossbar (receiving the SGD updates) and the
    *slow* crossbar (receiving the occasional partial transfers from
    the fast) can have different specs, but all additional slow
    crossbars (receiving transfers from the left neighboring crossbar
    in the list of ``unit_cell_devices``) need to be of the same spec.

    The rate of transfer (e.g. learning rate and how often and how
    many columns/rows per transfer) and the type (ie. with ADC or without,
    with noise etc.) can be adjusted.

    Each transfer event that is triggered by counting the update
    cycles (in units of either mini-batch or single mat-vecs),
    ``n_reads_per_transfer`` columns/rows are read from the left device
    using the forward pass with transfer vectors as input and
    transferred to the right (taking the order of the
    ``unit_cell_devices`` list) using the outer-product update with
    the read-out vectors and the transfer vectors. Currently, transfer
    vectors are fixed to be one-hot vectors. The columns/rows to take are
    in sequential order and warped around at the edge of the
    crossbar. The learning rate and forward and update specs of the
    transfer can be user-defined.

    The weight that is seen in the forward and backward pass is
    governed by the :math:`\gamma` weightening setting.

    Note:
        Here the devices could be either transferred in analog
        (essentially within the unit cell) or on separate arrays (using
        the usual (non-ideal) forward pass and update steps. This can be
        set with ``transfer_forward`` and ``transfer_update``.

    .. _Gokmen & Haensch (2020): https://www.frontiersin.org/articles/10.3389/fnins.2020.00103/full
    """

    bindings_class: ClassVar[Type] = devices.TransferResistiveDeviceParameter

    gamma: float = 0.0
    r"""Weighting factor to compute the effective SGD weight from the hidden
    matrices.

    The default scheme is:

    .. math:: g^{n-1} W_0 + g^{n-2} W_1 + \ldots + g^0  W_{n-1}
    """

    gamma_vec: List[float] = field(default_factory=list,
                                   metadata={'hide_if': []})
    """User-defined weightening.

    User-defined weightening can be given as a list if weights in which case
    the default weightening scheme with ``gamma`` is not used.
    """

    transfer_every: float = 1.0
    """Transfers every :math:`n` mat-vec operations or :math:`n` batches.

    Transfers every :math:`n` mat-vec operations (rounded to multiples/ratios
    of ``m_batch`` for CUDA). If ``units_in_mbatch`` is set, then the units are
    in ``m_batch`` instead of mat-vecs, which is equal to the overall the
    weight re-use during a while mini-batch.

    Note:
        If ``transfer_every`` is 0.0 *no transfer* will be made.

    If not given explicitely with ``transfer_every_vec``, then the higher
    transfer cycles are geometrically scaled, the first is set to
    transfer_every. Each next transfer cycle is multiplied by ``x_size
    / n_reads_per_transfer``.
    """

    no_self_transfer: bool = True
    """Whether to set the transfer rate of the last device (which is applied to
    itself) to zero."""

    transfer_every_vec: List[float] = field(default_factory=list,
                                            metadata={'hide_if': []})
    """Transfer cycles lengths.

    A list of :math:`n` entries, to explicitly set the transfer cycles lengths.
    In this case, the above defaults are ignored.
    """

    units_in_mbatch: bool = True
    """Units for ``transfer_every``.

    If set, then the cycle length units of ``transfer_every`` are in
    ``m_batch`` instead of mat-vecs, which is equal to the overall of the
    weight re-use during a while mini-batch.
    """

    n_reads_per_transfer: int = 1
    """Number of consecutive reads to use during transfer events.

    How many consecutive columns or rows to read (from one tile) and write (to the next
    tile) every transfer event. For read, the input is a 1-hot vector. Once the
    final columns or row is reached, reading starts again from the first.
    """

    transfer_columns: bool = True
    """Whether to read and transfer columns or rows.

    If set, read is done with an additional forward pass
    determined by the ``transfer_forward`` settings. If not set, rows
    are transferred instead, that is, the read is done internally
    with a backward pass instead. However, the parameters defining the
    backward are still given by setting the ``transfer_forward`` field for
    convenience.
    """

    with_reset_prob: float = 0.0
    """Whether to apply reset of the columns that were transferred with a given
    probability.

    Note:
        Reset is only available in case of column reads
        (``transfer_columns==True``).
    """

    random_selection: bool = False
    """Whether to select a random starting column or row.

    Whether to select a random starting column or row for each
    transfer event and not take the next column or row that was
    previously not transferred as a starting column or row (the
    default).
    """

    fast_lr: float = 0.0
    """Whether to set the `fast` tile's learning rate.

    If set, then the SGD gradient update onto the first (fast) tile is
    set to this learning rate and is kept constant even when the SGD
    learning rate is scheduled. The SGD learning rate is then only
    used to scale the transfer LR (see ``scale_transfer_lr``).
    """

    transfer_lr: float = 1.0
    """Learning rate (LR) for the update step of the transfer event.

    Per default all learning rates are identical. If ``scale_transfer_lr`` is
    set, the transfer LR is scaled by current learning rate of the SGD.

    Note:
        LR is always a positive number, sign will be correctly
        applied internally.
    """

    transfer_lr_vec: List[float] = field(default_factory=list,
                                         metadata={'hide_if': []})
    """Transfer LR for each individual transfer in the device chain can be
    given."""

    scale_transfer_lr: bool = True
    """Whether to give the transfer_lr in relative units.

    ie. whether to scale the transfer LR with the current LR of the SGD.
    """

    transfer_forward: IOParameters = field(
        default_factory=IOParameters)
    """Input-output parameters that define the read of a transfer event.

    :class:`~aihwkit.simulator.config.utils.AnalogTileInputOutputParameters` that define the read
    (forward or backward) of an transfer event. For instance the amount of noise
    or whether transfer is done using a ADC/DAC etc.
    """

    transfer_update: UpdateParameters = field(
        default_factory=UpdateParameters)
    """Update parameters that define the type of update used for each transfer
    event.

    Update parameters :class:`~aihwkit.simulator.config.utils.AnalogTileUpdateParameters` that
    define the type of update used for each transfer event.
    """

    def as_bindings(self) -> devices.TransferResistiveDeviceParameter:
        """Return a representation of this instance as a simulator bindings object."""
        if not isinstance(self.unit_cell_devices, list):
            raise ConfigError('unit_cell_devices should be a list of devices')

        n_devices = len(self.unit_cell_devices)

        transfer_parameters = parameters_to_bindings(self)

        param_fast = self.unit_cell_devices[0].as_bindings()
        param_slow = self.unit_cell_devices[1].as_bindings()

        if not transfer_parameters.append_parameter(param_fast):
            raise ConfigError('Could not add unit cell device parameter')

        for _ in range(n_devices - 1):
            if not transfer_parameters.append_parameter(param_slow):
                raise ConfigError('Could not add unit cell device parameter')

        return transfer_parameters


@dataclass
class BufferedTransferCompound(TransferCompound):
    r"""Abstract device model that takes 2 or more devices and
    implements a buffered transfer-based learning rule.

    Different to :class:`TransferCompound`, however,  readout is done
    first onto a digital buffer (in floating point precision), from
    which then the second analog matrix is updated. This second step is
    very similar to the analog update in :class:`MixedPrecisionCompound`.

    Note, however, that in contrast to :class:`MixedPrecisionCompound`
    the rank-update is still done in analog with parallel update using
    pulse trains.

    The buffer is assumed to be in floating point precision and
    only one row/column at a time needs to be processed in one update cycle,
    thus greatly reducing on-chip memory requirements.

    For details, see `Gokmen (2021)`_.

    .. _Gokmen (2021): https://www.frontiersin.org/articles/10.3389/frai.2021.699148/full
    """

    bindings_class: ClassVar[Type] = devices.BufferedTransferResistiveDeviceParameter

    thres_scale: float = 1.0
    """Threshold scale for buffer to determine whether to transfer to next
    device. Will be multiplied by the device granularity to get the
    threshold.
    """

    step: float = 1.0
    """Value to fill the ``d`` vector for the update if buffered value is
    above threshold.
    """

    momentum: float = 0.0
    """Momentum of the buffer.

    After transfer, this momentum fraction stays on the buffer instead
    of subtracting all of what was transferred.
    """

###############################################################################
# Specific compound-devices with digital rank update
###############################################################################


@dataclass
class DigitalRankUpdateCell(_PrintableMixin):
    """Parameters that modify the behavior of the digital rank update cell.

    This is the base class for devices that compute the rank update in
    digital and then (occasionally) transfer the information to the
    (analog) crossbar array that is used during forward and backward.
    """

    bindings_class: ClassVar[Type] = devices.AbstractResistiveDeviceParameter

    device: Union[PulsedDevice,
                  OneSidedUnitCell,
                  VectorUnitCell,
                  ReferenceUnitCell] = field(default_factory=ConstantStepDevice)
    """(Analog) device that are used for forward and backward."""

    def as_bindings(self) -> devices.AbstractResistiveDeviceParameter:
        """Return a representation of this instance as a simulator bindings object."""
        raise NotImplementedError

    def requires_diffusion(self) -> bool:
        """Return whether device has diffusion enabled."""
        return self.device.requires_diffusion()

    def requires_decay(self) -> bool:
        """Return whether device has decay enabled."""
        return self.device.requires_decay()


@dataclass
class MixedPrecisionCompound(DigitalRankUpdateCell):
    r"""Abstract device model that takes 1 (analog) device and
    implements a transfer-based learning rule, where the outer product
    is computed in digital.

    Here, the outer product of the activations and error is done on a
    full-precision floating-point :math:`\chi` matrix. Then, with a
    threshold given by the ``granularity``, pulses will be applied to
    transfer the information row-by-row to the analog matrix.

    For details, see `Nandakumar et al. Front. in Neurosci. (2020)`_.

    Note:
        This version of update is different from a parallel update in
        analog other devices are implementing with stochastic pulsing,
        as here :math:`{\cal O}(n^2)` digital computations are needed
        to compute the outer product (rank update). This need for
        digital compute in potentially high precision might result in
        inferior run time and power estimates in real-world
        applications, although sparse integer products can potentially
        be employed to speed up to improve run time estimates. For
        details, see discussion in `Nandakumar et al. Front. in
        Neurosci. (2020)`_.

    .. _`Nandakumar et al. Front. in Neurosci. (2020)`: https://doi.org/10.3389/fnins.2020.00406
    """

    bindings_class: ClassVar[Type] = devices.MixedPrecResistiveDeviceParameter

    transfer_every: int = 1
    """Transfers every :math:`n` mat-vec operations.
    Transfers every :math:`n` mat-vec operations (rounded to multiples/ratios
    of ``m_batch``).

    Standard setting is 1.0 for mixed precision, but it could potentially be
    reduced to get better run time estimates.
    """

    n_rows_per_transfer: int = -1
    r"""How many consecutive rows to write to the tile from the :math:`\chi`
    matrix.

    ``-1`` means full matrix read each transfer event.
    """

    random_row: bool = False
    """Whether to select a random starting row.

    Whether to select a random starting row for each transfer event and not
    take the next row that was previously not transferred as a starting row
    (the default).
    """

    granularity: float = 0.0
    r"""Granularity of the device.

    Granularity :math:`\varepsilon` of the device that is used to
    calculate the number of pulses transferred from :math:`\chi` to
    analog.

    If 0, it will take granularity from the analog device used.
    """

    transfer_lr: float = 1.0
    r"""Scale of the transfer to analog .

    The update onto the analog tile will be proportional to
    :math:`\langle\chi/\varepsilon\rangle\varepsilon\lambda_\text{tr}`,
    where :math:`\lambda_\text{tr}` is given by ``transfer_lr`` and
    :math:`\varepsilon` is the granularity.
    """

    n_x_bins: int = 0
    """The number of bins to discretize (symmetrically around zero) the
    activation before computing the outer product.

    Dynamic quantization is used by computing the absolute max value of each
    input. Quantization can be turned off by setting this to 0.
    """

    n_d_bins: int = 0
    """The number of bins to discretize (symmetrically around zero) the
    error before computing the outer product.

    Dynamic quantization is used by computing the absolute max value of each
    error vector. Quantization can be turned off by setting this to 0.
    """

    def as_bindings(self) -> devices.MixedPrecResistiveDeviceParameter:
        """Return a representation of this instance as a simulator bindings object."""
        mixed_prec_parameter = parameters_to_bindings(self)
        param_device = self.device.as_bindings()

        if not mixed_prec_parameter.set_device_parameter(param_device):
            raise ConfigError('Could not add device parameter')

        return mixed_prec_parameter

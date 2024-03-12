# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Configuration for NVM devices for Analog (Resistive Device) tiles."""

# pylint: disable=too-many-instance-attributes, too-many-lines

from dataclasses import dataclass, field
from typing import ClassVar, List, Type, Optional, Union, Any
from numpy import exp

from aihwkit.simulator.parameters.enums import RPUDataType
from aihwkit.simulator.parameters.helpers import _PrintableMixin, parameters_to_bindings
from aihwkit.simulator.parameters.inference import DriftParameter, SimpleDriftParameter

# legacy
from aihwkit.simulator.configs.compounds import (  # pylint: disable=unused-import
    VectorUnitCell,
    ReferenceUnitCell,
    OneSidedUnitCell,
    DifferenceUnitCell,
    TransferCompound,
    BufferedTransferCompound,
    MixedPrecisionCompound,
    DynamicTransferCompound,
    ChoppedTransferCompound,
)


@dataclass
class FloatingPointDevice(_PrintableMixin):
    """Floating point reference.

    Implements ideal devices forward/backward/update behavior.
    """

    bindings_class: ClassVar[Optional[Union[Type, str]]] = "FloatingPointTileParameter"
    bindings_module: ClassVar[str] = "devices"

    diffusion: float = 0.0
    """Standard deviation of diffusion process."""

    lifetime: float = 0.0
    r"""One over `decay_rate`, ie :math:`1/r_\text{decay}`."""

    drift: SimpleDriftParameter = field(default_factory=SimpleDriftParameter)
    """Parameter governing a power-law drift."""

    def as_bindings(self, data_type: RPUDataType) -> Any:
        """Return a representation of this instance as a simulator bindings object."""
        return parameters_to_bindings(self, data_type)

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
    :class:`~aihwkit.similar.parameters.inference.DriftParameter`.

    Important:
        Similar to reset, drift is *not* applied automatically each
        mini-batch but requires an explicit call to
        :meth:`~aihwkit.simulator.tiles.base.Base.drift_weights` each
        time the drift should be applied.

    """

    bindings_class: ClassVar[Optional[Union[Type, str]]] = "PulsedResistiveDeviceParameter"
    bindings_module: ClassVar[str] = "devices"

    construction_seed: int = 0
    """If not equal 0, will set a unique seed for hidden parameters during
    construction."""

    corrupt_devices_prob: float = 0.0
    """Probability for devices to be corrupt (weights fixed to random value
    with hard bounds, that is min and max bounds are set to equal)."""

    corrupt_devices_range: float = 0.1
    """Range around zero for establishing corrupt devices."""

    diffusion: float = 0.0
    """Standard deviation of diffusion process."""

    diffusion_dtod: float = 0.0
    """Device-to device variation of diffusion rate in relative units."""

    drift: DriftParameter = field(
        default_factory=DriftParameter, metadata={"hide_if": DriftParameter()}
    )
    """Parameter governing a power-law drift."""

    dw_min: float = 0.001
    """Mean of the minimal update step sizes across devices and directions."""

    dw_min_dtod: float = 0.3
    """Device-to-device std deviation of ``dw_min`` (in relative units to
    ``dw_min``)."""

    dw_min_dtod_log_normal: bool = False
    """Device-to-device std deviation ``dw_min_dtod`` given using a
    log-normal instead of normal distribution."""

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

    count_pulses: bool = False
    """Whether to count the positive and negative pulses that were applied.

    Only for GPU devices currently implemented. Some runtime penalty expected.

    Pulses can be obtained by ``analog_tile.tile.get_pulse_counters()``
    """

    def as_bindings(self, data_type: RPUDataType) -> Any:
        """Return a representation of this instance as a simulator bindings object."""
        return parameters_to_bindings(self, data_type)

    def requires_diffusion(self) -> bool:
        """Return whether device has diffusion enabled."""
        return self.diffusion > 0.0

    def requires_decay(self) -> bool:
        """Return whether device has decay enabled."""
        return self.lifetime > 0.0


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

    bindings_class: ClassVar[Optional[Union[Type, str]]] = "IdealResistiveDeviceParameter"
    bindings_module: ClassVar[str] = "devices"

    construction_seed: int = 0
    """If not ``0``, set a unique seed for hidden parameters during
    construction."""

    diffusion: float = 0.0
    """Standard deviation of diffusion process."""

    lifetime: float = 0.0
    r"""One over `decay_rate`, ie :math:`1/r_\text{decay}`."""

    reset_std: float = 0.01
    """Standard deviation around zero mean in case reset is called."""

    def as_bindings(self, data_type: RPUDataType) -> Any:
        """Return a representation of this instance as a simulator bindings object."""
        return parameters_to_bindings(self, data_type)

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

    bindings_class: ClassVar[Optional[Union[Type, str]]] = "ConstantStepResistiveDeviceParameter"


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
        \gamma_{ij}^+ &\=& - |\gamma^+ + \gamma_\text{d-to-d}^+ \xi|/b^\text{max}_{ij}\\
        \gamma_{ij}^- &\=& - |\gamma^- + \gamma_\text{d-to-d}^- \xi|/b^\text{min}_{ij}
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

    bindings_class: ClassVar[Optional[Union[Type, str]]] = "LinearStepResistiveDeviceParameter"

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

        \gamma_{ij}^+ &\=& - |\gamma^+ + \gamma_\text{d-to-d}^+ \xi|/b^\text{max}

        \gamma_{ij}^- &\=& - |\gamma^- + \gamma_\text{d-to-d}^- \xi|/b^\text{min}

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

    apply_write_noise_on_set: bool = True
    r"""Whether setting the weights with ``set_weights`` will add
    write noise to the apparent weight state or not.

    If ``False`` the persistent weight state will be equal to the
    apparent state initially.
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

    bindings_class: ClassVar[Optional[Union[Type, str]]] = "SoftBoundsResistiveDeviceParameter"

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

    apply_write_noise_on_set: bool = True
    r"""Whether setting the weights with ``set_weights`` will add
    write noise to the apparent weight state or not.

    If ``False`` the persistent weight state will be equal to the
    apparent state initially.
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

    alpha: float = 0.001 / 2
    r"""The slope of the soft bounds model :math:`dw \propto \alpha w` for both
    up and down direction."""

    range_min: float = -1.0
    """Setting of the weight when starting the :math:`P_max` up pulse
    experiment."""

    range_max: float = 1.0
    """Value of the weight for :math:`P_max` number of up pulses."""

    #  these values will be set from the above, so we hide it.
    w_min: float = field(default_factory=lambda: None, metadata={"hide_if": None})  # type: ignore
    w_max: float = field(default_factory=lambda: None, metadata={"hide_if": None})  # type: ignore
    dw_min: float = field(default_factory=lambda: None, metadata={"hide_if": None})  # type: ignore
    up_down: float = field(default_factory=lambda: None, metadata={"hide_if": None})  # type: ignore

    def as_bindings(self, data_type: RPUDataType) -> Any:
        """Return a representation of this instance as a simulator bindings object."""
        params = SoftBoundsDevice()
        for key, value in self.__dict__.items():
            if key not in ["range_min", "range_max", "alpha", "p_max"]:
                setattr(params, key, value)

        b_factor = (self.range_max - self.range_min) / (1 - exp(-self.p_max * self.alpha))
        params.w_min = self.range_min
        params.w_max = self.range_min + b_factor
        params.dw_min = b_factor * self.alpha
        params.up_down = 1 + 2 * self.range_min / b_factor

        return parameters_to_bindings(params, data_type)


@dataclass
class SoftBoundsReferenceDevice(PulsedDevice):
    r"""Pulsed update behavioral model: soft bounds with reference device.

    Pulsed update behavioral model, where the update step response size
    of the material is linearly dependent.

    In particular, the update behavior is

    .. math::
        \delta W_+ = \alpha_{+}(1 - \frac{w}{\beta_{max}}) (1 + \sigma \xi)

        \delta W_- = \alpha_{-}(1 - \frac{w}{\beta_{min}}) (1 + \sigma \xi)

    Where the same device-to-device variation can be given as for the
    ``PulsedDevice``. In addition, a device-to-device variation can be
    directly given on the slope.  The :math:`\alpha_{+}` and :math:`\alpha_{-}` are
    the scaling factors that determine the magnitude of positive and negative
    weight updates.

    Moreover, a fixed reference conductance can be subtracted from the
    resulting weight, which implemented a differential read of :math:`w - r`.

    """

    bindings_class: ClassVar[
        Optional[Union[Type, str]]
    ] = "SoftBoundsReferenceResistiveDeviceParameter"

    mult_noise: bool = False
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

    apply_write_noise_on_set: bool = True
    r"""Whether setting the weights with ``set_weights`` will add
    write noise to the apparent weight state or not.

    If ``False`` the persistent weight state will be equal to the
    apparent state initially.
    """

    slope_up_dtod: float = 0.0
    r"""Device-to-device variation on the up-pulse slope.

    Note:

        Since the up slope is proportional to
        :math:`\propto\frac{1}{b_\text{max}}` the device-to-device variation
        of the weight max ``w_max_dtod`` will also introduce a slope
        variation. Turn that off, if the variation should be only on
        the slope directly.

    """

    slope_down_dtod: float = 0.0
    r"""Device-to-device variation on the down-pulse slope.

    Note:

        Since the up slope is proportional to
        :math:`\propto\frac{1}{b_\text{min}}` the device-to-device variation
        of the weight max ``w_min_dtod`` will also introduce a slope
        variation. Turn that off, if the variation should be only on
        the slope directly.
    """

    reference_mean: float = 0.0
    """Added to all devices of the reference :math:`r`.
    """

    reference_std: float = 0.0
    """Normal distributed device-to-device variation added to the reference :math:`r`.
    """

    subtract_symmetry_point: bool = False
    r"""Whether to add the computed symmetry point of the devices onto the
    reference :math:`r`.

    The symmetry point is given by:

    .. math::
        w_* = \frac{\alpha_{+} - \alpha_{-}}{\frac{\alpha_{+}}{b_\text{max}}
                    - \frac{\alpha_{-}}{b_\text{min}}}
    """


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

    bindings_class: ClassVar[Optional[Union[Type, str]]] = "ExpStepResistiveDeviceParameter"

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

    apply_write_noise_on_set: bool = True
    r"""Whether setting the weights with ``set_weights`` will add
    write noise to the apparent weight state or not.

    If ``False`` the persistent weight state will be equal to the
    apparent state initially.
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

    bindings_class: ClassVar[Optional[Union[Type, str]]] = "PowStepResistiveDeviceParameter"

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

    apply_write_noise_on_set: bool = True
    r"""Whether setting the weights with ``set_weights`` will add
    write noise to the apparent weight state or not.

    If ``False`` the persistent weight state will be equal to the
    apparent state initially.
    """


@dataclass
class PowStepReferenceDevice(PulsedDevice):
    r"""Pulsed update behavioral model: power-dependent step.

    Pulsed update behavioral model, where the update step response
    size of the material has a power-dependent with resistance. This
    device model implements (a shifted from of) the `Fusi & Abott
    (2007)`_ synapse model (see also `Frascaroli et al. (2108)`_).

    This device model is identical to
    :class:`~aihwkit.simulator.configs.devices.PowStepDevice`,
    however, here it does not implement any write noise functionality
    but instead has an additional option to subtract a reference
    conductance from the weights which is statically set during
    initialization.

    The model based on :class:`~PulsedDevice` and thus shares most
    parameters and functionality. However, it implements new ``update
    once`` function, where the update step size depends in the
    following way. If we set :math:`\omega_{ij} =
    \frac{b_{ij}^\text{max} - w_{ij}}{b_{ij}^\text{max} -
    b_{ij}^\text{min}}` the relative distance of the current weight to
    the upper bound, then the update per pulse is for the upwards
    direction:

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

    Note:

        In contrast to
        :class:`~aihwkit.simulator.configs.devices.PowStepDevice`,
        write noise is not supported.

    ..  _Fusi & Abott (2007): https://www.nature.com/articles/nn1859
    ..  _Frascaroli et al. (2108): https://www.nature.com/articles/s41598-018-25376-x

    """

    bindings_class: ClassVar[
        Optional[Union[Type, str]]
    ] = "PowStepReferenceResistiveDeviceParameter"

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

    subtract_symmetry_point: bool = False
    r"""Whether store the symmetry point of each device onto the reference device.

    The symmetry point is only numerically estimated, since an
    analytically for is not available due to the power step model.
    """

    n_estimation_steps: int = -1
    """The number of times to run an (noise-free) up / down pulse
    combination for the numerical estimation of the symmetry points.

    In case of a non-positive number, the number of estimations steps
    is set 10 times the expected number of states of each device
    (maxed out at 10000).
    """

    reference_mean: float = 0.0
    """Added to all devices of the reference :math:`r`.
    """

    reference_std: float = 0.0
    """Normal distributed device-to-device variation added to the reference :math:`r`.
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

    bindings_class: ClassVar[Optional[Union[Type, str]]] = "PiecewiseStepResistiveDeviceParameter"

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

    apply_write_noise_on_set: bool = True
    r"""Whether setting the weights with ``set_weights`` will add
    write noise to the apparent weight state or not.

    If ``False`` the persistent weight state will be equal to the
    apparent state initially.
    """

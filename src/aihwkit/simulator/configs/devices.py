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

"""Configuration for Analog (Resistive Device) tiles."""

# pylint: disable=too-many-instance-attributes

from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, List, Type

from aihwkit.exceptions import ConfigError
from aihwkit.simulator.configs.helpers import (
    _PrintableMixin, parameters_to_bindings
)
from aihwkit.simulator.configs.utils import (
    IOParameters, UpdateParameters, VectorUnitCellUpdatePolicy
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


    **Decay**:

    .. math:: w_{ij} \leftarrow w_{ij}\,(1-\alpha_\text{decay}\delta_{ij})

    Weight decay is generally off and has to be activated explicitly
    by using :meth:`decay` on an analog tile. Note that the device
    ``decay_lifetime`` parameters (1 over decay rates
    :math:`\delta_{ij}`) are analog tile specific and are thus set and
    fixed during RPU initialization. :math:`\alpha_\text{decay}` is a
    scaling factor that can be given during run-time.


    **Diffusion**:

    Similar to the decay, diffusion is only activated by inserting a specific
    operator. However, the parameters of the diffusion
    process are set during RPU initialization and are fixed for the
    remainder.

    .. math:: w_{ij} \leftarrow w_{ij} + \rho_{ij} \, \xi;

    where :math:`xi` is a standard Gaussian variable and :math:`\rho_{ij}` the
    diffusion rate for a cross-point `ij`

    Note:
       If diffusion happens to move the weight beyond the hard bounds of the
       weight it is ensured to be clipped appropriately.
    """

    bindings_class: ClassVar[Type] = devices.PulsedResistiveDeviceParameter

    construction_seed: int = 0
    """If not equal 0, will set a unique seed for hidden parameters
    during construction"""

    corrupt_devices_prob: float = 0.0
    """Probability for devices to be corrupt (weights fixed to random value
    with hard bounds, that is min and max bounds are set to equal)."""

    corrupt_devices_range: int = 1000
    """Range around zero for establishing corrupt devices."""

    diffusion: float = 0.0
    """Standard deviation of diffusion process."""

    diffusion_dtod: float = 0.0
    """Device-to device variation of diffusion rate in relative units."""

    dw_min: float = 0.001
    """Mean of the minimal update step sizes across devices and directions."""

    dw_min_dtod: float = 0.3
    """Device-to-device std deviation of dw_min (in relative units to ``dw_min``)."""

    dw_min_std: float = 0.3
    r"""Cycle-to-cycle variation size of the update step (related to
    :math:`\sigma_\text{c-to-c}` above) in relative units to ``dw_min``.

    Note:
        Many spread (device-to-device variation) parameters are
        given in relative units. For instance e.g. a setting of
        ``dw_min_std`` of 0.1 would mean 10% spread around the
        mean and thus a resulting standard deviation
        (:math:`\sigma_\text{c-to-c}`) of ``dw_min`` *
        ``dw_min_std``.
    """

    enforce_consistency: bool = True
    """Whether to enforce during initialization that max weight bounds cannot
    be smaller than min weight bounds, and up direction step size is positive
    and down negative. Switches the opposite values if encountered during
    init."""

    lifetime: float = 0.0
    r"""One over `decay_rate`, ie :math:`1/r_\text{decay}`."""

    lifetime_dtod: float = 0.0
    """Device-to-device variation in the decay rate (in relative units)."""

    perfect_bias: bool = False
    """No up-down differences and device-to-device variability in the bounds
    for the devices in the bias row."""

    reset: float = 0.01
    """The reset values and spread per cross-point ``ij`` when using reset functionality
    of the device."""

    reset_dtod: float = 0.0
    """See ``reset``."""

    reset_std: float = 0.01
    """See ``reset``."""

    up_down: float = 0.0
    r"""Up and down direction step sizes can be systematically different and also
    vary across devices.
    :math:`\Delta w_{ij}^d` is set during RPU initialization (for each cross-point `ij`):

    .. math::

        \Delta w_{ij}^d = d\; \Delta w_\text{min}\, \left(
        1 + d \beta_{ij} + \sigma_\text{d-to-d}\xi\right)

    where \xi is again a standard Gaussian. :math:`\beta_{ij}`
    is the directional up `versus` down bias.  At initialization
    ``up_down_dtod`` and ``up_down`` defines this bias term:

    .. math::

        \beta_{ij} = \beta_\text{up-down} + \xi
        \sigma_\text{up-down-dtod}

    where \xi is again a standard Gaussian number and
    :math:`\beta_\text{up-down}` corresponds to
    ``up_down``. Note that ``up_down_dtod`` is again given in
    relative units to ``dw_min``.
    """

    up_down_dtod: float = 0.01
    """See ``up_down``."""

    w_max: float = 0.6
    """See ``w_min``."""

    w_max_dtod: float = 0.3
    """See ``w_min_dtod``."""

    w_min: float = -0.6
    """Mean of hard bounds across device cross-point `ij`. The parameters
    ``w_min`` and ``w_max`` are used to set the min/max bounds independently.

    Note:
        For this abstract device, we assume that weights can have
        positive and negative values and are symmetrically around
        zero. In physical circuit terms, this might be implemented
        as a difference of two resistive elements.
    """

    w_min_dtod: float = 0.3
    """Device-to-device variation of the hard bounds, of min and max value,
    respectively. All are given in relative units to ``w_min``, or ``w_max``,
    respectively."""

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
    """If not equal 0, will set a unique seed for hidden parameters
    during construction"""

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
    :class:`~aihwkit.simulator.parameters.ConstantStepResistiveDeviceParameters`.
    """

    bindings_class: ClassVar[Type] = devices.ConstantStepResistiveDeviceParameter


@dataclass
class LinearStepDevice(PulsedDevice):
    r"""Pulsed update behavioral model: linear step.

    Pulsed update behavioral model, where the update step response
    size of the material is linearly dependent with resistance (up to
    hard bounds).

    This model is very similar to :class:`~ConstantStepResistiveDevice` and thus
    shares all parameters and functionality. In addition, it only
    implements a more general `update once` function, where the update
    step size can depend linearly on the weight itself.

    For each coincidence the weights is updated once. Here, the
    positive (negative) update step size decreases linearly in the
    following manner (compare to the `update once` for
    :class:`~ConstantStepResistiveDevice`):

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
    respectively (see description for :class:`~ConstantStepResistiveDevice`).

    Note:
       If :math:`\gamma=1` and :math:`\gamma_\text{d-to-d}=0` this
       update implements `soft bounds`, since the updates step becomes
       equal to :math:`1/b`.

    Note:
       If :math:`\gamma=0` and :math:`\gamma_\text{d-to-d}=0` and
       additive noise, this update is identical to
       :class:`~ConstantStepResistiveDevice`.
    """

    bindings_class: ClassVar[Type] = devices.LinearStepResistiveDeviceParameter

    gamma_up: float = 0.0
    r"""The value of :math:`\gamma^+`.

    Intuitively, a value of 0.1 means that the update step size in up
    direction at the weight bounds is 10% decreased relative to that
    origin :math:`w=0`.

    Note:
       In principle one could fix :math:`\gamma=\gamma^-=\gamma^+` since
       up/down variation can be given by ``up_down_dtod``, see
       :class:`~ConstantStepResistiveDevice`.

    Note:
       The hard-bounds are still observed, so that the weight cannot
       grow beyond its bounds.
    """

    gamma_down: float = 0.0
    r"""The value of :math:`\gamma^-`.
    """

    gamma_up_dtod: float = 0.05
    r"""Device-to-device variation for :math:`\gamma^+`, i.e. the
    value of :math:`\gamma_\text{d-to-d}^+`.
    """

    gamma_down_dtod: float = 0.05
    r"""Device-to-device variation for :math:`\gamma^-`, i.e. the
    value of :math:`\gamma_\text{d-to-d}^-`.
    """

    allow_increasing: bool = False
    """Whether to allow the situation where update sizes increase
    towards the bound instead of saturating (and thus becoming smaller)
    """

    mean_bound_reference: bool = True
    r"""Whether to use instead of the above:

    .. math::

        \gamma_{ij}^+ &=& - |\gamma^+ + \gamma_\text{d-to-d}^+ \xi|/b^\text{max}

        \gamma_{ij}^- &=& - |\gamma^- + \gamma_\text{d-to-d}^- \xi|/b^\text{min}

    where :math:`b^\text{max}` and :math:`b^\text{max}` are the
    values given by ``w_max`` and ``w_min``, see
    :class:`~ConstantStepResistiveDevice`.
    """

    mult_noise: bool = True
    """Whether to use multiplicative noise instead of additive
    cycle-to-cycle noise"""


@dataclass
class SoftBoundsDevice(PulsedDevice):
    r"""Pulsed update behavioral model: soft bounds.

    Pulsed update behavioral model, where the update step response size
    of the material is linearly dependent and it goes to zero at the
    bound.

    This model is based on :class:`~LinearStepResistiveDevice` with
    parameters set to model soft bounds.
    """

    bindings_class: ClassVar[Type] = devices.SoftBoundsResistiveDeviceParameter

    mult_noise: bool = True
    """Whether to use multiplicative noise instead of additive
    cycle-to-cycle noise"""


@dataclass
class ExpStepDevice(PulsedDevice):
    r"""Exponential update step or CMOS-like update behavior.

    This model is derived from ``PulsedResistiveDevice`` and uses all its
    parameters. ``ExpStepResistiveDevice`` only implements a new 'update once'
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
    :class:`~ConstantStepResistiveDevice` for details.

    All additional parameter (:math:`a_\text{es}`,
    :math:`b_\text{es}`, :math:`\gamma^{(d)}`, :math:`A^{(d)}` ) are
    tile-wise fitting parameters (ie. no device-to-device variation in
    these parameters).  Note that the other parameter involved can be
    still defined with device-to-device variation and (additional)
    up-down bias (see :class:`~ConstantStepResistiveDevice`).
    """
    # pylint: disable=invalid-name

    bindings_class: ClassVar[Type] = devices.ExpStepResistiveDeviceParameter

    A_up: float = 0.00081
    """Factor ``A`` for the up direction"""

    A_down: float = 0.36833
    """Factor ``A`` for the down direction"""

    gamma_up: float = 12.44625
    """Exponent for the up direction."""

    gamma_down: float = 12.78785
    """Exponent for the down direction."""

    a: float = 0.244
    """Global slope parameter"""

    b: float = 0.2425
    """Global offset parameter"""


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
    """The update policy of which if the devices will be receiving the
    update of a mini-batch."""

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
            raise ConfigError("unit_cell_devices should be a list of devices")

        for param in self.unit_cell_devices:
            device_parameters = param.as_bindings()
            if not vector_parameters.append_parameter(device_parameters):
                raise ConfigError("Could not add unit cell device parameter")

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
        While user-defined weighting can be given it is suggested to
        keep it to the default ``[1, -1]`` to implement the reference
        device subtraction.
    """

    def as_bindings(self) -> devices.VectorResistiveDeviceParameter:
        """Return a representation of this instance as a simulator bindings object."""
        vector_parameters = parameters_to_bindings(self)

        if not isinstance(self.unit_cell_devices, list):
            raise ConfigError("unit_cell_devices should be a list of devices")

        if len(self.unit_cell_devices) > 2:
            self.unit_cell_devices = self.unit_cell_devices[:2]
        elif len(self.unit_cell_devices) == 1:
            self.unit_cell_devices = [self.unit_cell_devices[0],
                                      deepcopy(self.unit_cell_devices[0])]
        elif len(self.unit_cell_devices) != 2:
            raise ConfigError("ReferenceUnitCell expects two unit_cell_devices")

        for param in self.unit_cell_devices:
            device_parameters = param.as_bindings()
            if not vector_parameters.append_parameter(device_parameters):
                raise ConfigError("Could not add unit cell device parameter")

        return vector_parameters


@dataclass
class DifferenceUnitCell(UnitCell):
    """Abstract device model takes an arbitrary device per crosspoint and
    implements an explicit plus-minus device pair.

    A plus minus pair is implemented by using only one-sided updated
    of the given devices. Note that reset might need to be called
    otherwise the one-sided device quickly saturates during learning.

    The output current is the difference of both devices.

    Meta parameter setting of the pairs are assumed to be identical
    (however, device-to-device variation is still present).

    Caution:
        Reset needs to be added `manually` by calling the
        reset_columns method of a tile.
    """

    bindings_class: ClassVar[Type] = devices.DifferenceResistiveDeviceParameter

    def as_bindings(self) -> devices.DifferenceResistiveDeviceParameter:
        """Return a representation of this instance as a simulator bindings object."""

        if not isinstance(self.unit_cell_devices, list):
            raise ConfigError("unit_cell_devices should be a list of devices")

        difference_parameters = parameters_to_bindings(self)
        device_parameters = self.unit_cell_devices[0].as_bindings()

        # need to be exactly 2 and same parameters
        if not difference_parameters.append_parameter(device_parameters):
            raise ConfigError("Could not add unit cell device parameter")

        if not difference_parameters.append_parameter(device_parameters):
            raise ConfigError("Could not add unit cell device parameter")

        return difference_parameters


@dataclass
class TransferCompound(UnitCell):
    r"""Abstract device model that takes 2 or more devices and
    implements a 'transfer' based learning rule.

    It uses a (partly) hidden weight (where the SGD update is
    accumulated), which then is transferred partly and occasionally to
    the visible weight.

    The rate of transfer (e.g. learning rate and how often and how
    many columns per transfer) and the type (ie. with ADC or without,
    with noise etc.) can be adjusted.

    The weight that is seen in the forward and backward pass is
    governed by the :math:`\gamma` weightening setting.

    In principle, a deeper chain of transferred weights can be setup,
    however, only the device parameters of the first versus the others
    can be different. However, all devices need to be specified in the
    list.

    Note:
        Here the devices could be either transferred in analog
        (essentially within the unit cell) or on separate arrays (using
        the usual (non-ideal) forward pass and update steps. This can be
        set with ``transfer_forward`` and ``transfer_update``.
    """

    bindings_class: ClassVar[Type] = devices.TransferResistiveDeviceParameter

    gamma: float = 0.0
    """
    Weightening factor g**(n-1) W[0] + g**(n-2) W[1] + .. + g**0 W[n-1]
    """

    gamma_vec: List[float] = field(default_factory=list,
                                   metadata={'hide_if': []})
    """
    User-defined weightening can be given as a list if weights in
    which case the default weightening scheme with ``gamma`` is not
    used.
    """

    transfer_every: float = 0.0
    """Transfers every :math:`n` mat-vec operations (rounded to
    multiples/ratios of m_batch for CUDA). If ``units_in_mbatch`` is
    set, then the units are in ``m_batch`` instead of mat-vecs, which
    is equal to the overall the weight re-use during a while
    mini-batch.

    If 0 it is set to ``x_size / n_cols_per_transfer``.

    The higher transfer cycles are geometrically scaled, the first is
    set to transfer_every. Each next transfer cycle is multiplied by
    by ``x_size / n_cols_per_transfer``.
    """

    no_self_transfer: bool = True
    """Whether to set the transfer rate of the last device (which is
    applied to itself) to zero.
    """

    transfer_every_vec: List[float] = field(default_factory=list,
                                            metadata={'hide_if': []})
    """A list of :math:`n` entries, to explicitly set the transfer
    cycles lengths. In this case, the above defaults are ignored.
    """

    units_in_mbatch: bool = True
    """If set, then the cycle length units of ``transfer_every`` are
    in ``m_batch`` instead of mat-vecs, which is equal to the overall
    the weight re-use during a while mini-batch.
    """

    n_cols_per_transfer: int = 1
    """How many consecutive columns to read (from one tile) and write
    (to the next tile) every transfer event. For read, the input is a
    1-hot vector. Once the final column is reached, reading starts
    again from the first.
    """

    with_reset_prob: float = 0.0
    """Whether to apply reset of the columns that were transferred
    with a given probability."""

    random_column: bool = False
    """Whether to select a random starting column for each transfer
    event and not take the next column that was previously not
    transferred as a starting column (the default).
    """

    transfer_lr: float = 1.0
    """Learning rate (LR) for the update step of the transfer
    event. Per default all learning rates are identical. If
    ``scale_transfer_lr`` is set, the transfer LR is scaled by current
    learning rate of the SGD.

    Note:
        LR is always a positive number, sign will be correctly
        applied internally.
    """

    transfer_lr_vec: List[float] = field(default_factory=list,
                                         metadata={'hide_if': []})
    """Transfer LR for each individual transfer in the device chain
    can be given.
    """

    scale_transfer_lr: bool = True
    """Whether to give the transfer_lr in relative units, ie whether
    to scale the transfer LR with the current LR of the SGD.
    """

    transfer_forward: IOParameters = field(
        default_factory=IOParameters)
    """Input-output parameters
    :class:`~AnalogTileInputOutputParameters` that define the read
    (forward) of an transfer event. For instance the amount of noise
    or whether transfer is done using a ADC/DAC etc."""

    transfer_update: UpdateParameters = field(
        default_factory=UpdateParameters)
    """Update parameters :class:`~AnalogTileUpdateParameters` that
    define the type of update used for each transfer event.
    """

    def as_bindings(self) -> devices.TransferResistiveDeviceParameter:
        """Return a representation of this instance as a simulator bindings object."""

        if not isinstance(self.unit_cell_devices, list):
            raise ConfigError("unit_cell_devices should be a list of devices")

        n_devices = len(self.unit_cell_devices)

        transfer_parameters = parameters_to_bindings(self)

        param_fast = self.unit_cell_devices[0].as_bindings()
        param_slow = self.unit_cell_devices[1].as_bindings()

        if not transfer_parameters.append_parameter(param_fast):
            raise ConfigError("Could not add unit cell device parameter")

        for _ in range(n_devices - 1):
            if not transfer_parameters.append_parameter(param_slow):
                raise ConfigError("Could not add unit cell device parameter")

        return transfer_parameters

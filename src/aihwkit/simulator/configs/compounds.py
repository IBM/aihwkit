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

# pylint: disable=too-many-instance-attributes, too-many-lines

"""Compound configuration for Analog (Resistive Device) tiles."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, List, Type, Union, TYPE_CHECKING
from warnings import warn

from aihwkit.exceptions import ConfigError
from aihwkit.simulator.configs.helpers import (
    _PrintableMixin, parameters_to_bindings
)
from aihwkit.simulator.configs.utils import (
    IOParameters, UpdateParameters
)
from aihwkit.simulator.configs.enums import VectorUnitCellUpdatePolicy
from aihwkit.simulator.rpu_base import devices

if TYPE_CHECKING:
    from aihwkit.simulator.configs.devices import PulsedDevice


@dataclass
class UnitCell(_PrintableMixin):
    """Parameters that modify the behaviour of a unit cell."""

    bindings_class: ClassVar[Type] = devices.VectorResistiveDeviceParameter

    bindings_ignore: ClassVar[List] = ['diffusion', 'lifetime']

    unit_cell_devices: List = field(default_factory=list)
    """Devices that compose this unit cell."""

    construction_seed: int = 0
    """If not ``0``, set a unique seed for hidden parameters during
    construction.

    Applies to all ``unit_cell_devices``.
    """

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

    momentum: float = 0.1
    """Momentum of the buffer.

    After transfer, this momentum fraction stays on the buffer instead
    of subtracting all of what was transferred.
    """

    forget_buffer: bool = True
    """Whether to forget the value of the buffer after transfer.

    If enabled, the buffer is reset to the momentum times the
    transferred value. Thus, if the number of pulses is limited to
    e.g. 1 (``desired_BL`` in the ``transfer_update``) the transfer
    might be clipped and the potentially larger buffer values are
    forgotten. If disabled, then the buffer values are faithfully
    subtracted by the amount transferred (times one minus momentum).
    """

    transfer_update: UpdateParameters = field(
        default_factory=lambda: UpdateParameters(desired_bl=1, update_bl_management=False,
                                                 update_management=False))
    """Update parameters that define the type of update used for each transfer
    event.

    Update parameters :class:`~aihwkit.simulator.config.utils.AnalogTileUpdateParameters` that
    define the type of update used for each transfer event.
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

    bindings_ignore: ClassVar[List] = ['diffusion', 'lifetime']

    device: Union['PulsedDevice',
                  OneSidedUnitCell,
                  VectorUnitCell,
                  ReferenceUnitCell] = field(
                      default_factory=VectorUnitCell)
    """(Analog) device that are used for forward and backward."""

    construction_seed: int = 0
    """If not ``0``, set a unique seed for hidden parameters during
    construction.

    Applies to ``device``.
    """

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

    stoc_round_x: bool = True
    """Whether to use stochastic rounding in case of quantization of the input x.
    """

    stoc_round_d: bool = True
    """Whether to use stochastic rounding in case of quantization of the error d.
    """

    def as_bindings(self) -> devices.MixedPrecResistiveDeviceParameter:
        """Return a representation of this instance as a simulator bindings object."""
        mixed_prec_parameter = parameters_to_bindings(self)
        param_device = self.device.as_bindings()

        if not mixed_prec_parameter.set_device_parameter(param_device):
            raise ConfigError('Could not add device parameter')

        return mixed_prec_parameter

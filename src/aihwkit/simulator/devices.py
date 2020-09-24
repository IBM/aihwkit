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

# pylint: disable=too-few-public-methods

"""High level analog devices."""

from dataclasses import is_dataclass
from enum import Enum
from typing import Any, Optional, List, Union

from aihwkit.simulator.parameters import (
    FloatingPointTileParameters, AnalogTileParameters,
    AbstractResistiveDeviceParameters,
    IdealResistiveDeviceParameters,
    PulsedResistiveDeviceBaseParameters,
    ConstantStepResistiveDeviceParameters,
    LinearStepResistiveDeviceParameters,
    SoftBoundsResistiveDeviceParameters,
    ExpStepResistiveDeviceParameters,
    DifferenceUnitCellParameters,
    VectorUnitCellParameters,
    TransferUnitCellParameters,
    AnalogTileBackwardInputOutputParameters,
    AnalogTileInputOutputParameters,
    AnalogTileUpdateParameters,
    PulseType
)
from aihwkit.simulator.rpu_base.tiles import FloatingPointTile, AnalogTile
from aihwkit.simulator.rpu_base import devices


class BaseResistiveDevice:
    """Base resistive device."""

    def create_tile(self, x_size: int, d_size: int) -> FloatingPointTile:
        """Return an analog tile of the specified dimensions.

        Args:
            x_size: number of rows.
            d_size: number of columns.
        """
        raise NotImplementedError

    @staticmethod
    def _parameters_to_bindings(params: Any) -> Any:
        """Convert a dataclass parameter into a bindings class."""
        result = params.bindings_class()
        for field, value in params.__dict__.items():
            # Convert enums to the bindings enums.
            if isinstance(value, Enum):
                enum_class = getattr(devices, value.__class__.__name__)
                enum_value = getattr(enum_class, value.value)
                setattr(result, field, enum_value)
            elif is_dataclass(value):
                setattr(result, field, BaseResistiveDevice._parameters_to_bindings(value))
            else:
                setattr(result, field, value)

        return result


class FloatingPointResistiveDevice(BaseResistiveDevice):
    """Floating point reference.

    This implements ideal devices forward/backward/update behavior.
    """

    def __init__(self,
                 params_basic: Optional[FloatingPointTileParameters] = None):
        self.params_devices = params_basic or FloatingPointTileParameters()

    def create_tile(self, x_size: int, d_size: int) -> FloatingPointTile:
        """Return a floating point tile of the specified dimensions."""
        # Create the tile.
        meta_parameter = self._parameters_to_bindings(self.params_devices)
        tile = meta_parameter.create_array(x_size, d_size)

        return tile


class PulsedResistiveDevice(BaseResistiveDevice):
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

    def __init__(self,
                 params_devices: Optional[Union[AbstractResistiveDeviceParameters,
                                                List[PulsedResistiveDeviceBaseParameters]
                                                ]] = None,
                 params_forward: Optional[AnalogTileInputOutputParameters] = None,
                 params_backward: Optional[AnalogTileBackwardInputOutputParameters] = None,
                 params_update: Optional[AnalogTileUpdateParameters] = None
                 ):
        self.params = AnalogTileParameters(
            forward_io=params_forward or AnalogTileInputOutputParameters(),
            backward_io=params_backward or AnalogTileBackwardInputOutputParameters(),
            update=params_update or AnalogTileUpdateParameters()
        )
        self.params_devices = params_devices or ConstantStepResistiveDeviceParameters()

    def create_tile(self, x_size: int, d_size: int) -> AnalogTile:
        """Returns an analog tile of the specified dimensions.

        Args:
            x_size: number of rows.
            d_size: number of columns.
        """
        # Prepare the basic parameters.
        meta_parameter = self.params.bindings_class()
        meta_parameter.forward_io = self._parameters_to_bindings(self.params.forward_io)
        meta_parameter.backward_io = self._parameters_to_bindings(self.params.backward_io)
        meta_parameter.update = self._parameters_to_bindings(self.params.update)

        # Create the tile.
        devices_parameters = self._parameters_to_bindings(self.params_devices)
        tile = meta_parameter.create_array(x_size, d_size, devices_parameters)

        return tile


class IdealResistiveDevice(PulsedResistiveDevice):
    """Ideal update behavior (using floating point), but forward/backward
    might be non-ideal.

    Ideal update behavior (using floating point), however,
    forward/backward might still have a non-ideal ADC or noise added.
    """
    def __init__(self,
                 params_devices: Optional[IdealResistiveDeviceParameters] = None,
                 params_forward: Optional[AnalogTileInputOutputParameters] = None,
                 params_backward: Optional[AnalogTileBackwardInputOutputParameters] = None,
                 params_update: Optional[AnalogTileUpdateParameters] = None
                 ):
        params_update = params_update or AnalogTileUpdateParameters()
        params_update.pulse_type = PulseType.NONE
        params_devices = params_devices or IdealResistiveDeviceParameters()
        super().__init__(params_devices, params_forward, params_backward, params_update)


class ConstantStepResistiveDevice(PulsedResistiveDevice):
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

    def __init__(self,
                 params_devices: Optional[ConstantStepResistiveDeviceParameters] = None,
                 params_forward: Optional[AnalogTileInputOutputParameters] = None,
                 params_backward: Optional[AnalogTileBackwardInputOutputParameters] = None,
                 params_update: Optional[AnalogTileUpdateParameters] = None
                 ):
        params_devices = params_devices or ConstantStepResistiveDeviceParameters()
        super().__init__(params_devices, params_forward, params_backward, params_update)


class LinearStepResistiveDevice(PulsedResistiveDevice):
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
    def __init__(self,
                 params_devices: Optional[LinearStepResistiveDeviceParameters] = None,
                 params_forward: Optional[AnalogTileInputOutputParameters] = None,
                 params_backward: Optional[AnalogTileBackwardInputOutputParameters] = None,
                 params_update: Optional[AnalogTileUpdateParameters] = None
                 ):
        params_devices = params_devices or LinearStepResistiveDeviceParameters()
        super().__init__(params_devices, params_forward, params_backward, params_update)


class SoftBoundsResistiveDevice(PulsedResistiveDevice):
    r"""Pulsed update behavioral model: soft bounds.

    Pulsed update behavioral model, where the update step response size
    of the material is linearly dependent and it goes to zero at the
    bound.

    This model is based on :class:`~LinearStepResistiveDevice` with
    parameters set to model soft bounds.
    """
    def __init__(self,
                 params_devices: Optional[SoftBoundsResistiveDeviceParameters] = None,
                 params_forward: Optional[AnalogTileInputOutputParameters] = None,
                 params_backward: Optional[AnalogTileBackwardInputOutputParameters] = None,
                 params_update: Optional[AnalogTileUpdateParameters] = None
                 ):
        params_devices = params_devices or SoftBoundsResistiveDeviceParameters()
        super().__init__(params_devices, params_forward, params_backward, params_update)


class ExpStepResistiveDevice(PulsedResistiveDevice):
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
    def __init__(self,
                 params_devices: Optional[ExpStepResistiveDeviceParameters] = None,
                 params_forward: Optional[AnalogTileInputOutputParameters] = None,
                 params_backward: Optional[AnalogTileBackwardInputOutputParameters] = None,
                 params_update: Optional[AnalogTileUpdateParameters] = None
                 ):
        params_devices = params_devices or ExpStepResistiveDeviceParameters()
        super().__init__(params_devices, params_forward, params_backward, params_update)


class VectorUnitCell(PulsedResistiveDevice):
    """Abstract resistive device that combines multiple pulsed resistive
    devices in a single 'unit cell'.

    For instance, a vector device can consist of 2 resistive devices
    where the sum of the two resistive values are coded for each
    weight of a cross point.

    Args:
       params_devices: List of pulsed resistive device parameters
       params_forward: Parameters governing the forward pass
       params_backward: Parameters governing the backward pass
       params_update: Parameters governing update pulse selection etc.
       **vector_kwargs: args that we be passed to
         class:`~aihwkit.simulator.parameters.VectorUnitCellParameters`
    """
    def __init__(self,
                 params_devices: Optional[List[PulsedResistiveDeviceBaseParameters]] = None,
                 params_forward: Optional[AnalogTileInputOutputParameters] = None,
                 params_backward: Optional[AnalogTileBackwardInputOutputParameters] = None,
                 params_update: Optional[AnalogTileUpdateParameters] = None,
                 **vector_kwargs: Any
                 ):

        self.params_vector = VectorUnitCellParameters(**vector_kwargs)

        super().__init__(params_devices, params_forward, params_backward, params_update)

    def create_tile(self, x_size: int, d_size: int) -> AnalogTile:
        """Returns an analog tile of the specified dimensions.

        Args:
            x_size: number of rows.
            d_size: number of columns.
        """
        # Prepare the basic parameters.
        meta_parameter = self.params.bindings_class()
        meta_parameter.forward_io = self._parameters_to_bindings(self.params.forward_io)
        meta_parameter.backward_io = self._parameters_to_bindings(self.params.backward_io)
        meta_parameter.update = self._parameters_to_bindings(self.params.update)

        # Create the tile.
        if not isinstance(self.params_devices, list):
            raise ValueError('Expect a list of device parameters')

        vector_parameters = self._parameters_to_bindings(self.params_vector)

        for param in self.params_devices:
            device_parameters = self._parameters_to_bindings(param)
            vector_parameters.append_parameter(device_parameters)

        tile = meta_parameter.create_array(x_size, d_size, vector_parameters)

        return tile


class DifferenceUnitCell(PulsedResistiveDevice):
    """Abstract device model takes an arbitrary device per crosspoint and
    implements an explicit plus-minus device pair.

    A plus minus pair is implemented by using only one-sided updated
    of the given devices. Note that reset might need to be called
    otherwise the one-sided device quickly saturates during learning.

    The output current is the difference of both devices

    Meta parameter setting of the pairs are assumed to be identical
    (however, device-to-device variation is still present).

    Caution:
       Reset needs to be added `manually` by calling the
       reset_columns method of a tile.
    """
    def __init__(self,
                 params_devices: Optional[PulsedResistiveDeviceBaseParameters] = None,
                 params_forward: Optional[AnalogTileInputOutputParameters] = None,
                 params_backward: Optional[AnalogTileBackwardInputOutputParameters] = None,
                 params_update: Optional[AnalogTileUpdateParameters] = None
                 ):
        super().__init__(params_devices, params_forward, params_backward, params_update)

    def create_tile(self, x_size: int, d_size: int) -> AnalogTile:
        """Returns an analog tile of the specified dimensions.

        Args:
            x_size: number of rows.
            d_size: number of columns.
        """
        # Prepare the basic parameters.
        meta_parameter = self.params.bindings_class()
        meta_parameter.forward_io = self._parameters_to_bindings(self.params.forward_io)
        meta_parameter.backward_io = self._parameters_to_bindings(self.params.backward_io)
        meta_parameter.update = self._parameters_to_bindings(self.params.update)

        difference_parameters = self._parameters_to_bindings(DifferenceUnitCellParameters())
        device_parameters = self._parameters_to_bindings(self.params_devices)

        # need to be exactly 2 and same parameters
        difference_parameters.append_parameter(device_parameters)
        difference_parameters.append_parameter(device_parameters)

        tile = meta_parameter.create_array(x_size, d_size, difference_parameters)

        return tile


class TransferUnitCell(PulsedResistiveDevice):
    r"""Abstract device model that takes 2 or more devices per crosspoint and
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
    can be different.

    Args:
       params_devices: List of pulsed resistive device parameters

          Note:
              This has to be a list, where the length of the list is
              the length of the chain of devices to transfer
              to. However, for all resistive devices that are
              transferred two (all except the first) the device
              parameters are taken to be the same (copied from the
              second in this list).

       params_forward: Parameters governing the forward pass
       params_backward: Parameters governing the backward pass
       params_update: Parameters governing update pulse selection etc. of the SGD update
       params_transfer_update: Parameters governing the update when doing the transfer
       params_transfer_forward: Parameters governing the forward when
          doing the read-out for the transfer
       **transfer_kwargs: additional args that we be passed to
         class:`~aihwkit.simulator.parameters.TransferUnitCellParameters`
    """
    def __init__(self,
                 params_devices: Optional[List[PulsedResistiveDeviceBaseParameters]] = None,
                 params_forward: Optional[AnalogTileInputOutputParameters] = None,
                 params_backward: Optional[AnalogTileBackwardInputOutputParameters] = None,
                 params_update: Optional[AnalogTileUpdateParameters] = None,
                 params_transfer_forward: Optional[AnalogTileInputOutputParameters] = None,
                 params_transfer_update: Optional[AnalogTileUpdateParameters] = None,
                 **transfer_kwargs: Any
                 ):

        transfer_forward = params_transfer_forward or params_forward or \
                           AnalogTileInputOutputParameters()

        transfer_up = params_transfer_update or params_update or AnalogTileUpdateParameters()

        self.transfer_params = TransferUnitCellParameters(**transfer_kwargs)
        self.transfer_params.params_transfer_update = transfer_up
        self.transfer_params.params_transfer_forward = transfer_forward

        super().__init__(params_devices, params_forward, params_backward, params_update)

    def create_tile(self, x_size: int, d_size: int) -> AnalogTile:
        """Returns an analog tile of the specified dimensions.

        Args:
            x_size: number of rows.
            d_size: number of columns.
        """
        if not isinstance(self.params_devices, list):
            raise ValueError('`self.param_devices` must be a list of device parameters')
        n_devices = len(self.params_devices)
        if n_devices < 2:
            raise RuntimeError('`self.param_devices` needs to contain at least 2 devices')

        # Prepare the basic parameters.
        meta_parameter = self.params.bindings_class()
        meta_parameter.forward_io = self._parameters_to_bindings(self.params.forward_io)
        meta_parameter.backward_io = self._parameters_to_bindings(self.params.backward_io)
        meta_parameter.update = self._parameters_to_bindings(self.params.update)

        # Create the tile.
        transfer_parameters = self._parameters_to_bindings(self.transfer_params)

        param_fast = self._parameters_to_bindings(self.params_devices[0])
        param_slow = self._parameters_to_bindings(self.params_devices[1])

        transfer_parameters.append_parameter(param_fast)

        for _ in range(n_devices-1):
            transfer_parameters.append_parameter(param_slow)

        tile = meta_parameter.create_array(x_size, d_size, transfer_parameters)

        return tile

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

from dataclasses import asdict
from enum import Enum
from typing import Any, Optional

from aihwkit.simulator.parameters import (
    FloatingPointTileParameters, AnalogTileParameters,
    ConstantStepResistiveDeviceParameters,
    AnalogTileBackwardInputOutputParameters,
    AnalogTileInputOutputParameters,
    AnalogTileUpdateParameters,
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
        for field, value in asdict(params).items():
            # Convert enums to the bindings enums.
            if isinstance(value, Enum):
                enum_class = getattr(devices, value.__class__.__name__)
                enum_value = getattr(enum_class, value.value)
                setattr(result, field, enum_value)
            else:
                setattr(result, field, value)

        return result


class FloatingPointResistiveDevice(BaseResistiveDevice):
    """Floating point resistive devices.

    This implements ideal devices update behavior (floating
    point update).
    """

    def __init__(self,
                 params_basic: Optional[FloatingPointTileParameters] = None):
        self.params_devices = params_basic or FloatingPointTileParameters()

    def create_tile(self, x_size: int, d_size: int) -> FloatingPointTile:
        """Return a floating point tile of the specified dimensions."""
        # Create the tile.
        meta_parameter = self._parameters_to_bindings(self.params_devices)
        tile = meta_parameter.create_array(x_size, d_size)

        # Initialize the weights.
        tile.set_weights_uniform_random(-0.1, 0.1)
        return tile


class ConstantStepResistiveDevice(BaseResistiveDevice):
    r"""ConstantStep resistive devices.

    Device are used as part of an
    :class:`~aihwkit.simulator.tiles.AnalogTile` to implement the
    `update once` characteristics, i.e. the material response properties
    when a single update pulse is given (a coincidence between row and
    column pulse train happened).

    The form implemented for `ConstantStep` is:

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

    For parameters regarding the devices settings, see
    :class:`~aihwkit.simulator.parameters.ConstantStepResistiveDeviceParameters`.


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
                 params_devices: Optional[ConstantStepResistiveDeviceParameters] = None,
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

        # Initialize the weights.
        tile.set_weights_uniform_random(self.params_devices.w_min,
                                        self.params_devices.w_max)
        return tile

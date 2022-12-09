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

"""Parameter context for analog tiles."""

from typing import Optional, Type, Union, Any, TYPE_CHECKING

from torch import ones, dtype, Tensor, no_grad
from torch.nn import Parameter
from torch import device as torch_device

if TYPE_CHECKING:
    from aihwkit.simulator.tiles.base import BaseTile


class AnalogContext(Parameter):
    """Context for analog optimizer."""

    def __new__(cls: Type['AnalogContext'], analog_tile: 'BaseTile',
                parameter: Optional[Parameter] = None) -> 'AnalogContext':
        # pylint: disable=signature-differs
        if parameter is None:
            return Parameter.__new__(cls, data=ones((), device=analog_tile.device),
                                     requires_grad=True)
        parameter.__class__ = cls
        return parameter

    def __init__(self, analog_tile: 'BaseTile',
                 parameter: Optional[Parameter] = None):  # pylint: disable=unused-argument
        super().__init__()
        self.analog_tile = analog_tile
        self.use_torch_update = False
        self.use_indexed = False
        self.analog_input = []  # type: list
        self.analog_grad_output = []  # type: list
        self.reset(analog_tile)

    def set_data(self, data: Tensor) -> None:
        """Set the data value of the Tensor."""
        # pylint: disable=attribute-defined-outside-init
        with no_grad():
            self.data.copy_(data)

    def get_data(self) -> Tensor:
        """Get the data value of the underlying Tensor."""
        # pylint: disable=attribute-defined-outside-init
        return self.data.detach()

    def reset(self, analog_tile: Optional['BaseTile'] = None) -> None:
        """Reset the gradient trace and optionally sets the tile pointer."""

        if analog_tile is not None:
            self.analog_tile = analog_tile
            self.analog_tile.analog_ctx = self

        self.analog_input = []
        self.analog_grad_output = []

    def has_gradient(self) -> bool:
        """Return whether a gradient trace was stored."""
        return len(self.analog_input) > 0

    def __copy__(self) -> Parameter:
        """Turn off copying of the pointers. Context will be re-created
        when tile is created"""
        return Parameter(self.data)

    def __deepcopy__(self, memo: Any) -> Parameter:
        """ Turn off deep copying. Context will be re-created when tile is created """
        return Parameter(self.data)

    def cuda(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'AnalogContext':
        """Move the context to a cuda device.

        Args:
             device: the desired device of the tile.

        Returns:
            This context in the specified device.
        """
        # pylint: disable=attribute-defined-outside-init
        self.data = self.data.cuda(device)  # type: Tensor

        if not self.analog_tile.is_cuda:
            self.analog_tile = self.analog_tile.cuda(device)
        self.reset(self.analog_tile)

        return self

    def cpu(self) -> 'AnalogContext':
        """Move the context to CPU.

        Note:
            This is a no-op for CPU context.

        Returns:
            self
        """
        super().cpu()
        if self.analog_tile is not None:
            self.analog_tile = self.analog_tile.cpu()  # will raise an error if not possile
        return self

    def to(self, *args: Any, **kwargs: Any) -> 'AnalogContext':
        """Move analog tiles of the current context to a device.

        Note:
            Please be aware that moving analog tiles from GPU to CPU is
            currently not supported.

        Caution:
            Other tensor conversions than moving the device to CUDA,
            such as changing the data type are not supported for analog
            tiles and will be simply ignored.

        Returns:
            This module in the specified device.
        """
        # pylint: disable=invalid-name, attribute-defined-outside-init
        self.data = self.data.to(*args, **kwargs)

        device = None
        if 'device' in kwargs:
            device = kwargs['device']
        elif len(args) > 0 and not isinstance(args[0], (Tensor, dtype)):
            device = torch_device(args[0])

        if device is not None:
            device = torch_device(device)
            if device.type == 'cuda' and not self.analog_tile.is_cuda:
                self.analog_tile = self.analog_tile.cuda(device)
            self.reset(self.analog_tile)

        return self

    def __repr__(self) -> str:
        return 'AnalogContext of ' + self.analog_tile.get_brief_info()

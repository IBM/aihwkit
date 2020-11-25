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

"""Analog Modules that contain children Modules."""

from typing import Callable, Optional, Union

from torch import device as torch_device
from torch.nn import Sequential

from aihwkit.exceptions import ModuleError
from aihwkit.nn.modules.base import AnalogModuleBase


class AnalogSequential(Sequential):
    """An analog-aware sequential container.

    Specialization of torch ``nn.Sequential`` with extra functionality for
    handling analog layers:
    * correct handling of ``.cuda()`` for children modules.
    * apply analog-specific functions to all its children (drift and program
      weights).

    Note:
        This class is recommended to be used in place of ``nn.Sequential`` in
        order to correctly propagate the actions to all the children analog
        layers. If using regular containers, please be aware that operations
        need to be applied manually to the children analog layers when needed.
    """
    # pylint: disable=abstract-method

    def _apply_to_analog(self, fn: Callable) -> 'AnalogSequential':
        """Apply a function to all the analog layers in this module.

        Args:
            fn: function to be applied.

        Returns:
            This module after the function has been applied.
        """
        for module in self.modules():
            if isinstance(module, AnalogModuleBase):
                fn(module)

        return self

    def cuda(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'AnalogSequential':
        super().cuda(device)

        self._apply_to_analog(lambda m: m.cuda())

        return self

    def drift_analog_weights(self, t_inference: float = 0.0) -> None:
        """(Program) and drift all analog inference layers of a given model.

        Args:
            t_inference: assumed time of inference (in sec)

        Raises:
            ModuleError: if the layer is not in evaluation mode.
        """
        if self.training:
            raise ModuleError('drift_analog_weights can only be applied in '
                              'evaluation mode')

        self._apply_to_analog(lambda m: m.drift_analog_weights(t_inference))

    def program_analog_weights(self) -> None:
        """Program all analog inference layers of a given model.

        Raises:
            ModuleError: if the layer is not in evaluation mode.
        """
        if self.training:
            raise ModuleError('program_analog_weights can only be applied in '
                              'evaluation mode')

        self._apply_to_analog(lambda m: m.program_weights())

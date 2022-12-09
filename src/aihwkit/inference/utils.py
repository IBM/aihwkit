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

"""Utilities for inference."""

from torch.nn import Module
from aihwkit.exceptions import ModuleError


def drift_analog_weights(model: Module, t_inference: float = 0.0) -> None:
    """(Program) and drift all analog inference layers of a given model.

    Args:
        model: torch model with analog layers
        t_inference: assumed time of inference (in sec)

    Raises:
        ModuleError: if the layer is not in evaluation mode.
    """
    # avoid circular import
    # pylint: disable=import-outside-toplevel
    from aihwkit.nn.modules.base import AnalogModuleBase
    if model.training:
        raise ModuleError('drift_analog_weights can only be applied in '
                          'evaluation mode')

    for module in model.modules():
        if not isinstance(module, AnalogModuleBase):
            continue
        module.drift_analog_weights(t_inference)


def program_analog_weights(model: Module) -> None:
    """Program all analog inference layers of a given model.

    Args:
        model: torch model with analog layers

    Raises:
        ModuleError: if the layer is not in evaluation mode.
    """
    # avoid circular import
    # pylint: disable=import-outside-toplevel
    from aihwkit.nn.modules.base import AnalogModuleBase
    if model.training:
        raise ModuleError('program_analog_weights can only be applied in '
                          'evaluation mode')

    for module in model.modules():
        if not isinstance(module, AnalogModuleBase):
            continue
        module.program_analog_weights()

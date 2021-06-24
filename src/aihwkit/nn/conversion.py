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

"""Digital/analog model conversion utilities.

This module includes tools for converting a given torch model to a
model containing analog layers.
"""

from typing import TypeVar, Optional, Dict
import copy

from torch.nn import Module, Linear, Conv1d, Conv2d, Conv3d, Sequential

from aihwkit.nn import (
    AnalogLinear, AnalogConv1d, AnalogConv2d, AnalogConv3d, AnalogSequential
)

RPUConfigGeneric = TypeVar('RPUConfigGeneric')

_DEFAULT_CONVERSION_MAP = {Linear: AnalogLinear,
                           Conv1d: AnalogConv1d,
                           Conv2d: AnalogConv2d,
                           Conv3d: AnalogConv3d,
                           Sequential: AnalogSequential}


def convert_to_analog(
        module: Module,
        rpu_config: RPUConfigGeneric,
        realistic_read_write: bool = False,
        weight_scaling_omega: float = 0.0,
        conversion_map: Optional[Dict] = None
) -> Module:
    """Convert a given digital model to analog counter parts.

    Note:
        The torch device (cuda/cpu) is inferred from the original
        models parameters, however, if multiple torch
        devices are used in a given module, the corresponding analog
        module is not moved to any device.

    Args:
        module: The torch module to convert. All layers that are
            defined in the ``conversion_map``.
        rpu_config: RPU config to apply to all converted tiles.
            Applied to all converted tiles.
        realistic_read_write: Whether to use closed-loop programming
            when setting the weights. Applied to all converted tiles.
        weight_scaling_omega: If non-zero, applied weights of analog
            layers will be scaled by ``weight_scaling_omega`` divided by
            the absolute maximum value of the original weight matrix.

            Note:
                Make sure that the weight max and min setting of the
                device support the desired analog weight range.
        conversion_map: Dictionary of module classes to be replaced in
            case of custom replacement rules. By default all ``Conv`` and ``Linear``
            layers are replaced with their analog counterparts.

            Note:
                The analog layer needs to have a class method
                ``from_digital`` which will be called during the
                conversion.

    Returns:
        module with replaced digital layers with analog layers.
    """
    module = copy.deepcopy(module)

    if conversion_map is None:
        conversion_map = _DEFAULT_CONVERSION_MAP

    # Convert parent.
    if module.__class__ in conversion_map:
        module = conversion_map[module.__class__].from_digital(  # type: ignore
            module, rpu_config, realistic_read_write, weight_scaling_omega)

    # Convert children.
    convert_dic = {}
    for name, mod in module.named_children():

        n_grand_children = len(list(mod.named_children()))
        if n_grand_children > 0:
            new_mod = convert_to_analog(mod, rpu_config, realistic_read_write,
                                        weight_scaling_omega, conversion_map)

        elif mod.__class__ in conversion_map:
            new_mod = conversion_map[mod.__class__].from_digital(   # type: ignore
                mod, rpu_config, realistic_read_write, weight_scaling_omega)
        else:
            continue

        devices = {p.device for p in mod.parameters()}
        if len(devices) == 1:
            # We only use "to" if device is unique.
            new_mod = new_mod.to(list(devices)[0])
        convert_dic[name] = new_mod

    for name, new_mod in convert_dic.items():
        module._modules[name] = new_mod  # pylint: disable=protected-access

    return module

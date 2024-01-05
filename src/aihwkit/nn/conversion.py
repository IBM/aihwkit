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

"""Digital/analog model conversion utilities.

This module includes tools for converting a given torch model to a
model containing analog layers.
"""

from typing import TypeVar, Optional, Dict, Callable, Set, List
from copy import deepcopy

from torch.nn import Module, Linear, Conv1d, Conv2d, Conv3d, Sequential

from aihwkit.exceptions import ArgumentError
from aihwkit.simulator.tiles.module import TileModule
from aihwkit.nn.modules.container import AnalogWrapper
from aihwkit.nn.modules.base import AnalogLayerBase

from aihwkit.nn import (
    AnalogLinear,
    AnalogConv1d,
    AnalogConv2d,
    AnalogConv3d,
    AnalogLinearMapped,
    AnalogConv1dMapped,
    AnalogConv2dMapped,
    AnalogConv3dMapped,
    AnalogSequential,
)

RPUConfigGeneric = TypeVar("RPUConfigGeneric")

_DEFAULT_CONVERSION_MAP = {
    Linear: AnalogLinear,
    Conv1d: AnalogConv1d,
    Conv2d: AnalogConv2d,
    Conv3d: AnalogConv3d,
    Sequential: AnalogSequential,
}

_DEFAULT_MAPPED_CONVERSION_MAP = {
    Linear: AnalogLinearMapped,
    Conv1d: AnalogConv1dMapped,
    Conv2d: AnalogConv2dMapped,
    Conv3d: AnalogConv3dMapped,
    Sequential: AnalogSequential,
}

_DEFAULT_DIGITAL_CONVERSION_SET = {
    *_DEFAULT_CONVERSION_MAP.values(),
    *_DEFAULT_MAPPED_CONVERSION_MAP.values(),
}


def specific_rpu_config_id(
    module_name: str, module: Module, rpu_config: RPUConfigGeneric
) -> RPUConfigGeneric:
    """ID default function for specifying the ``RPUConfig`` during conversion
    for specific layers.

    A similar function can be given to the conversion.

    Args:
       module_name: The name of the module currently converted to analog
       module: the actual digital module to be converted
       rpu_config: a copy of the generic ``RPUConfig`` given to the
           overall conversion.

    Returns:
       modified ``RPUConfig``
    """
    # pylint: disable=unused-argument
    return rpu_config


def convert_to_analog(
    module: Module,
    rpu_config: RPUConfigGeneric,
    tile_module_class: Optional[TileModule] = None,
    conversion_map: Optional[Dict] = None,
    specific_rpu_config_fun: Optional[Callable] = None,
    module_name: str = "",
    ensure_analog_root: bool = True,
    exclude_modules: Optional[List[str]] = None,
    inplace: bool = False,
    verbose: bool = False,
) -> Module:
    """Convert a given digital model to its analog counterpart.

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
        tile_module_class: Custom tile module class
        conversion_map: Dictionary of module classes to be replaced in
            case of custom replacement rules. By default all ``Conv`` and ``Linear``
            layers are replaced with their analog counterparts.

            Note:
                The analog layer needs to have a class method
                ``from_digital`` which will be called during the
                conversion.

        specific_rpu_config_fun: Function that modifies the generic
            RPUConfig for specific modules. See
            :func:`~specific_rpu_config_id` as an example how to
            specify it.

        module_name: Explicitly given name of the base (root) module,
            given to ``specific_rpu_config_fun``.

        ensure_analog_root: Whether to ensure that the root module is
            of layer type `AnalogLayerBase` so that custom analog are
            methods such as `drift_analog_weigths` are available. If
            set, it will wrap the model if `AnalogWrapper` if necessary.

            Note:

                Since the module structure changes when wrapped, the
                checkpoint names will also change if this is
                enabled (for legacy load this might need to be disabled).

        exclude_modules: List of module names that are in the
            conversion map but should be excluded from the conversion

        inplace: Whether to perform in-place conversion (without deepcopy)

        verbose: Increase verbosity. Will print converted layers.

    Returns:
        Module where all the digital layers are replaced with analog
        mapped layers.

    Raises:
        ArgumentError: in case conversion map is not a dictionary

    """
    # pylint: disable=too-many-branches, too-many-locals
    if exclude_modules is None:
        exclude_modules = []

    if not inplace:
        module = deepcopy(module)

    if conversion_map is None:
        conversion_map = _DEFAULT_CONVERSION_MAP

    if not isinstance(conversion_map, dict):
        raise ArgumentError("Conversion map should be a dictionary")

    if specific_rpu_config_fun is None:
        specific_rpu_config_fun = specific_rpu_config_id

    # Convert parent.
    if module.__class__ in conversion_map and module_name not in exclude_modules:
        if verbose:
            print(f"Converted '{module_name}' to '{conversion_map[module.__class__].__name__}'.")
        module = conversion_map[module.__class__].from_digital(
            module,
            specific_rpu_config_fun(module_name, module, deepcopy(rpu_config)),
            tile_module_class,
        )

    # Convert children.
    convert_dic = {}
    for name, mod in module.named_children():
        full_name = module_name + "." + name if module_name else name

        if full_name in exclude_modules:
            continue

        n_grand_children = len(list(mod.named_children()))
        if n_grand_children > 0:
            convert_dic[name] = convert_to_analog(
                mod,
                rpu_config,
                tile_module_class,
                conversion_map,
                specific_rpu_config_fun,
                full_name,
                True,
                exclude_modules,
                True,
                verbose,
            )
            continue

        if mod.__class__ not in conversion_map.keys():
            continue
        if verbose:
            print(f"Converted '{full_name}' to '{conversion_map[mod.__class__].__name__}'.")
        convert_dic[name] = conversion_map[mod.__class__].from_digital(
            mod, specific_rpu_config_fun(full_name, mod, deepcopy(rpu_config)), tile_module_class
        )

    for name, new_mod in convert_dic.items():
        module._modules[name] = new_mod  # pylint: disable=protected-access

    # in case of root, make sure it is wrapped as analog
    if ensure_analog_root and not module_name and not isinstance(module, AnalogLayerBase):
        module = AnalogWrapper(module)

    return module


def convert_to_analog_mapped(
    module: Module,
    rpu_config: RPUConfigGeneric,
    tile_module_class: Optional[TileModule] = None,
    specific_rpu_config_fun: Optional[Callable] = None,
    module_name: str = "",
    ensure_analog_root: bool = True,
    exclude_modules: Optional[List[str]] = None,
    inplace: bool = False,
    verbose: bool = False,
) -> Module:
    """Convert a given digital model to its analog counterpart with tile
    mapping support.

    Note:
        The torch device (cuda/cpu) is inferred from the original
        models parameters, however, if multiple torch
        devices are used in a given module, the corresponding analog
        module is not moved to any device.

    Args:
        module: The torch module to convert. All layers that are
            defined in the ``conversion_map``.
        rpu_config: RPU config to apply to all converted tiles.
        tile_module_class: Custom tile module class
        specific_rpu_config_fun: Function that modifies the generic
            RPUConfig for specific modules. See
            :func:`~specific_rpu_config_id` as an example how to
            specify it.

        module_name: Explicitly given name of the base (root) module,
            given to ``specific_rpu_config_fun``.

        ensure_analog_root: Whether to ensure that the root module is
            of layer type `AnalogLayerBase` so that custom analog are
            methods such as `drift_analog_weigths` are available. If
            set, it will wrap the model if `AnalogWrapper` if necessary.

            Note:

                Since the module structure changes when wrapped, the
                checkpoint names will also change if this is
                enabled (for legacy load this might need to be disabled).

        exclude_modules: List of modules names that are in the
            conversion map but should be excluded from the conversion

        inplace: Whether to for in place conversion (without deepcopy)

        verbose: Increase verbosity. Will print converted layers.


    Returns:
        Module where all the digital layers are replaced with analog
        mapped layers.

    """
    return convert_to_analog(
        module,
        rpu_config,
        tile_module_class,
        _DEFAULT_MAPPED_CONVERSION_MAP,
        specific_rpu_config_fun,
        module_name,
        ensure_analog_root,
        exclude_modules,
        inplace,
        verbose,
    )


def convert_to_digital(
    module: Module,
    conversion_set: Optional[Set] = None,
    module_name: str = "",
    inplace: bool = False,
) -> Module:
    """Convert a given analog model to digital counter parts.

    Note:
        The torch device (cuda/cpu) is inferred from the original
        models parameters, however, if multiple torch
        devices are used in a given module, the corresponding analog
        module is not moved to any device.

    Args:
        module: The torch module to convert having analog layers. All
            analog layer classes to be converted are defined in the
            ``conversion_set``.

        conversion_set: Set of analog module classes to be replaced. By default all
            ``AnalogConv*`` and ``AnalogLinear*`` layers are replaced
            with their digital counterparts.

            Note:
                The analog layer needs to have a class method
                ``to_digital`` which will be called during the
                conversion.

        module_name: Explicitly given name of the base (root) module

        inplace: Whether to for in place conversion (without deepcopy)

    Returns:
        Module where all the analog layers are replaced with digital
        layers

    Raises:
        ArgumentError: in case conversion map is not a set or contains
            layers that cannot be converted
    """
    # pylint: disable=too-many-branches

    if not inplace:
        module = deepcopy(module)

    if conversion_set is None:
        conversion_set = _DEFAULT_DIGITAL_CONVERSION_SET

    if not isinstance(conversion_set, set):
        raise ArgumentError("Conversion set should be a set")

    # Convert parent.
    if module.__class__ in conversion_set or isinstance(module, AnalogWrapper):
        if not hasattr(module.__class__, "to_digital"):
            raise ArgumentError(
                "The conversion set should only contain "
                "analog layers that support `to_digital` conversion."
            )
        module = module.__class__.to_digital(module)

    # Convert children.
    convert_dic = {}
    for name, mod in module.named_children():
        full_name = module_name + "." + name if module_name else name
        n_grand_children = len(list(mod.named_children()))

        if n_grand_children > 0:
            convert_dic[name] = convert_to_digital(mod, conversion_set, full_name, inplace=True)
            continue

        if mod.__class__ not in conversion_set:
            continue

        if not hasattr(mod, "to_digital"):
            raise ArgumentError(
                "The conversion set should only contain "
                "analog layers that support `to_digital` conversion."
            )
        convert_dic[name] = mod.to_digital(mod)  # type: ignore

    for name, new_mod in convert_dic.items():
        module._modules[name] = new_mod  # pylint: disable=protected-access

    return module

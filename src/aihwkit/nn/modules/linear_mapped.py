# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Analog mapped linear layer."""

from typing import Optional, Type

from torch.nn import Linear

from aihwkit.exceptions import ConfigError
from aihwkit.simulator.parameters.base import RPUConfigBase
from aihwkit.simulator.parameters.mapping import MappableRPU
from aihwkit.simulator.tiles.array import TileModuleArray
from aihwkit.nn.modules.linear import AnalogLinear


class AnalogLinearMapped(AnalogLinear):
    """Linear layer that uses one or several analog tiles.

    Linear layer that uses an analog tile during its forward, backward
    and update passes. In contrast to
    :class:`~aihwkit.bb.modules.linear.Linear` the maximal in and/or
    out dimension can be restricted, in which case the linear layer is
    split into multiple parts and computed on multiple tiles of given
    max sizes.

    In contrast to :class:`~aihwkit.nn.modules.linear.Linear`, the
    bias vector (if requested) is always handled in digital (floating
    point).

    Internally the
    :class:`~aihwkit.nn.modules.array.TileModuleArray` is used
    for the logical tiling process.

    Note:
        Mapping is controlled by the
        :class:`~aihwkit.simulator.parameters.mapping.MappingParameter`.

    Note:
        The tensor parameters of this layer (``.weight`` and ``.bias``) are not
        guaranteed to contain the same values as the internal weights and biases
        stored in the analog tile. Please use ``set_weights`` and
        ``get_weights`` when attempting to read or modify the weight/bias.

    Args:
        in_features: input vector size (number of columns).
        out_features: output vector size (number of rows).
        rpu_config: resistive processing unit configuration.
        bias: whether to use a bias row on the analog tile or not

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rpu_config: Optional[MappableRPU] = None,
        tile_module_class: Optional[Type] = TileModuleArray,
    ):
        super().__init__(in_features, out_features, bias, rpu_config, tile_module_class)

    @classmethod
    def from_digital(
        cls,
        module: Linear,
        rpu_config: RPUConfigBase,
        tile_module_class: Optional[Type] = TileModuleArray,
    ) -> "AnalogLinear":
        """Return an AnalogLinear layer from a torch Linear layer.

        Args:
            module: The torch module to convert. All layers that are
                defined in the ``conversion_map``.
            rpu_config: RPU config to apply to all converted tiles.
                Applied to all converted tiles.
            tile_module_class: Class of the underlying
                `TileModule`. If not given, will select based on
                the `MappingParameter` setting either
                :class:`~aihwkit.simulator.tiles.base.TileModuleBase` or
                :class:`~aihwkit.simulator.tiles.array.TileModuleArray`

        Returns:
            an AnalogLinear layer based on the digital Linear ``module``.

        Raises:
            ConfigError: In case the ``RPUConfig`` is not of type ``MappableRPU``
        """
        if not isinstance(rpu_config, MappableRPU):
            raise ConfigError("Only mappable RPUConfigs are supported.")

        analog_layer = cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            rpu_config,
            tile_module_class,
        )

        analog_layer.set_weights(module.weight, module.bias)
        return analog_layer.to(module.weight.device)

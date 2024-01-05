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

"""Analog layers."""
from typing import Optional, Type

from torch import Tensor
from torch.nn import Linear

from aihwkit.nn.modules.base import AnalogLayerBase
from aihwkit.simulator.parameters.base import RPUConfigBase


class AnalogLinear(AnalogLayerBase, Linear):
    """Linear layer that uses an analog tile.

    Linear layer that uses an analog tile during its forward, backward and
    update passes.

    Note:
        The tensor parameters of this layer (``.weight`` and ``.bias``) are not
        guaranteed to contain the same values as the internal weights and biases
        stored in the analog tile. Please use ``set_weights`` and
        ``get_weights`` when attempting to read or modify the weight/bias. This
        read/write process can simulate the (noisy and inexact) analog writing
        and reading of the resistive elements.

    Args:
        in_features: input vector size (number of columns).
        out_features: output vector size (number of rows).
        bias: whether to use a bias row on the analog tile or not.
            for setting initial weights and during reading of the weights.
        rpu_config: resistive processing unit configuration.
        tile_module_class: Class for the tile module (default
            will be specified from the ``RPUConfig``).
    """

    # pylint: disable=abstract-method

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rpu_config: Optional[RPUConfigBase] = None,
        tile_module_class: Optional[Type] = None,
    ):
        # Call super()
        Linear.__init__(self, in_features, out_features, bias=bias)

        # Create tile
        if rpu_config is None:
            # pylint: disable=import-outside-toplevel
            from aihwkit.simulator.configs.configs import SingleRPUConfig

            rpu_config = SingleRPUConfig()

        AnalogLayerBase.__init__(self)

        if tile_module_class is None:
            tile_module_class = rpu_config.get_default_tile_module_class(out_features, in_features)

        self.analog_module = tile_module_class(out_features, in_features, rpu_config, bias)
        # Unregister weight/bias as a parameter.
        self.unregister_parameter("weight")
        if bias:
            self.unregister_parameter("bias")
        else:
            # Seems to be a torch bug.
            self._parameters.pop("bias", None)
        self.bias = bias

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters (weight and bias)."""
        if hasattr(self, "analog_module"):
            bias = self.bias
            self.weight, self.bias = self.get_weights()  # type: ignore
            super().reset_parameters()
            self.set_weights(self.weight, self.bias)  # type: ignore
            self.weight, self.bias = None, bias

    def forward(self, x_input: Tensor) -> Tensor:
        """Compute the forward pass."""
        # pylint: disable=arguments-differ, arguments-renamed

        return self.analog_module(x_input)  # type: ignore

    @classmethod
    def from_digital(
        cls, module: Linear, rpu_config: RPUConfigBase, tile_module_class: Optional[Type] = None
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
                :class:`~aihwkit.simulator.tiles.base.TileModule` or
                :class:`~aihwkit.simulator.tiles.array.TileModuleArray`

        Returns:
            an AnalogLinear layer based on the digital Linear ``module``.
        """
        analog_layer = cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            rpu_config,
            tile_module_class,
        )

        analog_layer.set_weights(module.weight, module.bias)
        return analog_layer.to(module.weight.device)

    @classmethod
    def to_digital(cls, module: "AnalogLinear", realistic: bool = False) -> "Linear":
        """Return an nn.Linear layer from an AnalogLinear layer.

        Args:
            module: The analog module to convert.
            realistic: whehter to estimate the weights with the
                non-ideal forward pass. If not set, analog weights are
                (unrealistically) copies exactly

        Returns:
            an torch Linear layer with the same dimension and weights
            as the analog linear layer.
        """
        weight, bias = module.get_weights(realistic=realistic)
        digital_layer = Linear(module.in_features, module.out_features, bias is not None)
        digital_layer.weight.data = weight.data
        if bias is not None:
            digital_layer.bias.data = bias.data
        analog_tile = next(module.analog_tiles())
        return digital_layer.to(device=analog_tile.device, dtype=analog_tile.get_dtype())

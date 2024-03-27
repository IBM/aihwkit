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
from typing import Optional, Type, List

from torch import Tensor
from torch.nn import Linear, ParameterList

from aihwkit.exceptions import ModuleError
from aihwkit.nn.modules.base import AnalogLayerBase
from aihwkit.simulator.parameters.base import RPUConfigBase
from aihwkit.nn import AnalogLinear

class AnalogLinearBitSlicing(AnalogLayerBase, Linear):
    """Linear layer that implements bit-slicing in an analog tile.

    A linear layer that implements bit-slicing in an analog tile during its forward, backward and
    update passes. The bit-slicing is implemented by adding multiple AnalogLinear layers over which
    the weights are sliced and distributed

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
        number_slices: number of slices the weights should be distributed over
        bias: whether to use a bias row on the analog tile or not.
            for setting initial weights and during reading of the weights.
        evenly_sliced: whether to slice the weights evenly over all slices, or to
            go from MSB to LSB (factor = 2^x over the slices)
        rpu_config: resistive processing unit configuration.
        tile_module_class: Class for the tile module (default
            will be specified from the ``RPUConfig``).
        significance_factors: the factors by which each slice is multiplied, can
            be specified per slice
    """

    # pylint: disable=abstract-method

    def __init__(
        self,
        in_features: int,
        out_features: int,
        number_slices: int,
        bias: bool = True,
        evenly_sliced: bool = True,
        rpu_config: Optional[RPUConfigBase] = None,
        tile_module_class: Optional[Type] = None,
        significance_factors: Optional[List[float]] = None,
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

        self.significance_factors = []
        if significance_factors is None:
            if evenly_sliced is True:
                self.significance_factors = [1] * number_slices
            else:
                #we increase the significance by 2 for every slice
                self.significance_factors = [1]
                for _ in range(1, number_slices):
                    self.significance_factors.append(self.significance_factors[-1]*2)
        else:
            if len(significance_factors) != number_slices:
                raise ModuleError(f"Length of factors must equal number of slices exactly")
            self.significance_factors = significance_factors
        

        #self.analog_module = tile_module_class(out_features, in_features, rpu_config, bias)
        self.analog_slices = ParameterList(AnalogLinear(in_features, out_features, bias, rpu_config, tile_module_class) for i in range(number_slices))
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
            for weight_slice in self.analog_slices:
                bias = weight_slice.bias
                weight_slice.weight, weight_slice.bias = weight_slice.get_weights()  # type: ignore
                weight_slice.super().reset_parameters()
                weight_slice.set_weights(weight_slice.weight, weight_slice.bias)  # type: ignore
                weight_slice.weight, weight_slice.bias = None, bias

    def forward(self, x_input: Tensor) -> Tensor:
        """Compute the forward pass."""
        # pylint: disable=arguments-differ, arguments-renamed
        forward_output = 0 #to initialize the forward_output for the loop
        for idx, weight_slice in enumerate(self.analog_slices):
            forward_output += (weight_slice(x_input) * self.significance_factors[idx]) 

        return forward_output  # type: ignore

    @classmethod
    def from_digital(
        cls, module: Linear, rpu_config: RPUConfigBase, tile_module_class: Optional[Type] = None
    ) -> "AnalogLinearBitSlicing":
        """Return an AnalogLinearBitSlicing layer from a torch Linear layer.

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
            an AnalogLinearBitSlicing layer based on the digital Linear ``module``. Defaults to evenly sliced weights, with 8 slices.
        """
        analog_layer = cls(
            module.in_features,
            module.out_features,
            8,
            module.bias is not None,
            True,
            rpu_config,
            tile_module_class,
        )

        #slice total weight over number of slices and distribute over each slice part of the weight evenly
        for weight_slice in analog_layer.analog_slices:
            weight_slice.set_weights(module.weight/len(analog_layer.analog_slices), module.bias/len(analog_layer.analog_slices))

        return analog_layer.to(module.weight.device)

    @classmethod
    def to_digital(cls, module: "AnalogLinearBitSlicing", realistic: bool = False) -> "Linear":
        """Return an nn.Linear layer from an AnalogLinearBitSlicing layer.

        Args:
            module: The analog module to convert.
            realistic: whehter to estimate the weights with the
                non-ideal forward pass. If not set, analog weights are
                (unrealistically) copies exactly

        Returns:
            an torch Linear layer with the same dimension and weights
            as the analog linear layer.
        """
        #to get the correct tensor shapes
        weight, bias = module.analog_slices.get_weights()[0], module.analog_slices.get_weights()[1] 

        #loop over slices, multiply factors with weights, add and give as weight/bias
        for idx, weight_slice in enumerate(module.analog_slices):
            slice_weight, slice_bias = weight_slice.get_weights(realistic=realistic)
            weight += slice_weight*module.significance_factors[idx]
            bias += slice_bias*module.significance_factors[idx]

        digital_layer = Linear(module.in_features, module.out_features, bias is not None)
        digital_layer.weight.data = weight
        if bias is not None:
            digital_layer.bias.data = bias
        analog_tile = next(module.analog_tiles())
        return digital_layer.to(device=analog_tile.device, dtype=analog_tile.get_dtype())

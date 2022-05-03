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

"""Analog layers."""

from typing import Optional

from torch import Tensor
from torch.nn import Linear

from aihwkit.nn.functions import AnalogFunction
from aihwkit.nn.modules.base import AnalogModuleBase, RPUConfigAlias
from aihwkit.simulator.configs import SingleRPUConfig


class AnalogLinear(AnalogModuleBase, Linear):
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
        rpu_config: resistive processing unit configuration.
        bias: whether to use a bias row on the analog tile or not.
        realistic_read_write: whether to enable realistic read/write
            for setting initial weights and during reading of the weights.
        weight_scaling_omega: the weight value that the current max
            weight value will be scaled to. If zero, no weight scaling will
            be performed.
    """
    # pylint: disable=abstract-method

    __constants__ = ['in_features', 'out_features', 'realistic_read_write', 'weight_scaling_omega',
                     'digital_bias', 'analog_bias', 'use_bias']
    in_features: int
    out_features: int
    realistic_read_write: bool
    weight_scaling_omega: float
    digital_bias: bool
    analog_bias: bool
    use_bias: bool

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False,
            weight_scaling_omega: Optional[float] = None,
              ):
        # Call super() after tile creation, including ``reset_parameters``.
        Linear.__init__(self, in_features, out_features, bias=bias)

        # Create tile
        if rpu_config is None:
            rpu_config = SingleRPUConfig()

        AnalogModuleBase.__init__(
            self,
            in_features,
            out_features,
            bias,
            realistic_read_write,
            rpu_config.mapping
        )
        self.analog_tile = self._setup_tile(rpu_config)

        # Register tile
        self.register_analog_tile(self.analog_tile)

        # Set weights from the reset_parameters call
        self.set_weights(self.weight, self.bias, remap_weights=True,
                         weight_scaling_omega=weight_scaling_omega)

        # Unregister weight/bias as a parameter but keep it as a
        # field (needed for syncing still)
        self.unregister_parameter('weight')
        if self.analog_bias:
            self.unregister_parameter('bias')

    @classmethod
    def from_digital(
            cls,
            module: Linear,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False,
    ) -> 'AnalogLinear':
        """Return an AnalogLinear layer from a torch Linear layer.

        Args:
            module: The torch module to convert. All layers that are
                defined in the ``conversion_map``.
            rpu_config: RPU config to apply to all converted tiles.
                Applied to all converted tiles.
            realistic_read_write: Whether to use closed-loop programming
                when setting the weights. Applied to all converted tiles.

                Note:
                    Make sure that the weight max and min settings of the
                    device support the desired analog weight range.

        Returns:
            an AnalogLinear layer based on the digital Linear ``module``.
        """
        analog_module = cls(module.in_features,
                            module.out_features,
                            module.bias is not None,
                            rpu_config,
                            realistic_read_write)

        analog_module.set_weights(module.weight, module.bias)
        return analog_module

    def reset_parameters(self) -> None:
        """Reset the parameters (weight and bias)."""
        super().reset_parameters()
        if self.analog_tile_count():
            self.set_weights(self.weight, self.bias)

    def forward(self, x_input: Tensor) -> Tensor:
        """Compute the forward pass."""
        # pylint: disable=arguments-differ, arguments-renamed

        out = AnalogFunction.apply(
                self.analog_tile.get_analog_ctx(), x_input,
                self.analog_tile.shared_weights, not self.training)

        out = self.analog_tile.apply_out_scaling(out, (-1, ))

        if self.digital_bias:
            return out + self.bias
        return out

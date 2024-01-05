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

"""High level analog tiles (floating point)."""

from typing import Optional, Tuple, TYPE_CHECKING

from torch import Tensor

from aihwkit.simulator.rpu_base import tiles
from aihwkit.simulator.tiles.rpucuda import RPUCudaSimulatorTileWrapper
from aihwkit.simulator.tiles.module import TileModule
from aihwkit.simulator.tiles.periphery import TileWithPeriphery
from aihwkit.simulator.tiles.functions import AnalogFunction
from aihwkit.simulator.parameters.base import RPUConfigGeneric

if TYPE_CHECKING:
    from aihwkit.simulator.configs import FloatingPointRPUConfig


class FloatingPointTile(TileModule, TileWithPeriphery, RPUCudaSimulatorTileWrapper):
    r"""Floating point tile.

    Implements a floating point or ideal analog tile.

    A linear layer with this tile is perfectly linear, it just uses
    the RPUCuda library for execution.

    **Forward pass**:

    .. math:: \mathbf{y} = W\mathbf{x}

    :math:`W` are the weights, :math:`\mathbf{x}` is the input
    vector. :math:`\mathbf{y}` is output of the vector matrix
    multiplication.  Note that if bias is used, :math:`\mathbf{x}` is
    concatenated with 1 so that the last column of :math:`W` are the
    biases.


    **Backward pass**:

    Typical backward pass with transposed weights:

    .. math:: \mathbf{d'} = W^T\mathbf{d}

    where :math:`\mathbf{d}` is the error
    vector. :math:`\mathbf{d}_o` is output of the backward matrix
    vector multiplication.


    **Weight update**:

    Usual learning rule for back-propagation:

    .. math:: w_{ij} \leftarrow w_{ij} + \lambda d_i\,x_j


    **Decay**:

    .. math:: w_{ij} \leftarrow w_{ij}(1-\alpha r_\text{decay})

    Weight decay can be called by calling the analog tile decay.

    Note:
        ``life_time`` parameter is set during
        initialization. alpha is a scaling factor that can be given
        during run-time.


    **Diffusion**:

    .. math::  w_{ij} \leftarrow w_{ij} +  \xi\;r_\text{diffusion}

    Similar to the decay, diffusion is only done when explicitly
    called. However, the parameter of the diffusion process are
    set during initialization and are fixed for the
    remainder. :math:`\xi` is a standard Gaussian process.

    Args:
        out_size: output vector size of the tile, ie. the dimension of
            :math:`\mathbf{y}` in case of :math:`\mathbf{y} =
            W\mathbf{x}` (or equivalently the dimension of the
            :math:`\boldsymbol{\delta}` of the backward pass).
        in_size: input vector size, ie. the dimension of the vector
            :math:`\mathbf{x}` in case of :math:`\mathbf{y} =
            W\mathbf{x}`).
        rpu_config: resistive processing unit configuration.
        bias: whether to add a bias column to the tile, ie. :math:`W`
            has an extra column to code the biases. Internally, the
            input :math:`\mathbf{x}` will be automatically expanded by
            an extra dimension which will be set to 1 always.
        in_trans: Whether to assume an transposed input (batch first).
        out_trans: Whether to assume an transposed output (batch first).
    """

    def __init__(
        self,
        out_size: int,
        in_size: int,
        rpu_config: "FloatingPointRPUConfig",
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
    ):
        TileModule.__init__(self)
        RPUCudaSimulatorTileWrapper.__init__(
            self, out_size, in_size, rpu_config, bias, in_trans, out_trans  # type: ignore
        )
        TileWithPeriphery.__init__(self)

    def _create_simulator_tile(
        self, x_size: int, d_size: int, rpu_config: RPUConfigGeneric
    ) -> tiles.FloatingPointTile:
        """Create a simulator tile.


        Args:
            x_size: input size
            d_size: output size
            rpu_config: resistive processing unit configuration

        Returns:
            a simulator tile based on the specified configuration.
        """
        meta_parameter = rpu_config.device.as_bindings(self.get_data_type())
        return meta_parameter.create_array(x_size, d_size)

    def forward(
        self, x_input: Tensor, tensor_view: Optional[Tuple] = None  # type: ignore
    ) -> Tensor:
        """Torch forward function that calls the analog forward"""
        # pylint: disable=arguments-differ

        out = AnalogFunction.apply(
            self.get_analog_ctx(), self, x_input, self.shared_weights, not self.training
        )

        if tensor_view is None:
            tensor_view = self.get_tensor_view(out.dim())
        out = self.apply_out_scaling(out, tensor_view)

        if self.digital_bias:
            return out + self.bias.view(*tensor_view)
        return out

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

"""High level analog tiles (floating point)."""

from copy import deepcopy
from typing import Optional, Union, TYPE_CHECKING

from torch import device as torch_device
from torch.cuda import device as cuda_device

from aihwkit.exceptions import CudaError
from aihwkit.simulator.rpu_base import cuda, tiles
from aihwkit.simulator.tiles.base import BaseTile

if TYPE_CHECKING:
    from aihwkit.simulator.configs import FloatingPointRPUConfig


class FloatingPointTile(BaseTile):
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
            rpu_config: Optional['FloatingPointRPUConfig'] = None,
            bias: bool = False,
            in_trans: bool = False,
            out_trans: bool = False,
    ):
        if not rpu_config:
            # Import `FloatingPointRPUConfig` dynamically to avoid import cycles.
            # pylint: disable=import-outside-toplevel
            from aihwkit.simulator.configs import FloatingPointRPUConfig
            rpu_config = FloatingPointRPUConfig()
        super().__init__(out_size, in_size, rpu_config, bias, in_trans, out_trans)

    def cpu(self) -> 'BaseTile':
        """Return a copy of this tile in CPU memory.

        Note:
            CUDA tiles weight can be accessed by `get_weights` etc
            methods, there is no need to move them to CPU and it is
            currently not supported.

        Returns:
            self in case of CPU

        Raises:
            CudaError: if a CUDA tile is moved to CPU
        """
        if self.is_cuda:
            raise CudaError('Currently it is not possible to move CUDA tile to cpu.')

        return self

    def cuda(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'BaseTile':
        """Return a copy of this tile in CUDA memory.

        Args:
            device: CUDA device

        Returns:
            Self with the underlying C++ tile moved to CUDA memory.

        Raises:
            CudaError: if the library has not been compiled with CUDA.
        """
        if not cuda.is_compiled():
            raise CudaError('aihwkit has not been compiled with CUDA support')

        device = torch_device('cuda', cuda_device(device).idx)

        if self.is_cuda and device != self.device:
            raise CudaError('Cannot switch CUDA devices of existing Cuda tiles')

        if isinstance(self.tile, tiles.FloatingPointTile):
            with cuda_device(device):
                self.tile = tiles.CudaFloatingPointTile(self.tile)
                self.is_cuda = True
                self.device = device
                self.analog_ctx.cuda(device)

        return self

    def _create_simulator_tile(
            self,
            x_size: int,
            d_size: int,
            rpu_config: 'FloatingPointRPUConfig'
    ) -> tiles.FloatingPointTile:
        """Create a simulator tile.

        Args:
            x_size: input size
            d_size: output size
            rpu_config: resistive processing unit configuration

        Returns:
            a simulator tile based on the specified configuration.
        """
        meta_parameter = rpu_config.device.as_bindings()

        return meta_parameter.create_array(x_size, d_size)


class CudaFloatingPointTile(FloatingPointTile):
    """Floating point tile (CUDA).

    Floating point tile that uses GPU for its operation. The instantiation is
    based on an existing non-cuda tile: all the source attributes are copied
    except for the simulator tile, which is recreated using a GPU tile.

    Caution:
        Deprecated. Use ``FloatingPointTile(..).cuda()`` instead.

    Args:
        source_tile: tile to be used as the source of this tile
    """

    is_cuda = True

    def __init__(self, source_tile: FloatingPointTile):
        if not cuda.is_compiled():
            raise CudaError('aihwkit has not been compiled with CUDA support')

        # Create a new instance of the rpu config.
        new_rpu_config = deepcopy(source_tile.rpu_config)

        # Create the tile, replacing the simulator tile.
        super().__init__(source_tile.out_size, source_tile.in_size, new_rpu_config,
                         source_tile.bias, source_tile.in_trans, source_tile.out_trans)
        self.cuda(self.device)

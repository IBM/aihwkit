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

"""High level analog tiles (indexed)."""

from typing import List, Optional, Union

from torch import Tensor
from torch.cuda import current_stream, current_device
from torch.cuda import device as cuda_device
from torch import device as torch_device


from aihwkit.simulator.devices import (
    BaseResistiveDevice,
    FloatingPointResistiveDevice
)
from aihwkit.simulator.tiles import (
    AnalogTile,
    FloatingPointTile
)
from aihwkit.simulator.rpu_base import tiles, cuda


class IndexedFloatingPointTile(FloatingPointTile):
    """Floating point tile (indexed)."""

    def __init__(
            self,
            out_size: int,
            in_size: int,
            resistive_device: Optional[FloatingPointResistiveDevice] = None,
            bias: bool = False,
            in_trans: bool = False,
            out_trans: bool = False):
        super().__init__(out_size, in_size, resistive_device, bias, in_trans, out_trans)
        self.image_sizes = []  # type: List[int]

    def set_indexed(self, indices: Tensor, image_sizes: list) -> None:
        """Sets the index matrix for convolutions ans switches to
        indexed forward/backward/update versions.

        Args:
          indices : torch.tensor with int indices
          image_sizes: [C_in, H_in, W_in, H_out, W_out] sizes
        """
        self.image_sizes = image_sizes

        if len(image_sizes) != 5:
            raise ValueError("Expect 5 sizes [C_in, H_in, W_in, H_out, W_out]!")

        if self.in_trans or self.out_trans:
            raise ValueError("Transposed indexed versions not supported (assumes NCHW)")

        self.tile.set_matrix_indices(indices)

    def forward(self, x_input: Tensor, is_test: bool = False) -> Tensor:
        """Perform the forward pass.

        Args:
            x_input: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
            is_test: whether to assume testing mode.

        Returns:
            torch.Tensor: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.
        """
        # Unset the require_grad of the tensor when chaining.
        if x_input.grad_fn:
            x_input = x_input.detach()

        _, _, _, height_out, width_out = self.image_sizes
        return self.tile.forward_indexed(x_input, height_out, width_out, is_test)

    def backward(self, d_input: Tensor) -> Tensor:
        """Perform the backward pass.

        Args:
            d_input: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.

        Returns:
            torch.Tensor: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
        """
        channel_in, height_in, width_in, _, _ = self.image_sizes
        return self.tile.backward_indexed(d_input, channel_in, height_in, width_in)

    def update(self, x_input: Tensor, d_input: Tensor) -> None:
        """Perform the update pass.

        Args:
            x_input: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
            d_input: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.
        """
        return self.tile.update_indexed(x_input, d_input)

    def cuda(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'CudaIndexedFloatingPointTile':
        """Return a copy of this tile in CUDA memory.

        Args:
            device: CUDA device
        """
        if not cuda.is_compiled():
            raise RuntimeError('aihwkit has not been compiled with CUDA support')

        with cuda_device(device):
            tile = CudaIndexedFloatingPointTile(
                self.out_size, self.in_size, self.resistive_device,
                self.bias, self.in_trans, self.out_trans)
        return tile


class IndexedAnalogTile(AnalogTile):
    """Analog tile (indexed)."""

    def __init__(
            self,
            out_size: int,
            in_size: int,
            resistive_device: Optional[BaseResistiveDevice] = None,
            bias: bool = False,
            in_trans: bool = False,
            out_trans: bool = False):
        super().__init__(out_size, in_size, resistive_device, bias, in_trans, out_trans)
        self.image_sizes = []  # type: List[int]

    def set_indexed(self, indices: Tensor, image_sizes: list) -> None:
        """Sets the index matrix for convolutions ans switches to
        indexed forward/backward/update versions.

        Args:
          indices : torch.tensor with int indices
          image_sizes: [C_in, H_in, W_in, H_out, W_out] sizes
        """
        self.image_sizes = image_sizes

        if len(image_sizes) != 5:
            raise ValueError("Expect 5 sizes [C_in, H_in, W_in, H_out, W_out]!")

        if self.in_trans or self.out_trans:
            raise ValueError("Transposed indexed versions not supported (assumes NCHW)")

        self.tile.set_matrix_indices(indices)

    def forward(self, x_input: Tensor, is_test: bool = False) -> Tensor:
        """Perform the forward pass.

        Args:
            x_input: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
            is_test: whether to assume testing mode.

        Returns:
            torch.Tensor: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.
        """
        # Unset the require_grad of the tensor when chaining.
        if x_input.grad_fn:
            x_input = x_input.detach()

        _, _, _, height_out, width_out = self.image_sizes
        return self.tile.forward_indexed(x_input, height_out, width_out, is_test)

    def backward(self, d_input: Tensor) -> Tensor:
        """Perform the backward pass.

        Args:
            d_input: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.

        Returns:
            torch.Tensor: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
        """
        channel_in, height_in, width_in, _, _ = self.image_sizes
        return self.tile.backward_indexed(d_input, channel_in, height_in, width_in)

    def update(self, x_input: Tensor, d_input: Tensor) -> None:
        """Perform the update pass.

        Args:
            x_input: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
            d_input: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.
        """
        return self.tile.update_indexed(x_input, d_input)

    def cuda(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'CudaIndexedAnalogTile':
        """Return a copy of this tile in CUDA memory.

        Args:
            device: CUDA device
        """
        if not cuda.is_compiled():
            raise RuntimeError('aihwkit has not been compiled with CUDA support')

        with cuda_device(device):
            tile = CudaIndexedAnalogTile(
                self.out_size, self.in_size, self.resistive_device,
                self.bias, self.in_trans, self.out_trans)
        return tile


class CudaIndexedFloatingPointTile(IndexedFloatingPointTile):
    """Floating point tile (CUDA, indexed)."""

    is_cuda = True

    def __init__(
            self,
            out_size: int,
            in_size: int,
            resistive_device: Optional[FloatingPointResistiveDevice] = None,
            bias: bool = False,
            in_trans: bool = False,
            out_trans: bool = False):
        if not cuda.is_compiled():
            raise RuntimeError('aihwkit has not been compiled with CUDA support')

        super().__init__(out_size, in_size, resistive_device, bias, in_trans, out_trans)

        self.tile = tiles.CudaFloatingPointTile(self.tile)
        self.stream = current_stream()
        self.device = torch_device(current_device())

    def cuda(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'CudaIndexedFloatingPointTile':
        if self.stream != current_stream(device):
            raise ValueError("Cannot switch devices of existing Cuda tiles")

        return self


class CudaIndexedAnalogTile(IndexedAnalogTile):
    """Analog tile (CUDA, indexed)."""

    is_cuda = True

    def __init__(
            self,
            out_size: int,
            in_size: int,
            resistive_device: Optional[BaseResistiveDevice] = None,
            bias: bool = False,
            in_trans: bool = False,
            out_trans: bool = False):
        if not cuda.is_compiled():
            raise RuntimeError('aihwkit has not been compiled with CUDA support')
        super().__init__(out_size, in_size, resistive_device, bias, in_trans, out_trans)

        self.tile = tiles.CudaAnalogTile(self.tile)
        self.stream = current_stream()
        self.device = torch_device(current_device())

    def cuda(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'CudaIndexedAnalogTile':
        if self.stream != current_stream(device):
            raise ValueError("Cannot switch CUDA devices of existing Cuda tiles")

        return self

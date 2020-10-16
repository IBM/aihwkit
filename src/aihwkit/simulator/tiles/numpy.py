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

# type: ignore

"""High level analog tiles (numpy)."""

from numpy import array
from torch import Tensor, from_numpy

from aihwkit.simulator.tiles import AnalogTile, FloatingPointTile


class NumpyMixin:
    """Helper for overriding forward, backward and update methods."""

    def move_weights(self) -> Tensor:
        """Move the tile weights to an external variable.

        Move the weights currently stored in the tile to an externally
        managed variable (so the internal references to the weights use the same
        memory space as the returned array data).

        Caution:

            This is an advanced method, and assumes that the caller will be
            responsible for managing the shared weights variable manually.
            After calling this function, the tile will use the same pointer for
            the internal weights: freeing the variable or modifying it manually
            can result in several side effects.  Additionally, all tensor
            functions applied to the external weights MUST be in-place or errors
            will occur.

        Caution:

            Since the internal memory layout is transposed for CUDA the weight
            matrix is transposed as well.

        Returns:
            Tensor: a ``[out_size, in_size (+ 1)]`` contiguous array that
                contains the weights that uses the same memory addresses as the
                internal pointer to the weights.
        """
        shared_weights = from_numpy(self.tile.get_weights())
        if self.is_cuda:
            shared_weights.cuda()
        self.tile.set_shared_weights(shared_weights)
        return shared_weights

    def forward(self, x_input: array, is_test: bool = False) -> array:
        """Perform the forward pass.

        Args:
            x_input: ``[N, in_size]`` matrix. If ``in_trans`` is set, transposed.
            is_test: Whether to assume testing mode

        Returns:
            array: ``[N, out_size]`` matrix. If ``out_trans`` is set, transposed.
        """
        if self.is_cuda:
            x_tensor = from_numpy(x_input.astype('float32')).cuda()
            return self.tile.forward(x_tensor, self.bias,
                                     self.in_trans, self.out_trans,
                                     is_test).cpu().numpy()

        return self.tile.forward_numpy(x_input, self.bias,
                                       self.in_trans, self.out_trans,
                                       is_test)

    def backward(self, d_input: array) -> array:
        """Perform the backward pass.

        Args:
            d_input: ``[N, out_size]`` matrix. If ``out_trans`` is set, transposed.

        Returns:
            array: ``[N, in_size]`` matrix. If ``in_trans`` is set, transposed.
        """
        if self.is_cuda:
            d_tensor = from_numpy(d_input.astype('float32')).cuda()
            return self.tile.backward(d_tensor, self.bias,
                                      self.out_trans, self.in_trans).cpu().numpy()

        return self.tile.backward_numpy(d_input, self.bias,
                                        self.out_trans, self.in_trans)

    def update(self, x_input: array, d_input: array) -> None:
        """Perform the update pass.

        Args:
            x_input: ``[N, in_size]`` matrix. If ``in_trans`` is set, transposed.
            d_input: ``[N, out_size]`` matrix. If ``out_trans`` is set, transposed.
        """
        if self.is_cuda:
            x_tensor = from_numpy(x_input.astype('float32')).cuda()
            d_tensor = from_numpy(d_input.astype('float32')).cuda()
            return self.tile.update(x_tensor, d_tensor, self.bias,
                                    self.in_trans, self.out_trans)

        return self.tile.update_numpy(x_input, d_input, self.bias,
                                      self.in_trans, self.out_trans)


class NumpyFloatingPointTile(NumpyMixin, FloatingPointTile):
    """Floating point tile (numpy)."""


class NumpyAnalogTile(NumpyMixin, AnalogTile):
    """Analog tile (numpy)."""

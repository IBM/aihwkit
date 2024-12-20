# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Global drift compensation for inference."""

from torch.autograd import no_grad
from torch import abs as torch_abs
from torch import clamp, Tensor, eye

from aihwkit.inference.compensation.base import BaseDriftCompensation


class GlobalDriftCompensation(BaseDriftCompensation):
    """Global drift compensation.

    Uses a constant factor for compensating the drift.
    """

    @no_grad()
    def readout(self, out_tensor: Tensor) -> Tensor:
        """Read outs the mean abs."""
        return clamp(torch_abs(out_tensor).mean(), min=0.0001)

    @no_grad()
    def get_readout_tensor(self, in_size: int) -> Tensor:
        """Return the read-out tensor.

        Uses the set of one-hot vectors (eye).
        """
        return eye(in_size)

    def __str__(self) -> str:
        return "{}()".format(self.__class__.__name__)


class PerColumnDriftCompensation(BaseDriftCompensation):
    """Per column drift compensation.
    Uses a vector for compensating the drift.
    """

    @no_grad()
    def readout(self, out_tensor: Tensor) -> Tensor:
        """Read outs the per-column mean abs."""
        return clamp(torch_abs(out_tensor).mean(dim=0), min=0.0001)

    @no_grad()
    def get_readout_tensor(self, in_size: int) -> Tensor:
        """Return the read-out tensor.
        Uses the set of one-hot vectors (eye).
        """
        return eye(in_size)

    def __str__(self) -> str:
        return "{}()".format(self.__class__.__name__)

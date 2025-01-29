# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Global drift compensation for inference."""

from typing import Tuple

from torch.autograd import no_grad
from torch import abs as torch_abs
from torch import clamp, Tensor, eye

from aihwkit.inference.compensation.base import BaseDriftCompensation
from aihwkit.simulator.tiles.inference import InferenceTileWithPeriphery


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


class GlobalDriftCompensationWithExactReference(GlobalDriftCompensation):
    """Global drift compensation using an exact (ideal) reference readout.

    Uses a constant factor for compensating the drift.
    """

    @no_grad()
    def init_baseline(self, tile: InferenceTileWithPeriphery) -> Tuple[Tensor, Tensor]:
        """Initialize the base line for applying the compensation.

        Uses a all one tensor for read_out.

        Args:
            tile: forward output of the read out vector to compensate

        Returns:
            reference tensor readout
        """
        forward_output = tile._forward_drift_readout_tensor(True, exact_reference=True)
        ref_value = self.readout(forward_output)

        return ref_value


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

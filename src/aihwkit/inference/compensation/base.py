# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Base drift compensation for inference."""

from typing import Tuple

from torch import Tensor
from torch.autograd import no_grad


class BaseDriftCompensation:
    """Base class for drift compensations."""

    def __init__(self) -> None:
        pass

    @no_grad()
    def init_baseline(self, forward_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Initialize the base line for applying the compensation.

        Uses a all one tensor for read_out.

        Args:
            forward_output: forward output of the read out vector to compensate

        Returns:
            reference tensor readout
        """
        ref_value = self.readout(forward_output)

        return ref_value

    @no_grad()
    def get_readout_tensor(self, in_size: int) -> Tensor:
        """Return the read-out tensor.

        Called once during :meth:`~init_baseline`.
        """
        raise NotImplementedError

    @no_grad()
    def readout(self, out_tensor: Tensor) -> Tensor:
        """Implement the read out math."""
        raise NotImplementedError

    @no_grad()
    def apply(self, forward_output: Tensor, ref_value: Tensor) -> Tensor:
        """Read out the current value from the output of the forward
        pass and returns the drift compensation alpha scale."""
        current_value = self.readout(forward_output)
        ratio = ref_value / current_value

        return ratio

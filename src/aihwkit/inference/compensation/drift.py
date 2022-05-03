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

"""Global drift compensation for inference."""

from torch.autograd import no_grad
from torch import abs as torch_abs
from torch import clamp, ones, Tensor

from aihwkit.inference.compensation.base import BaseDriftCompensation


class GlobalDriftCompensation(BaseDriftCompensation):
    """Global drift compensation.

    Uses a constant factor for compensating the drift.
    """

    @no_grad()
    def readout(self, out_tensor: Tensor) -> Tensor:
        """Read outs the abs max."""
        return clamp(torch_abs(out_tensor).max(), min=0.0001)

    @no_grad()
    def get_readout_tensor(self, in_size: int) -> Tensor:
        """Return the read-out tensor.

        Uses a single all one vector.
        """
        return ones((1, in_size))

    def __str__(self) -> str:
        return '{}()'.format(self.__class__.__name__)

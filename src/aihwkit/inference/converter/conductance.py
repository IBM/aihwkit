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

"""Conductance converters for the phenomenological noise models for inference."""

from typing import Dict, List, Optional, Tuple

from torch import abs as torch_abs  # pylint: disable=unused-import
from torch import Tensor
from torch.autograd import no_grad

from aihwkit.inference.converter.base import BaseConductanceConverter

_ZERO_CLIP = 1e-7


class SinglePairConductanceConverter(BaseConductanceConverter):
    r"""Single pair of devices.

    Assuming a single pair of devices per cross-point, taking positive
    and negative weights, respectively, where one device is always at
    0.

    Args:
        g_max: In :math:`\mu S`, the maximal conductance, ie the value
            the absolute max of the weights will be mapped to.
        g_min: In :math:`\mu S`, the minimal conductance, ie the value
            the logical zero of the weights will be mapped to.
    """

    def __init__(self, g_max: Optional[float] = None, g_min: Optional[float] = None):
        self.g_max = 25.0 if g_max is None else g_max
        self.g_min = 0.0 if g_min is None else g_min
        self.scale_ratio = None

        if self.g_max < 0.0:
            raise ValueError("g_max should be a positive value")
        if self.g_min < 0.0:
            raise ValueError("g_min should be a positive value")
        if self.g_min >= self.g_max:
            raise ValueError("g_min should be smaller than g_max")

    def __str__(self) -> str:
        return "{}(g_max={:1.2f}, g_min={:1.2f})".format(
            self.__class__.__name__, self.g_max, self.g_min
        )

    @no_grad()
    def convert_to_conductances(self, weights: Tensor) -> Tuple[List[Tensor], Dict]:
        abs_max = torch_abs(weights).max()
        scale_ratio = (self.g_max - self.g_min) / abs_max.clamp(min=_ZERO_CLIP)
        scaled_weights = weights * scale_ratio

        conductances = [
            scaled_weights.clamp(min=0.0, max=self.g_max) + self.g_min,
            (-scaled_weights).clamp(min=0.0, max=self.g_max) + self.g_min,
        ]
        params = {"scale_ratio": scale_ratio}

        return conductances, params

    @no_grad()
    def convert_back_to_weights(self, conductances: List[Tensor], params: Dict) -> Tensor:
        if len(conductances) != 2:
            raise ValueError("conductances must contain exactly two elements")
        if "scale_ratio" not in params:
            raise ValueError("params do not contain scale_ratio")

        weights = ((conductances[0] - self.g_min) - (conductances[1] - self.g_min)) / params[
            "scale_ratio"
        ]

        return weights

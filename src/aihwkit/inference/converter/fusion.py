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

"""Conductance converter for the fusion chip."""

from typing import Dict, List, Tuple

from torch import abs as torch_abs
from torch import Tensor, sign
from torch.autograd import no_grad

from aihwkit.inference.converter.base import BaseConductanceConverter

_ZERO_CLIP = 1e-7


class FusionConductanceConverter(BaseConductanceConverter):
    r"""Fusion chip conductance converter

    Here a single pair of devices per cross-point is assumed, taking
    positive and negative weights, respectively, where one device is
    always at 0. However, the zero device will not be explicitly
    programmed, as the reset conductance value is assumed to be
    neglegible.

    """

    def __init__(self) -> None:
        self.g_max = 45.0  # fixed for fusion
        self.scale_ratio = None

    def __str__(self) -> str:
        return "{}(g_max={:1.2f})".format(self.__class__.__name__, self.g_max)

    @no_grad()
    def convert_to_conductances(self, weights: Tensor) -> Tuple[List[Tensor], Dict]:
        abs_max = torch_abs(weights).max()
        scale_ratio = self.g_max / abs_max.clamp(min=_ZERO_CLIP)
        scaled_weights = weights * scale_ratio
        sign_weights = sign(scaled_weights)
        conductances = [abs(scaled_weights).clamp(min=0.0, max=self.g_max)]
        params = {"scale_ratio": scale_ratio, "sign_weights": sign_weights}

        return conductances, params

    @no_grad()
    def convert_back_to_weights(self, conductances: List[Tensor], params: Dict) -> Tensor:
        if len(conductances) != 1:
            raise ValueError("conductances must contain exactly 1 element")
        if "scale_ratio" not in params:
            raise ValueError("params do not contain scale_ratio")
        if "sign_weights" not in params:
            raise ValueError("params do not contain sign_weights")

        weights = (params["sign_weights"] * conductances[0]) / params["scale_ratio"]
        return weights

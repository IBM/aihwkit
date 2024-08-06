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

from torch import Tensor, sign
from torch.autograd import no_grad
from sklearn.linear_model import LinearRegression

from aihwkit.exceptions import ArgumentError
from aihwkit.inference.converter.base import BaseConductanceConverter

_ZERO_CLIP = 1e-10


class FusionConductanceConverter(BaseConductanceConverter):
    r"""Fusion chip conductance converter

    Here a single pair of devices per cross-point is assumed, taking
    positive and negative weights, respectively, where one device is
    always at 0. However, the zero device will not be explicitly
    programmed, as the reset conductance value is assumed to be
    negligible.

    Args:
        g_max: In :math:`\mu S`, the maximal conductance, ie the value
            the absolute max of the weights will be mapped to.
            The valid g_max value fusion chip is between 10 and 40.
            If no value is specified, the default value 40 is used.

    Returns:
        None

    Raises:
        ArgumentError: in case g_max is not in the range of 10 to 40.

    """

    def __init__(self, g_max: int = 40) -> None:
        if g_max < 10 or g_max > 40:
            raise ArgumentError(
                "Specified fusion g_max value must be between 10 and 40."
            )
        self.g_max = g_max
        self.scale_ratio = None

    def __str__(self) -> str:
        return "{}(g_max={:1.2f})".format(self.__class__.__name__, self.g_max)

    @no_grad()
    def convert_to_conductances(self, weights: Tensor) -> Tuple[List[Tensor], Dict]:
        abs_max_per_col = weights.abs().max(dim=1).values
        scale_ratio_per_col = self.g_max / abs_max_per_col.clamp(min=_ZERO_CLIP)
        scaled_weights = weights * scale_ratio_per_col.unsqueeze(1)
        conductances = [abs(scaled_weights).clamp(min=0.0, max=self.g_max)]
        params = {
            "original_weights": weights,
        }
        return conductances, params

    @no_grad()
    def convert_back_to_weights(
        self, conductances: List[Tensor], params: Dict
    ) -> Tensor:
        if len(conductances) != 1:
            raise ValueError("conductances must contain exactly 1 element")
        if "original_weights" not in params:
            raise ValueError("params do not contain original_weights")

        weights = conductances[0] * sign(params['original_weights'])
        for col in range(weights.shape[0]):
            reg = LinearRegression().fit(weights[col].flatten().numpy().reshape(-1, 1), params[
                'original_weights'][col].flatten().numpy().reshape(-1, 1))
            weights[col] *= reg.coef_[0][0]

        return weights

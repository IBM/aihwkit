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

# pylint: disable=too-many-instance-attributes

"""Phenomenological noise models for ReRAM devices for inference."""

from copy import deepcopy
from typing import List, Optional, Dict

from torch import randn_like, Tensor
from torch.autograd import no_grad

from aihwkit.exceptions import ArgumentError
from aihwkit.inference.noise.base import BaseNoiseModel
from aihwkit.inference.converter.base import BaseConductanceConverter
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter


class ReRamWan2022NoiseModel(BaseNoiseModel):
    r"""Noise model that was inferred from ReRam publication data.

    This ReRam model is and approximation to the data published by
    `Wan et al. Nature (2022)`_.

    Conductance dependence of the deviations from the target
    conductance was estimated from the published figures and fitted
    with a 4-th order polynomial (only 1 sec, 1 day, 2 day).

    No separate data is available for read noise (1/f).

    Note:

        To account for short-term read noise (about 1\%) one should
        additional set the ``forward.w_noise`` parameter to about 0.01
        (with w_noise_type=WeightNoiseType.ADDITIVE_CONSTANT)

    Args:

        coeff_dic: polynomial coefficients in :math:`\mu S`,
            :math:`\sum_i c_i \left(\frac{g_t}{g_\max}\right)^i` for
            each time. If not given, the fitted measurement is taken
            at selected time points only

        g_converter: Instantiated class of the conductance converter
            (defaults to single pair).
        g_max: In :math:`\mu S`, the maximal conductance, i.e. the value
            the absolute max of the weights will be mapped to.
        noise_scale: Additional scale for the noise.
        coeff_g_max_reference: reference :math:`g_\max` value
            when fitting the coefficients, since the result of the
            polynomial fit is given in uS. If
            ``coeff_g_max_reference`` is not given and
            `coeffs` are given explicitely, it will be set to
            ``g_max`` of the conductance converter.

    .. _`Wan et al. Nature (2022)`: https://www.nature.com/articles/s41586-022-04992-8

    """

    def __init__(
        self,
        coeff_dic: Optional[Dict[float, List]] = None,
        g_converter: Optional[BaseConductanceConverter] = None,
        g_max: Optional[float] = None,
        noise_scale: float = 1.0,
        coeff_g_max_reference: Optional[float] = None,
    ):
        g_converter = deepcopy(g_converter) or SinglePairConductanceConverter(g_max=g_max)
        super().__init__(g_converter)

        self.g_max = getattr(self.g_converter, "g_max", g_max)
        if self.g_max is None:
            raise ValueError("g_max cannot be established from g_converter")

        if coeff_g_max_reference is None:
            self.coeff_g_max_reference = self.g_max

        if coeff_dic is None:
            # standard g_max are defined in respect to 40.0 uS. Need to
            # adjust for that in case g_max is not equal to 40.0 uS

            coeff_dic = {
                1.0: [-16.815, 45.393, -43.853, 16.030, 0.348][::-1],
                3600 * 24.0: [-16.458, 47.095, -50.773, 22.086, 0.701][::-1],
                3600 * 24.0 * 2: [-11.934, 37.062, -43.507, 20.274, 0.782][::-1],
            }
            self.prog_coeff_g_max_reference = 40.0
        self.coeff_dic = coeff_dic
        self.noise_scale = noise_scale

    def _apply_poly(self, g_target: Tensor, coeff: List, scale: float = 1.0) -> Tensor:
        """Applied polynomial noise"""

        mat = 1
        sig_prog = coeff[0]
        for value in coeff[1:]:
            mat *= g_target / self.g_max
            sig_prog += mat * value

        sig_prog *= self.g_max / self.coeff_g_max_reference  # type: ignore
        g_prog = g_target + scale * sig_prog * randn_like(g_target)
        g_prog.clamp_(min=0.0)  # no negative conductances allowed

        return g_prog

    @no_grad()
    def apply_programming_noise_to_conductance(self, g_target: Tensor) -> Tensor:
        """Apply programming noise to a target conductance Tensor.

        Programming noise with additive Gaussian noise with
        conductance dependency of the variance given by a 2-degree
        polynomial.
        """

        min_key = min(list(self.coeff_dic.keys()))
        return self._apply_poly(g_target, self.coeff_dic[min_key], self.noise_scale)

    @no_grad()
    def generate_drift_coefficients(self, g_target: Tensor) -> Tensor:
        """Return target values as coefficients.

        Since ReRAM does not show drift in the usual sense, here
        simply the target values will given as coefficients to compute
        the long-term variations on-the-fly

        """
        return g_target

    @no_grad()
    def apply_drift_noise_to_conductance(
        self, g_prog: Tensor, g_target: Tensor, t_inference: float
    ) -> Tensor:
        """Apply the accumulated noise according to the time of inference.

        Will use unique 4th-order polynomial fit to the ReRAM
        measurements to the target values.

        Args:
            g_prog: will be ignored
            g_target: target conductance values that will be used to add noise
            t_inference: time of inference.

        Returns:
            conductances with noise applied

        Raises:
            ArgumentError: if `t_inference` is not one of
                ``(1, 24*3600, 2*24*3600)`` seconds (or any user-defined
                key in ``coeff_dic``), the error will be raised.
        """
        # pylint: disable=arguments-renamed

        if t_inference not in self.coeff_dic:
            raise ArgumentError(f"t_inference should be one of `{list(self.coeff_dic.keys())}`")

        g_final = self._apply_poly(g_target, self.coeff_dic[t_inference], self.noise_scale)

        return g_final.clamp(min=0.0)

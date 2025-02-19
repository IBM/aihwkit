# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=too-many-instance-attributes

"""Phenomenological noise models for ReRAM devices for inference."""

from copy import deepcopy
from typing import List, Optional, Dict

from torch import randn_like, Tensor
from torch.autograd import no_grad
from numpy import log, log10, sqrt
from aihwkit.exceptions import ArgumentError
from aihwkit.inference.noise.base import BaseNoiseModel
from aihwkit.inference.converter.base import BaseConductanceConverter
from aihwkit.inference.converter.conductance import (
    SinglePairConductanceConverter,
    SingleDeviceConductanceConverter,
)


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


class ReRamCMONoiseModel(BaseNoiseModel):
    r"""Noise model inferred from Analog Filamentary Conductive-Metal-Oxide
    (CMO)/HfOx ReRAM devices from IBM Research Europe - Zurich.

    This noise model is estimated from statistical characterization of CMO/HfOx devices from
    Falcone et al. (In Review)

    Programming noise:
        Described by a linear function with respect to the G target.
        Coefficients are considered for two acceptance ranges, 0.2% and 2% of target conductance

    Conductance Decay:
        Drift in CMO/HfOx devices showed independence of the target conductance value.
        Mean and STD of the conductance distribution were fitted with 1st-order polynomial
        as a function of the log(t) where t is the time of inference

    TODO:
    Read noise (1/f) characterization of CMO/HfO<sub>x</sub> available at Lombardo et al. DRC
    (2024) but not implemented.

    Note:

        To account for short-term read noise (about 1\%) one should
        additional set the ``forward.w_noise`` parameter to about 0.01
        (with w_noise_type=WeightNoiseType.ADDITIVE_CONSTANT)

    Args:
        coeff_dict:  acceptance range with coefficients for the programming noise in :math:`\mu S`,
        g_converter: Instantiated class of the conductance converter for a single device per
        cross-point.
        g_max: In :math:`\mu S`, the maximal conductance, i.e. the value the absolute max of
        the weights will be mapped to.
        g_min: In :math:`\mu S`, the minimal conductance, i.e. the value the absolute min of
        the weights will be mapped to.
        prog_noise_scale: Scale for the programming noise.
        read_noise_scale: Scale for the read and accumulated noise.
        drift_scale: Scale for the  drift coefficient.
        since the result of the polynomial fit is given in uS.
        decay_dict: mean and std coefficients for the drift noise in :math:`\mu S`,

    """

    def __init__(
        self,
        coeff_dict: Optional[Dict[float, List]] = None,
        g_max: Optional[float] = None,
        g_min: Optional[float] = None,
        prog_noise_scale: float = 1.0,
        read_noise_scale: float = 1.0,
        drift_scale: float = 1.0,
        decay_dict: Optional[Dict[str, List]] = None,
        read_dict: Optional[Dict[str, float]] = None,
        acceptance_range: float = 2e-2,
    ):
        g_converter = SingleDeviceConductanceConverter(
            g_max=g_max, g_min=g_min
        )
        super().__init__(g_converter)
        g_max = getattr(self.g_converter, "g_max", g_max)
        g_min = getattr(self.g_converter, "g_min", g_min)
        if g_max is None:
            raise ValueError("g_max cannot be established from g_converter")
        if g_min is None:
            raise ValueError("g_min cannot be established from g_converter")
        self.g_max = g_max
        self.g_min = g_min
        self.coeff_g_max_reference = self.g_max
        if coeff_dict is None:
            coeff_dict = {
                0.2: [0.00106879, 0.00081107][::-1],
                2: [0.01129027418, 0.0112185391][::-1],
            }
        if read_dict is None:
            read_dict = {
                "K" : 0.0277316483,
                "t_read" : 1e-6
            }
        if decay_dict is None:
            decay_dict = {
                "mean": [-0.08900206],
                "std": [0.04201137, 0.41183342],
            }
        if acceptance_range not in coeff_dict.keys():
            acceptance_range = min(coeff_dict.keys())
        self.coeff_dict = coeff_dict
        self.prog_noise_scale = prog_noise_scale
        self.read_noise_scale = read_noise_scale
        self.drift_scale = drift_scale
        self.decay_dict = decay_dict
        self.read_dict = read_dict
        self.acceptance_range = acceptance_range

    def _apply_poly(self, g_target: Tensor, coeff: List, scale: float = 1.0) -> Tensor:
        """Applied polynomial noise"""
        mat = 1
        sig_prog = coeff[0]
        for value in coeff[1:]:
            mat *= g_target  # / self.g_max
            sig_prog += mat * value
        sig_prog *= self.g_max / self.coeff_g_max_reference
        g_prog = g_target + sig_prog * randn_like(g_target) * scale
        return g_prog

    @no_grad()
    def apply_programming_noise_to_conductance(self, g_target: Tensor) -> Tensor:
        """Apply programming noise to a target conductance Tensor.

        Programming noise with additive Gaussian noise with
        conductance dependency of the variance given by a 1st-degree
        polynomial.
        Depends of the acceptance range of the program-and-verify loop
        """

        min_key = (
            self.acceptance_range
            if self.acceptance_range in self.coeff_dict.keys()
            else min(list(self.coeff_dict.keys()))
        )
        return self._apply_poly(g_target, self.coeff_dict[min_key], self.prog_noise_scale)

    @no_grad()
    def generate_drift_coefficients(self, g_target: Tensor) -> Tensor:
        """Conductance relaxation is independent of the conductance level
        """
        return g_target

    @no_grad()
    def apply_drift_noise_to_conductance(
        self, g_prog: Tensor, drift_noise_param: Tensor, t_inference: float
    ) -> Tensor:
        """Apply the accumulated noise according to the time of inference.

        Will use unique 1st-order polynomial fits the conductance mean shift
        and standard deviation shift from the ReRAM

        Args:
            g_prog: target conductance values that will be used to add noise
            drift_noise_param: ccoefficients of the mean and std drift
            t_inference: time of inference. Times in seconds

        Returns:
            conductances with noise applied

        """
        if t_inference == 0:
            return g_prog

        g_mean = g_prog + (self.decay_dict["mean"][0] * log(t_inference) * self.drift_scale)

        sigma_relaxation = (
            self.decay_dict["std"][0] * log(t_inference) + self.decay_dict["std"][1]
        )
        g_drift = g_mean + (randn_like(g_prog) * sigma_relaxation * self.drift_scale)
        sigma_read = self.read_dict["K"] * log10(g_drift) * sqrt(
            log((t_inference + self.read_dict["t_read"]) / (2 * self.read_dict["t_read"])))
        g_final = g_drift + sigma_read * randn_like(g_prog) * self.read_noise_scale

        return g_final.clamp(min=self.g_min, max=self.g_max)

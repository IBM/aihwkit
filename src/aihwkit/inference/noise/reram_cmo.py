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
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter, SingleDeviceConductanceConverter
import numpy
import torch
#torch.random.manual_seed(0)
class ReRamCMONoiseModel(BaseNoiseModel):
    r"""Noise model that was inferred from ReRam publication data.

    This noise model is estimated from experimental data of the MO/HfOx devices from Falcone et al.
    Conductance dependence of the deviations from the target (the mean of the conductance is considered as the target)
    the response is described by a 1st-order polynomial for 1s after programming
    Data for de day of the mean and std of the target is also available

    TODO:
    Read noise (1/f) characterization available but not implemented.

    Note:

        To account for short-term read noise (about 1\%) one should
        additional set the ``forward.w_noise`` parameter to about 0.01
        (with w_noise_type=WeightNoiseType.ADDITIVE_CONSTANT)

    Args:

        coeff_dic: the two coefficients at t=0,1 of the programming noise
        g_max: In :math:`\mu S`, the maximal conductance, i.e. the value
            the absolute max of the weights will be mapped to.
        noise_scale: Additional scale for the noise.
    """

    def __init__(
        self,
        coeff_dic: Optional[Dict[float, List]] = None,
        g_converter: Optional[BaseConductanceConverter] = None,
        g_max: Optional[float] = 88.19,
        g_min: Optional[float] = 9.0,
        noise_scale: float = 1.0,
        coeff_g_max_reference: Optional[float] = None,
        decay_dict: Optional[Dict[float, List]] = None,
        reference_drift: Optional[float] = None,
        tol : float = 2e-2
    ):
        g_converter = SingleDeviceConductanceConverter(g_max=g_max)
        super().__init__(g_converter)
        self.g_max = getattr(self.g_converter, "g_max", g_max)
        self.g_min = getattr(self.g_converter, "g_min", g_min)
        if self.g_max is None:
            raise ValueError("g_max cannot be established from g_converter")
        if coeff_g_max_reference is None:
            self.coeff_g_max_reference = self.g_max
        if coeff_dic is None:
            # standard g_max are defined in respect to 40.0 uS. Need to
            # adjust for that in case g_max is not equal to 40.0 uS
            coeff_dic = {
                0.2: [0.00106879, 0.00081107][::-1],
                2: [0.01129027418 ,0.0112185391][::-1]
            }
        if decay_dict is None:
            decay_dict = {
                'mean' : [-0.08900206, 49.92383444],#[::-1],
                'std': [0.04201137, 0.41183342],#[::-1],
            }
        if reference_drift is None:
            self.reference_drift = 50.0
            self.prog_coeff_g_max_reference = 88.199997
            #self.prog_coeff_g_max_reference = 50.0
        self.coeff_dic = coeff_dic
        self.noise_scale = noise_scale
        self.decay_dict = decay_dict
        self.tolerance = tol

    def _apply_poly(self, g_target: Tensor, coeff: List, scale: float = 1.0, sigma_relaxation : float = 0.0) -> Tensor:
        """Applied polynomial noise"""
        mat = 1
        sig_prog = coeff[0]
        for value in coeff[1:]:
            mat *= g_target #/ self.g_max
            sig_prog += mat * value

        sig_prog *= self.g_max / self.coeff_g_max_reference 
       # sig_prog *= randn_like(g_target)
        #sig_prog += sigma_relaxation
        g_prog = g_target + sig_prog * randn_like(g_target)
        #g_prog.clamp_(min=0.0)  # no negative conductances allowed

        return g_prog#, sig_prog

    @no_grad()
    def apply_programming_noise_to_conductance(self, g_target: Tensor) -> Tensor:
        """Apply programming noise to a target conductance Tensor.

        Programming noise with additive Gaussian noise with
        conductance dependency of the variance given by a 1-degree
        polynomial.
        """

        #min_key = min(list(self.coeff_dic.keys()))
        min_key = self.tolerance if self.tolerance in self.coeff_dic.keys() else min(list(self.coeff_dic.keys()))
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
        self, g_target: Tensor, drift_noise_param, t_inference: float
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
        #apply mean relaxation to gT
        if t_inference == 0:
            g_final = self._apply_poly(g_target, self.coeff_dic[self.tolerance], self.noise_scale, sigma_relaxation=0.0)
            #print("G target", g_target)
            #print("G prog", g_final)
            return g_final.clamp(min=self.g_min)
        g_mean = self.decay_dict['mean'][0]*numpy.log(t_inference) + (self.decay_dict['mean'][1]*g_target/self.reference_drift)
        #print("G mean", g_mean)
        #print((self.decay_dict['mean'][1]*g_target/self.reference_drift))
        sigma_relaxation = self.decay_dict['std'][0]*numpy.log(t_inference) + self.decay_dict['std'][1]
        g_final = g_mean + randn_like(g_target) * sigma_relaxation
        #print("G target", g_target)
        #print("G prog", g_mean)
        return g_final.clamp(min=self.g_min)

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

"""Phenomenological noise model for inference."""

from copy import deepcopy
from typing import List, Optional

from numpy import log as numpy_log
from numpy import sqrt
from torch import abs as torch_abs
from torch import randn_like, Tensor
from torch.autograd import no_grad

from aihwkit.inference.noise.base import BaseNoiseModel
from aihwkit.inference.converter.base import BaseConductanceConverter
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter


class StateIndependentNoiseModel(BaseNoiseModel):  # pylint: disable=too-many-instance-attributes
    r"""Standard noise model that has a non-conductance dependent drift and
     multiplicative read (1/f) noise.

     Programming noise is state-independent by default, however, it can
     be made conductance dependent, since the expected programming
     noise strength is modeled with a second-order polynomial in
     general.

    **Programming noise** is thus given by:

     .. math::

          \sigma_text{programming noise}=\gamma\,\left(c_0 + c_1 \frac{g_T}{g_\text{max}} +
               + c_2 \frac{g_T^2}{g_\text{max}^2}\right)

     where :math:`\gamma` is a additional convenience scale and :math:`g_T`
     is the target conductance established from the given
     ``g_converter`` from the weight matrix.  The default programming
     noise is constant (state independent): :math:`c_0=0.2\mu\mathrm{S}`
     and other coefficient set to :math:`0.0`.

     **Drift** is for each device is computed as

     .. math::

         g_\text{drift}(t) = g_\text{prog}(t / t_0) ^{- \nu}

     with the drift coefficient determined at the beginning for each
     device with

     .. math::

         \nu= \zeta\, |\nu_\text{mean} + \nu_\text{std}\xi|_+

     where :math:`\xi` is a Gaussian random number and
     :math:`|\cdot|_+` rectifies negative value to zero. :math:`\zeta`
     is an additional drift scale.

     **Read noise** is given by

     .. math::

         \sigma_\text{read} = \rho \frac{g_\text{drift}(t)}{g_\text{max}} \sqrt{\log\left(\frac{t
         + t_\text{read}}{2 t_\text{read}}\right)}

     This :math:`\sigma_\text{read}` is then used to add Gaussian noise
     of this magnitude to the drifted conductance. The read noise scale
     :math:`\rho` can be used to scale the read noise.

     Args:
         g_converter: instantiated class of the conductance converter
             (defaults to single pair)

         g_max: In :math:`\mu S`, the maximal conductance, ie the value
             the absolute max of the weights will be mapped to.

         prog_coeff: programming polynomial coefficients :math:`c_i` in
             :math:`\mu S`. Default is constant :math:`c_0=0.2` and
             other coefficient set to 0.0.

         prog_noise_scale: scale :math:\gamma: for the programming noise

         drift_nu_mean: mean :math:`\nu_\text{mean}` of power-law drift
             coefficient (:math:`\nu`) (before ``drift_scale``
             :math:`\zeta` is applied).

         drift_nu_std: device-to-device variability
             :math:`\nu_\text{std}` of the power-law drift coefficient
             (before ``drift_scale`` is applied)

         drift_scale: additional scale :math:`\zeta` applied to all
             drawn drift coefficients

         t_0: parameter of the drift (first reading time), see above.

             Note:
                 The ``t_inference`` is relative to this time ``t0``
                 e.g. ``t_inference`` counts from the completion of the
                 programming of a device.

         read_noise_scale: scale :math:`\rho` for scaling the read and
             accumulated noise :math:`1/f`.

         t_read: parameter of the :math:`1/f` noise (in seconds)

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        g_converter: Optional[BaseConductanceConverter] = None,
        g_max: Optional[float] = None,
        prog_coeff: Optional[List[float]] = None,
        prog_noise_scale: float = 1.0,
        drift_nu_mean: float = 0.1,
        drift_nu_std: float = 0.05,
        drift_scale: float = 1.0,
        t_0: float = 20.0,
        read_noise_scale: float = 1.0,
        t_read: float = 250.0e-9,
    ):
        g_converter = deepcopy(g_converter) or SinglePairConductanceConverter(g_max=g_max)
        super().__init__(g_converter)

        self.g_max = getattr(self.g_converter, "g_max", g_max)

        if self.g_max is None:
            raise ValueError("g_max cannot be established from g_converter")

        self.prog_coeff = [0.2, 0.0, 0.0] if prog_coeff is None else prog_coeff
        self.prog_noise_scale = prog_noise_scale
        self.drift_nu_mean = drift_nu_mean
        self.drift_nu_std = drift_nu_std
        self.drift_scale = drift_scale
        self.t_0 = t_0
        self.read_noise_scale = read_noise_scale
        self.t_read = t_read

    @no_grad()
    def apply_programming_noise_to_conductance(self, g_target: Tensor) -> Tensor:
        """Apply programming noise to a target conductance Tensor.

        Programming noise with additive Gaussian noise with
        conductance dependency of the variance given by a 2-degree
        polynomial.
        """
        mat = 1
        sig_prog = self.prog_coeff[0]
        for coeff in self.prog_coeff[1:]:
            mat *= g_target / self.g_max
            sig_prog += mat * coeff

        g_prog = g_target + self.prog_noise_scale * sig_prog * randn_like(g_target)
        g_prog.clamp_(min=0.0)  # no negative conductances allowed
        return g_prog

    @no_grad()
    def generate_drift_coefficients(self, g_target: Tensor) -> Tensor:
        """Return drift coefficients ``nu``."""

        mu_drift = self.drift_nu_mean
        sig_drift = self.drift_nu_std
        nu_drift = torch_abs(mu_drift + sig_drift * randn_like(g_target)).clamp(min=0.0)
        return nu_drift * self.drift_scale

    @no_grad()
    def apply_drift_noise_to_conductance(
        self, g_prog: Tensor, drift_noise_param: Tensor, t_inference: float
    ) -> Tensor:
        """Apply the noise and drift up to the assumed inference time
        point."""
        t = t_inference + self.t_0

        # drift
        if t > self.t_0:
            g_drift = g_prog * ((t / self.t_0) ** (-drift_noise_param))
        else:
            g_drift = g_prog

        # expected accumulated 1/f noise since start of programming at t=0
        if t > 0:
            sig_noise = sqrt(numpy_log((t + self.t_read) / (2 * self.t_read)))
            g_final = g_drift + torch_abs(
                g_drift / self.g_max
            ) * self.read_noise_scale * sig_noise * randn_like(g_drift)
        else:
            g_final = g_prog

        return g_final.clamp(min=0.0)

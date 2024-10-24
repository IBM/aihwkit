# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-branches

"""Phenomenological noise models for PCM devices for inference."""

from copy import deepcopy
from typing import List, Optional

from numpy import log as numpy_log
from numpy import sqrt
from torch import Tensor
from torch import abs as torch_abs
from torch import clamp, log, randn_like, zeros_like
from torch.autograd import no_grad

from aihwkit.inference.converter.base import BaseConductanceConverter
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter
from aihwkit.inference.noise.base import BaseNoiseModel

_ZERO_CLIP = 1e-7


class HermesNoiseModel(BaseNoiseModel):
    r"""A Noise model that was fitted and characterized on the PCM devices of the
    platform IBM HERMES Project Chip, see `Le Gallo et al. Nat. Electronics (2023)`_
    and `Khaddam-Aljameh et al. JSSC (2022)`_

    Expected weight noise at assumed time of inference with expected
    programming noise at 0.

    See also `Nandakumar et al. ICECS (2019)`_ for details about the
    statistical modelling methodology that was used.

    NOTE: The argument `num_devices` changes the programming method and the drift behavior of
    the model. When `num_devices` is 1, a conventional single device programming method is
    used. When `num_devices` is 2, the method from the work `Vasilopoulos et al. TED (2023)`_
    is employed (MSF), which is optimal and yields higher programming accuracy.
    For the drift characterization, though, when `num_devices` is 2 the model is applied as
    if the two devices host the same conductance and not as described in the aforementioned
    reference, due to it using a dynamic conductance mapping step which requires feedback from
    the chip in question. This simplification yields worse drift behavior in the current model
    than the one measured on-chip in the aforementioned work.

    Args:
        prog_coeff: Programming polynomial coeffs in
            :math:`\sum_i c_i \left(\frac{g_t}{g_\max}\right)^i`
        g_converter: instantiated class of the conductance converter
            (defaults to single pair)
        num_devices: The number of devices that are used to map a weight.
            Hermes supports either 1 or 2 devices per weight. When `num_devices`
            is 2, higher programming accuracy and less drift variability is measured.
            Defaults to 2.
        g_max: In :math:`\mu S`, the maximal conductance, ie the value
            the absolute max of the weights will be mapped to.
            When `num_devices = 2`, the maximum characterized conductance
            is 20.7 :math:`\mu S`. When `num_devices = 1` the maximum
            characterized value is 10.35 :math:`\mu S`. When `None` is passed (by default)
            the maximum conductance for the corresponding `num_devices` is selected.
        t_read: Parameter of the 1/f fit (in seconds).
        t_0: Parameter of the drift fit (first reading time). When `num_devices = 2`
            that time corresponds to 300s, while in the case that `num_devices = 1`
            it is 200s. If `None` is passed (by default) the time is selected according
            to the `num_devices` selection.

            Note:
                The ``t_inference`` is relative to this time `t0`
                e.g. t_inference counts from the completion of the programming
                of a device.
        prog_noise_scale: Scale for the programming noise.
        read_noise_scale: Scale for the read and accumulated noise.
        drift_scale: Scale for the  drift coefficient.
        prog_coeff_g_max_reference: reference :math:`g_\max` value
            when fitting the coefficients, since the result of the
            polynomial fit is given in uS. If
            ``prog_coeff_g_max_reference`` is not given and
            `prog_coeffs` are given explicitly, it will be set to
            ``g_max`` of the conductance converter.

    .. _`Nandakumar et al. ICECS (2019)`: https://ieeexplore.ieee.org/document/8964852
    .. _`Le Gallo et al. Nat. Electron. (2023)`: https://www.nature.com/articles/s41928-023-01010-1
    .. _`Khaddam-Aljameh et al. JSSC (2022)`: https://ieeexplore.ieee.org/document/9696185
    .. _`Vasilopoulos et al. TED (2023)`: https://ieeexplore.ieee.org/document/10281389

    """

    def __init__(
        self,
        prog_coeff: Optional[List[float]] = None,
        g_converter: Optional[BaseConductanceConverter] = None,
        num_devices: int = 2,
        g_max: Optional[float] = None,
        t_read: float = 512.0e-9,
        t_0: Optional[float] = None,
        prog_noise_scale: float = 1.0,
        read_noise_scale: float = 1.0,
        drift_scale: float = 1.0,
        prog_coeff_g_max_reference: Optional[float] = None,
    ):
        # The only valid options are 1 or 2 devices
        assert num_devices in [1, 2], "Hermes supports either 1 or 2 devices per weight"
        self.num_devices = num_devices

        # Figure out t0 depending on the num devices
        if t_0 is None:
            t_0 = 300.0 if self.num_devices == 2 else 200.0

        # Fix Gmax now
        if self.num_devices == 1:
            if g_max is not None:
                assert (
                    g_max <= 10.35
                ), "The maximum conductance characterized for single device unit cell is 10.35 uS"
            else:
                g_max = 10.35
        elif self.num_devices == 2:
            if g_max is not None:
                assert (
                    g_max <= 20.7
                ), "The maximum conductance characterized for single device unit cell is 10.35 uS"
            else:
                g_max = 20.7

        g_converter = deepcopy(g_converter) or SinglePairConductanceConverter(g_max=g_max)
        super().__init__(g_converter)

        self.g_max = getattr(self.g_converter, "g_max", g_max)
        if self.g_max is None:
            raise ValueError("g_max cannot be established from g_converter")

        if prog_coeff_g_max_reference is None:
            self.prog_coeff_g_max_reference = self.g_max

        if prog_coeff is None:
            # standard g_max are defined in respect to 20.7 or 10.35 uS. Need to
            # adjust for that in case g_max is not equal to the characterized maximum value
            if self.num_devices == 2:
                self.prog_coeff = [0.16603222, 4.71806468, -8.48101252, 4.68961419]
                self.prog_coeff_g_max_reference = 20.7
            elif self.num_devices == 1:
                self.prog_coeff = [0.15781817, 2.32443916, -2.16310839, 0.68841818]
                self.prog_coeff_g_max_reference = 10.35
        else:
            self.prog_coeff = prog_coeff

        self.t_0 = t_0
        self.t_read = t_read
        self.prog_noise_scale = prog_noise_scale
        self.read_noise_scale = read_noise_scale
        self.drift_scale = drift_scale

    @no_grad()
    def apply_programming_noise_to_conductance(self, g_target: Tensor) -> Tensor:
        """Apply programming noise to a target conductance Tensor.

        Programming noise with additive Gaussian noise with
        conductance dependency of the variance given by a 3-degree
        polynomial.
        """
        mat = 1
        sig_prog = self.prog_coeff[0]
        for coeff in self.prog_coeff[1:]:
            mat *= g_target / self.g_max
            sig_prog += mat * coeff

        sig_prog *= self.g_max / self.prog_coeff_g_max_reference  # type: ignore
        g_prog = g_target + self.prog_noise_scale * sig_prog * randn_like(g_target)
        g_prog.clamp_(min=0.0)  # no negative conductances allowed

        return g_prog

    @no_grad()
    def generate_drift_coefficients(self, g_target: Tensor) -> Tensor:
        """Return drift coefficients ``nu`` based on PCM measurements."""
        g_relative = clamp(torch_abs(g_target / self.g_max), min=_ZERO_CLIP)

        # gt should be normalized wrt g_max
        mu_drift, sig_drift = zeros_like(g_relative), zeros_like(g_relative)
        # Depending on the number of devices, different behavior is expected
        # for the standard deviation of the drift coefficient. The mean
        # behavior remains the same as it is considered that both devices
        # in the unit cell are programmed in the same state for simplicity.
        # The function for the mean and the standard deviation of
        # the ``nu`` factor are fitted with a branch function to match the
        # experimental data
        g_rel_low_mean, g_rel_high_mean = (
            g_relative[g_relative < 0.0945],
            g_relative[[g_relative >= 0.0945]],
        )
        mu_drift[g_relative < 0.0945] = (-0.0387 * log(g_rel_low_mean) - 0.0182).clamp(
            min=0.0720, max=0.13
        )
        mu_drift[g_relative >= 0.0945] = (
            -0.0436 * g_rel_high_mean**2 - 0.0126 * g_rel_high_mean + 0.0736
        )
        if self.num_devices == 1:
            g_rel_low_std, g_rel_high_std = (
                g_relative[g_relative < 0.3039],
                g_relative[[g_relative >= 0.3039]],
            )
            sig_drift[g_relative < 0.3039] = (-0.0120 * log(g_rel_low_std) - 0.0023).clamp(
                min=0.0124, max=0.04
            )
            sig_drift[g_relative >= 0.3039] = (
                -0.0165 * g_rel_high_std**2 + 0.0116 * g_rel_high_std + 0.0104
            )
        elif self.num_devices == 2:
            g_rel_low_std, g_rel_high_std = (
                g_relative[g_relative < 0.3055],
                g_relative[[g_relative >= 0.3055]],
            )
            sig_drift[g_relative < 0.3055] = (-0.0117 * log(g_rel_low_std) - 0.0057).clamp(
                min=0.0091, max=0.04
            )
            sig_drift[g_relative >= 0.3055] = (
                -0.0118 * g_rel_high_std**2 + 0.0093 * g_rel_high_std + 0.0073
            )

        nu_drift = torch_abs(mu_drift + sig_drift * randn_like(g_relative)).clamp(min=0.0)

        return nu_drift * self.drift_scale

    @no_grad()
    def apply_drift_noise_to_conductance(
        self,
        g_prog: Tensor,
        drift_noise_param: Tensor,
        t_inference: float,
    ) -> Tensor:
        """Apply the noise and drift up to the assumed inference time
        point based on PCM measurements."""
        t = t_inference + self.t_0

        # drift
        if t > self.t_0:
            g_drift = g_prog * ((t / self.t_0) ** (-drift_noise_param))
        else:
            g_drift = g_prog

        # expected accumulated 1/f noise since start of programming at t=0
        if t > 0:
            g_relative = torch_abs(g_prog) / self.g_max
            q_s = zeros_like(g_prog)
            if self.num_devices == 1:
                g_rel_low, g_rel_high = (
                    g_relative[g_relative < 0.1591],
                    g_relative[[g_relative >= 0.1591]],
                )
                q_s[g_relative < 0.1591] = (-0.0078 * log(g_rel_low) + 0.0038).clamp(
                    min=0.0179, max=0.04
                )
                q_s[g_relative >= 0.1591] = (
                    0.0664 * g_rel_high**3 - 0.1352 * g_rel_high**2 + 0.0768 * g_rel_high + 0.0088
                )
            elif self.num_devices == 2:
                g_rel_low, g_rel_high = (
                    g_relative[g_relative < 0.16],
                    g_relative[[g_relative >= 0.16]],
                )
                q_s[g_relative < 0.16] = (-0.0117 * log(g_rel_low) - 0.0069).clamp(
                    min=0.015, max=0.04
                )
                q_s[g_relative >= 0.16] = (
                    0.0069 * g_rel_high**3 - 0.0280 * g_rel_high**2 + 0.0211 * g_rel_high + 0.0123
                )
            sig_noise = q_s * sqrt(numpy_log((t + self.t_read) / (2 * self.t_read)))
            g_final = g_drift + torch_abs(g_drift) * self.read_noise_scale * sig_noise * randn_like(
                g_prog
            )
        else:
            g_final = g_prog

        return g_final.clamp(min=0.0)

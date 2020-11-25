# -*- coding: utf-8 -*-

# (C) Copyright 2020 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Phenomenological noise models for inference."""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from numpy import log as numpy_log
from numpy import sqrt
from torch import abs as torch_abs
from torch import clamp, log, ones, randn_like, Tensor
from torch.autograd import no_grad

_ZERO_CLIP = 1e-7


class BaseConductanceConverter:
    """Base class for converting DNN weights into conductances."""

    @no_grad()
    def convert_to_conductances(self, weights: Tensor) -> Tuple[List[Tensor], Dict]:
        """Converting a weight matrix into conductances.

        Caution:
            The conversion is assumed deterministic and repeatable.

        Args:
            weights: weight matrix tensor.

        Returns:
            Tuple of the list of conductance tensors and a params
            dictionary that is used for the reverse conversion.
        """
        raise NotImplementedError

    @no_grad()
    def convert_back_to_weights(self, conductances: List[Tensor], params: Dict) -> Tensor:
        """Converting a matrix of conductances into weights.

        Caution:
            The conversion is assumed deterministic and repeatable.

        Args:
            conductances: list of conductance tensors representing a weight matrix
            params: param dictionary that was returned from the conversion

        Returns:
            weight matrix
        """
        raise NotImplementedError


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

    def __init__(
            self,
            g_max: Optional[float] = None,
            g_min: Optional[float] = None
    ):
        self.g_max = 25.0 if g_max is None else g_max
        self.g_min = 0.0 if g_min is None else g_min
        self.scale_ratio = None

        if self.g_max < 0.:
            raise ValueError('g_max should be a positive value')
        if self.g_min < 0.:
            raise ValueError('g_min should be a positive value')
        if self.g_min > self.g_max:
            raise ValueError('g_min should be smaller than g_max')

    def __str__(self) -> str:
        return '{}(g_max={:1.2f}, g_min={:1.2f})'.format(
            self.__class__.__name__, self.g_max, self.g_min
        )

    @no_grad()
    def convert_to_conductances(self, weights: Tensor) -> Tuple[List[Tensor], Dict]:
        abs_max = torch_abs(weights).max()
        scale_ratio = (self.g_max - self.g_min) / abs_max.clamp(min=_ZERO_CLIP)
        scaled_weights = weights * scale_ratio

        conductances = [scaled_weights.clamp(min=0.0, max=self.g_max) + self.g_min,
                        (- scaled_weights).clamp(min=0.0, max=self.g_max) + self.g_min]
        params = {'scale_ratio': scale_ratio}

        return conductances, params

    @no_grad()
    def convert_back_to_weights(self, conductances: List[Tensor], params: Dict) -> Tensor:
        if len(conductances) != 2:
            raise ValueError('conductances must contain exactly two elements')
        if 'scale_ratio' not in params:
            raise ValueError('params do not contain scale_ratio')

        weights = ((conductances[0] - self.g_min) -
                   (conductances[1] - self.g_min)) / params['scale_ratio']

        return weights


class BaseNoiseModel:
    """Base class for phenomenological noise models for inference."""

    def __init__(
            self,
            g_converter: BaseConductanceConverter = None
    ):
        self.g_converter = g_converter or SinglePairConductanceConverter()

    @no_grad()
    def apply_noise(self, weights: Tensor, t_inference: float) -> Tensor:
        """Applies the expected noise.

        Applies the noise to a non-perturbed conductance matrix ``weights``
        at time of inference ``t_inference`` (in seconds) where 0 sec
        refers to the time when weight programming has finished.

        Note:
            The drift coefficients and intermediate noises etc. are
            sampled for each application of this function anew from the
            distributions, thus it samples the expected noise and drift
            behavior at time ``t_inference`` but not a continual
            trajectory of a given device instance over time (having
            e.g. constant drift coefficients).
        """
        target_conductances, params = self.g_converter.convert_to_conductances(weights)

        noisy_conductances = []
        for g_target in target_conductances:
            g_prog = self.apply_programming_noise_to_conductance(g_target)
            if t_inference > 0:
                nu_drift = self.generate_drift_coefficients(g_target)
                noisy_conductances.append(self.apply_drift_noise_to_conductance(
                    g_prog, nu_drift, t_inference))

        noisy_weights = self.g_converter.convert_back_to_weights(noisy_conductances, params)

        return noisy_weights

    @no_grad()
    def apply_programming_noise(self, weights: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Applies the expected programming noise to weights.

        Uses the :meth:`~apply_programming_noise_to_conductances` on
        each of the conductance slices.

        Parameters:
            weights: weights tensor

        Returns:
            weight tensor with programming noise applied, and tuple of
            all drift coefficients (per conductances slice) that are
            determined during programming.
        """
        target_conductances, params = self.g_converter.convert_to_conductances(weights)

        noisy_conductances = []
        nu_drift_list = []
        for g_target in target_conductances:

            noisy_conductances.append(self.apply_programming_noise_to_conductance(g_target))
            nu_drift_list.append(self.generate_drift_coefficients(g_target))
        noisy_weights = self.g_converter.convert_back_to_weights(noisy_conductances, params)

        return noisy_weights, nu_drift_list

    @no_grad()
    def apply_drift_noise(
            self,
            weights: Tensor,
            nu_drift_list: List[Tensor],
            t_inference: float
    ) -> Tensor:
        """Applies the expected drift noise to weights.

        Uses the :meth:`~apply_drift_noise_to_conductances` on
        each of the conductance slices.

        Parameters:
            weights: weights tensor (usually with programming noise already applied)
            nu_drift_list: list of drift nu for each conductance slice
            t_inference: assumed time of inference (in sec)

        Returns:
            weight tensor with drift noise applied
        """
        target_conductances, params = self.g_converter.convert_to_conductances(weights)

        noisy_conductances = []
        for g_target, nu_drift in zip(target_conductances, nu_drift_list):
            noisy_conductances.append(
                self.apply_drift_noise_to_conductance(g_target, nu_drift, t_inference))

        noisy_weights = self.g_converter.convert_back_to_weights(noisy_conductances, params)

        return noisy_weights

    @no_grad()
    def generate_drift_coefficients(self, g_target: Tensor) -> Tensor:
        """Generates drift coefficients ``nu`` based on the target conductances."""
        raise NotImplementedError

    @no_grad()
    def apply_programming_noise_to_conductance(self, g_target: Tensor) -> Tensor:
        r"""Apply programming noise to a target conductance ``Tensor``.

        Args:
            g_target: Target conductances

        Returns:
            Tensor of sampled drift coefficients :math:`\nu`, one for each
            target conductance value.
        """
        raise NotImplementedError

    @no_grad()
    def apply_drift_noise_to_conductance(
            self,
            g_prog: Tensor,
            nu_drift: Tensor,
            t_inference: float
    ) -> Tensor:
        r"""Apply the noise and drift up to the assumed inference time point.

        Args:
            g_prog: Tensor of conductance values after programming (in :math:`\muS`)
            nu_drift: drift nu
            t_inference: assumed time of inference (in sec)

        Returns:
            conductance Tensor with applied noise and drift
        """
        raise NotImplementedError


class PCMLikeNoiseModel(BaseNoiseModel):
    r"""Noise model that was fitted and characterized on real PCM devices.

    Expected weight noise at assumed time of inference with expected
    programming noise at 0.

    The statistical noise model is based on measured PCM devices.

    Args:
        prog_coeff: programming polynomial coeffs in :math:`\mu S`, c(0) + c(1)*gt + c(2)*gt^2)
        g_converter: instantiated class of the conductance converter (defaults to single pair)
        g_max: In :math:`\mu S`, the maximal conductance, ie the value
            the absolute max of the weights will be mapped to.
        t_read: parameter of the 1/f fit (in seconds)
        t_0: parameter of the drift fit (first reading time)

            Note:
                The ``t_inference`` is relative to this time `t0`
                e.g. t_inference counts from the completion of the programming
                of a device.
    """

    def __init__(
            self,
            prog_coeff: Optional[List[float]] = None,
            g_converter: Optional[BaseConductanceConverter] = None,
            g_max: Optional[float] = None,
            t_read: float = 250.0e-9,
            t_0: float = 20.0,
    ):
        g_converter = deepcopy(g_converter) or SinglePairConductanceConverter(g_max=g_max)
        super().__init__(g_converter)

        self.g_max = getattr(self.g_converter, 'g_max', g_max)

        if self.g_max is None:
            raise ValueError('g_max cannot be established from g_converter')

        self.prog_coeff = [0.26348, 1.9650, -1.1731] if prog_coeff is None else prog_coeff
        self.t_0 = t_0
        self.t_read = t_read

    def __str__(self) -> str:
        return ('{}(prog_coeff={}, g_converter={}, g_max={:1.2f}, t_read={}, '
                't_0={:1.2f})').format(
                    self.__class__.__name__, self.prog_coeff, self.g_converter,
                    self.g_max, self.t_read, self.t_0)

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

        g_prog = g_target + sig_prog * randn_like(g_target)
        g_prog.clamp_(min=0.0)  # no negative conductances allowed

        return g_prog

    @no_grad()
    def generate_drift_coefficients(self, g_target: Tensor) -> Tensor:
        """Return drift coefficients ``nu`` based on PCM measurements."""
        g_relative = clamp(torch_abs(g_target / self.g_max), min=_ZERO_CLIP)

        # gt should be normalized wrt g_max
        mu_drift = (-0.0155 * log(g_relative) + 0.0244).clamp(min=0.049, max=0.1)
        sig_drift = (-0.0125 * log(g_relative) - 0.0059).clamp(min=0.008, max=0.045)
        nu_drift = torch_abs(mu_drift + sig_drift * randn_like(g_relative)).clamp(min=0.0)

        return nu_drift

    @no_grad()
    def apply_drift_noise_to_conductance(
            self,
            g_prog: Tensor,
            nu_drift: Tensor,
            t_inference: float
    ) -> Tensor:
        """Apply the noise and drift up to the assumed inference time
        point based on PCM measurements."""
        t = t_inference + self.t_0

        # drift
        if t > self.t_0:
            g_drift = g_prog * ((t / self.t_0) ** (- nu_drift))
        else:
            g_drift = g_prog

        # expected accumulated 1/f noise since start of programming at t=0
        if t > 0:
            q_s = (0.0088 / ((torch_abs(g_prog) /
                              self.g_max) ** 0.65).clamp(min=1e-3)).clamp(max=0.2)
            sig_noise = q_s * sqrt(numpy_log((t + self.t_read) / (2 * self.t_read)))
            g_final = g_drift + torch_abs(g_drift) * sig_noise * randn_like(g_prog)
        else:
            g_final = g_prog

        return g_final.clamp(min=0.0)


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
        """Implements the read out math."""
        raise NotImplementedError

    @no_grad()
    def apply(self, forward_output: Tensor, ref_value: Tensor) -> Tensor:
        """Read out the current value from the output of the forward
        pass and returns the drift compensation alpha scale."""
        current_value = self.readout(forward_output)
        ratio = ref_value/current_value

        return ratio


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
        """Returns the read-out tensor.

        Uses a single all one vector.
        """
        return ones((1, in_size))

    def __str__(self) -> str:
        return '{}()'.format(self.__class__.__name__)

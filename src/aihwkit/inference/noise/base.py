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

"""Base class for the phenomenological noise models for inference."""

from typing import List, Tuple, Optional
from torch import Tensor
from torch.autograd import no_grad

from aihwkit.inference.converter.base import BaseConductanceConverter
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter


class BaseNoiseModel:
    """Base class for phenomenological noise models for inference."""

    def __init__(self, g_converter: Optional[BaseConductanceConverter] = None):
        self.g_converter = g_converter or SinglePairConductanceConverter()

    def __eq__(self, other: object) -> bool:
        return self.__class__ == other.__class__ and self.__dict__ == other.__dict__

    def __str__(
        self, exclude_keys: Optional[List[str]] = None, keys: Optional[List[str]] = None
    ) -> str:
        """Print instance."""
        ret = self.__class__.__name__ + "("
        if keys is None:
            keys = list(self.__dict__.keys())

        if exclude_keys is not None:
            for key in exclude_keys:
                keys.remove(key)

        for key in keys:
            ret += key + "={}, ".format(self.__dict__[key])
        ret = ret[:-2] + ")"
        return ret

    @no_grad()
    def apply_noise(self, weights: Tensor, t_inference: float) -> Tensor:
        """Apply the expected noise.

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
                noisy_conductances.append(
                    self.apply_drift_noise_to_conductance(g_prog, nu_drift, t_inference)
                )

        noisy_weights = self.g_converter.convert_back_to_weights(noisy_conductances, params)

        return noisy_weights

    @no_grad()
    def apply_programming_noise(self, weights: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Apply the expected programming noise to weights.

        Uses the :meth:`~apply_programming_noise_to_conductances` on
        each of the conductance slices.

        Args:
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
        self, weights: Tensor, drift_noise_parameters: List[Optional[Tensor]], t_inference: float
    ) -> Tensor:
        """Apply the expected drift noise to weights.

        Uses the :meth:`~apply_drift_noise_to_conductances` on
        each of the conductance slices.

        Args:
            weights: weights tensor (usually with programming noise already applied)
            drift_noise_parameters: list of drift nu for each conductance slice
            t_inference: assumed time of inference (in sec)

        Returns:
            weight tensor with drift noise applied
        """
        target_conductances, params = self.g_converter.convert_to_conductances(weights)

        noisy_conductances = []
        for g_target, drift_noise_param in zip(target_conductances, drift_noise_parameters):
            noisy_conductances.append(
                self.apply_drift_noise_to_conductance(g_target, drift_noise_param, t_inference)
            )

        noisy_weights = self.g_converter.convert_back_to_weights(noisy_conductances, params)

        return noisy_weights

    @no_grad()
    def generate_drift_coefficients(self, g_target: Tensor) -> Optional[Tensor]:
        """Generate drift coefficients.

        Generate coefficients once and passed through when
        long-term noise and drift is applied. Typical `nu_drift`.

        Args:
            g_target: Target conductances

        Returns:
            When not overriden, it simply returns None.
        """
        # pylint: disable=unused-argument

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
        self, g_prog: Tensor, drift_noise_param: Optional[Tensor], t_inference: float
    ) -> Tensor:
        r"""Apply the noise and drift up to the assumed inference time point.

        Args:
            g_prog: Tensor of conductance values after programming (in :math:`\muS`)
            drift_noise_param: typically drift nu
            t_inference: assumed time of inference (in sec)

        Returns:
            conductance Tensor with applied noise and drift
        """
        raise NotImplementedError

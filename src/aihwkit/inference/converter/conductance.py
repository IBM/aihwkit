# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Conductance converters for the phenomenological noise models for inference."""

from typing import Dict, List, Optional, Tuple

from torch import abs as torch_abs, stack
from torch import Tensor, zeros_like, from_numpy, linspace, allclose
from torch.autograd import no_grad

from numpy import interp

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


class NPairConductanceConverter(BaseConductanceConverter):
    r"""N pairs of conductance devices per unit cell (generalized).

    Assuming a N pairs of devices per cross-point, each having a relative
    weighting (i.e. F factor) as defined by the values in the f_lst parameter.
    For positive and negative weights, one device within in the pair is always
    set to g_min. The higher significant pair will only be used once the
    range of the lower significant pairs is exhausted. In this way, we minimize
    amplification of programming errors and read noise by the scale factors F.
    Note that the scale factors can also be values less than 1.0, however. The
    F factor can be implemented using an amplifying current mirror or by applying
    a longer pulse durations to one pair of conductance pair relative to another,
    such that their is a greater current contribution even though all devices are
    sized equally.

    Args:
        f_lst: In: list of weighting (i.e. scale) factors from lowest
            to hightest, used to determinethe significance of each
            conductance pair.
        g_max: In :math:`\mu S`, the maximal conductance, ie the value
            the absolute max of the weights will be mapped to.
        g_min: In :math:`\mu S`, the minimal conductance, ie the value
            the logical zero of the weights will be mapped to.
    """

    def __init__(self,
                 f_lst: List[float],
                 g_max: Optional[float] = None,
                 g_min: Optional[float] = None,
                 ):

        if not isinstance(f_lst, list):
            raise ValueError("f_lst parameter must be a list of F factors")

        if max(f_lst) < 0.:
            raise ValueError("f_lst parameter contains negative value")

        self.g_max = 25.0 if g_max is None else g_max
        self.g_min = 0.0 if g_min is None else g_min
        self.f_lst = f_lst
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
        max_weight_us = sum(f * (self.g_max - self.g_min) for f in self.f_lst)
        max_weight_unitless = torch_abs(weights).max().clamp(min=_ZERO_CLIP)
        scale_ratio = max_weight_us / max_weight_unitless
        weights_us = scale_ratio * weights

        lower_bound_us = 0.  # lower bound in uS
        conductances = []
        for f_factor in self.f_lst:
            conductances.append(((weights_us.clamp(min=0.0) - lower_bound_us) / f_factor
                                 + self.g_min).clamp(min=self.g_min, max=self.g_max))      # g_plus
            conductances.append((((-weights_us).clamp(min=0.0) - lower_bound_us) / f_factor
                                 + self.g_min).clamp(min=self.g_min, max=self.g_max))      # g_minus
            lower_bound_us += f_factor * (self.g_max - self.g_min)

        params = {'scale_ratio': scale_ratio,
                  'f_lst': self.f_lst}

        return conductances, params

    @no_grad()
    def convert_back_to_weights(self, conductances: List[Tensor], params: Dict) -> Tensor:
        if 'f_lst' not in params:
            raise ValueError("params does not contain f_lst")
        if len(conductances) % 2 != 0:
            raise ValueError("unit cell must have an even number of conductances")
        if 'scale_ratio' not in params:
            raise ValueError("params does not contain scale_ratio")

        weights = zeros_like(conductances[0])
        for f_factor, (g_plus, g_minus) in zip(self.f_lst,
                                               zip(conductances[::2], conductances[1::2])):
            weights += f_factor * (g_plus - g_minus)

        return weights / params['scale_ratio']


class DualPairConductanceConverter(NPairConductanceConverter):
    r"""Two pairs of conductance devices per unit cell (4 devices total).

    Assuming a two pairs of devices per cross-point, each having a relative
    weighting (i.e. F factor) as defined by the values in the f_lst parameter.
    For positive and negative weights, one device within in the pair is always
    set to g_min. The higher significant pair will only be used once the
    range of the lower significant pairs is exhausted. In this way, we minimize
    amplification of programming errors and read noise by the scale factors F.
    Note that the scale factors can also be values less than 1.0, however. The
    F factor can be implemented using an amplifying current mirror or by applying
    a longer pulse durations to one pair of conductance pair relative to another,
    such that their is a greater current contribution even though all devices are
    sized equally.

    Args:
        f_lst: In: list of weighting (i.e. scale) factors from lowest
            to hightest, used to determinethe significance of each
            conductance pair.
        g_max: In :math:`\mu S`, the maximal conductance, ie the value
            the absolute max of the weights will be mapped to.
        g_min: In :math:`\mu S`, the minimal conductance, ie the value
            the logical zero of the weights will be mapped to.
    """

    def __init__(self,
                 f_lst: List[float],
                 g_max: Optional[float] = None,
                 g_min: Optional[float] = None,
                 ):

        if len(f_lst) != 2:
            raise ValueError("f_lst parameter does not contain two values")

        super().__init__(f_lst=f_lst,
                         g_max=g_max,
                         g_min=g_min)


class CustomPairConductanceConverter(BaseConductanceConverter):
    r"""Arbitrary even number of devices.

    Assuming an arbitrary pair of devices per cross-point, each pair having a
    relative weight defined by the values in the f_lst parameter. The parameter
    g_lst is a list of lists that map the unitless weights to a series of
    conductance values. These lists allow us to interpolate and implement a
    function g(w) so that we can map unitless weight to their corresponding
    conductance values. In this way, we can implement very complex conductance
    programming schemes. The various F factors can be implemented using amplifying
    current mirrors or by applying longer pulse durations to the one conductance
    pair relative to others such that their is a greater current contribution even
    though all devices are sized equally.

    Args:
        f_lst: In: list of weighting (i.e. scale) factors used for the
            more significant conductance pairs and the less significant
            conductance pairs.
        g_lst: In: list of lists that map unitless weights to
            conductance values.
        g_max: In :math:`\mu S`, the maximal conductance, ie the value
            the absolute max of the weights will be mapped to.
        g_min: In :math:`\mu S`, the minimal conductance, ie the value
            the logical zero of the weights will be mapped to.
    """

    def __init__(self,
                 f_lst: List[float],
                 g_lst: List[List[float]],
                 g_max: Optional[float] = None,
                 g_min: Optional[float] = None,
                 invertibility_test: Optional[bool] = True,
                 ):
        self.g_max = 25.0 if g_max is None else g_max
        self.g_min = 0.0 if g_min is None else g_min

        if not isinstance(g_lst, list):
            raise ValueError("g_lst parameter must be a list")

        if not all(isinstance(g, list) for g in g_lst):
            raise ValueError("g_lst must be list of lists")

        if len(g_lst) % 2 != 0:
            raise ValueError("g_lst must have and even number of elements")

        if not isinstance(f_lst, list):
            raise ValueError("f_lst parameter must be a list")

        if 2 * len(f_lst) != len(g_lst):
            raise ValueError("must have one value in f_lst for every pair of values in g_lst")

        self.f_lst = f_lst
        self.g_lst = g_lst
        self.scale_ratio = None

        if self.g_max < 0.0:
            raise ValueError("g_max should be a positive value")
        if self.g_min < 0.0:
            raise ValueError("g_min should be a positive value")
        if self.g_min >= self.g_max:
            raise ValueError("g_min should be smaller than g_max")

        if invertibility_test:
            self.invertibility_test()

    def invertibility_test(self) -> None:
        r"""Test to make sure custom conductance converter specification is invertible

        This method tests to make sure the custom converter specification represents
        and invertible function, meaning the g = f(w) <--> w = f^{-1}(x) is true.
        Otherwise, converting unitless weights to conductances and then subsequently
        converting those conductances back to unitless weights will introduce changes
        into the weights, which should not be there. The default is to run this test
        upon instantiation and return an error if it passes. This prevents ill-defined
        custom conductance converter models from corrupting simulation results.
        """
        test_weights = linspace(-1, 1, 5)
        conductances, params = self.convert_to_conductances(test_weights)
        return_weights = self.convert_back_to_weights(conductances, params)
        if not allclose(test_weights, return_weights, atol=0.0001):
            raise ArithmeticError("CustomPairConductanceConverter is not an invertible function")

    def __str__(self) -> str:
        return "{}(g_max={:1.2f}, g_min={:1.2f})".format(
            self.__class__.__name__, self.g_max, self.g_min
        )

    @no_grad()
    def convert_to_conductances(self, weights: Tensor) -> Tuple[List[Tensor], Dict]:

        weights_us = zeros_like(Tensor(self.g_lst[0])).type_as(weights)
        for f_factor, (gp_lst, gm_lst) in zip(self.f_lst,
                                              zip(self.g_lst[::2], self.g_lst[1::2])):
            weights_us += f_factor * (Tensor(gp_lst) - Tensor(gm_lst)).type_as(weights)

        max_weight = torch_abs(weights).max()
        max_weight_us = torch_abs(weights_us).max()
        scale_ratio = max_weight_us / max_weight.clamp(min=_ZERO_CLIP)

        w_lst = (linspace(-max_weight_us,
                          max_weight_us,
                          len(self.g_lst[0])) / scale_ratio).tolist()

        conductances = []
        for f_factor, (gp_lst, gm_lst) in zip(self.f_lst,
                                              zip(self.g_lst[::2], self.g_lst[1::2])):
            conductances.append(from_numpy(interp(weights.cpu().numpy(),
                                                  w_lst,
                                                  gp_lst)).type_as(weights))
            conductances.append(from_numpy(interp(weights.cpu().numpy(),
                                                  w_lst,
                                                  gm_lst)).type_as(weights))

        params = {'scale_ratio': scale_ratio,
                  'f_lst': self.f_lst}

        return conductances, params

    @no_grad()
    def convert_back_to_weights(self, conductances: List[Tensor], params: Dict) -> Tensor:

        if 'f_lst' not in params:
            raise ValueError("params does not contain f_lst")

        if not isinstance(params['f_lst'], list):
            raise TypeError("f_lst parameter must be a list of f factors")

        if 2 * len(params['f_lst']) != len(conductances):
            raise ValueError("must have one value in f_lst for every pair of conductances")

        weights_us = zeros_like(conductances[0])
        for f_factor, (g_plus, g_minus) in zip(params['f_lst'],
                                               zip(conductances[::2], conductances[1::2])):
            weights_us += f_factor * (g_plus - g_minus)

        return weights_us / params['scale_ratio']   # back to unitless


class SingleDeviceConductanceConverter(BaseConductanceConverter):
    r"""Single devices to represent weights

    Assuming a single bidirectional device per cross-point
    Args:
        g_max: In :math:`\mu S`, the maximal conductance, ie the value
            the absolute max of the weights will be mapped to.
        g_min: In :math:`\mu S`, the minimal conductance, ie the value
            the logical zero of the weights will be mapped to.
    """

    def __init__(self, g_max: Optional[float] = None, g_min: Optional[float] = None):
        self.g_max = 88.199997 if g_max is None else g_max
        self.g_min = 9.0 if g_min is None else g_min
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
        w_min = weights.min()
        w_max = weights.max()
        scale_ratio = (self.g_max - self.g_min) / (w_max - w_min)
        scaled_weights = (weights - w_min) * scale_ratio
        conductance = scaled_weights + self.g_min
        params = {"scale_ratio": scale_ratio, "min": w_min}
        return conductance, params

    @no_grad()
    def convert_back_to_weights(self, conductances: Tensor, params: Dict) -> Tensor:
        if "scale_ratio" not in params:
            raise ValueError("params do not contain scale_ratio")
        if "min" not in params:
            raise ValueError("params do not contain min")

        if isinstance(conductances, list):
            conductances = stack(conductances)

        weights = params["min"] + ((conductances - self.g_min) / params["scale_ratio"])

        return weights

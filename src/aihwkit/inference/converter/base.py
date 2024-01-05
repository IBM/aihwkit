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

"""Base conductance converter for the phenomenological noise models for inference."""

from typing import Dict, List, Tuple

from torch import Tensor
from torch.autograd import no_grad


class BaseConductanceConverter:
    """Base class for converting DNN weights into conductances."""

    @no_grad()
    def convert_to_conductances(self, weights: Tensor) -> Tuple[List[Tensor], Dict]:
        """Convert a weight matrix into conductances.

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
        """Convert a matrix of conductances into weights.

        Caution:
            The conversion is assumed deterministic and repeatable.

        Args:
            conductances: list of conductance tensors representing a weight matrix
            params: param dictionary that was returned from the conversion

        Returns:
            weight matrix
        """
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        return self.__class__ == other.__class__ and self.__dict__ == other.__dict__

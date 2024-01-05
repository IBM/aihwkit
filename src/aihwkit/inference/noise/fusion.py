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

"""Fusion chip noise model for inference."""

from copy import deepcopy
from typing import List, Optional, Tuple, Any

from torch import Tensor
from torch.autograd import no_grad

from aihwkit.exceptions import TileError
from aihwkit.inference.noise.base import BaseNoiseModel
from aihwkit.inference.converter.base import BaseConductanceConverter
from aihwkit.inference.converter.fusion import FusionConductanceConverter


class FusionImportNoiseModel(BaseNoiseModel):
    r"""Using the Fusion Chip to use realistic and measured PCM noise.

    This noise model does not generate noise but instead applies the
    programmed noise and read values from the Fusion Chip to the
    weights.

    It will be automatically configured with the programmed weights
    from the Fusion Chip when using::

        from aihwkit.utils.export import fusion_import
        model = fusion_import(csv_file, model)

    Args:
        programmed_conductances: Condactances from the Fusion Chip
        g_converter: g_converter (default FusionConductanceConverter)
        **converter_params: parameters used for the conductance conversion
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        programmed_conductances: List[Tensor],
        g_converter: Optional[BaseConductanceConverter] = None,
        **converter_params: Any,
    ):
        g_converter = deepcopy(g_converter) or FusionConductanceConverter()
        self.programmed_conductances = programmed_conductances
        self.converter_params = converter_params

        super().__init__(g_converter)

    def __str__(self) -> str:  # type: ignore
        # pylint: disable=signature-differs
        return super().__str__(["programmed_conductances", "converter_params"])

    @no_grad()
    def apply_noise(self, weights: Tensor, t_inference: float) -> Tensor:
        raise NotImplementedError

    @no_grad()
    def apply_programming_noise(self, weights: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Simply return the original weights.

        Since the programming to the Fusion Chip will also include
        drift, we set the weight during drift noise to enable drift
        compensation against the target weight values

        Args:
            weights: Weights to program. Note that this is unused as
                the ``programmed_conductances`` for this weights are simply taken.

        Returns:
            weight tensor with programming noise applied, and tuple of
            all drift coefficients (per conductances slice) that are
            determined during programming.

        """
        return weights, [None]

    @no_grad()
    def apply_drift_noise(
        self, weights: Tensor, drift_noise_parameters: List[Optional[Tensor]], t_inference: float
    ) -> Tensor:
        """Apply the stored programming noise from the Fusion Chip to weights.

        In principle, additional drift could be applied here, however,
        currently nothing is implemented and the Fusion Chip programmed weights
        are simply return.

        Args:
            weights: weights tensor that was used to program to the Fusion Chip
            drift_noise_parameters: list of parameters (ignored)
            t_inference: assumed time of inference (in sec) (ignored)

        Returns:
            weight tensor with programming noise applied

        Raises:
            TileError: In case the dimensions of the programmed conductances mismatch

        """

        noisy_weights = self.g_converter.convert_back_to_weights(
            self.programmed_conductances, self.converter_params
        )

        if noisy_weights.shape != weights.shape:
            raise TileError("Fusion programmed weights shape mismatch")

        return noisy_weights

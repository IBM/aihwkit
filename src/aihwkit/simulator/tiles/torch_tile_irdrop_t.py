# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=too-many-locals, too-many-arguments

"""Low level implementation of torch-based tile."""

from typing import TYPE_CHECKING

from torch import Tensor

from aihwkit.exceptions import TorchTileConfigError
from aihwkit.simulator.tiles.torch_tile import TorchSimulatorTile
from aihwkit.simulator.tiles.analog_mvm_irdrop_t import AnalogMVMIRDropT

if TYPE_CHECKING:
    from aihwkit.simulator.configs.configs import TorchInferenceRPUConfigIRDropT


class TorchSimulatorTileIRDropT(TorchSimulatorTile):
    """Torch tile class including time-dependent IR drop calculation.

    Args:
        x_size: input size
        d_size: output size
        rpu_config: resistive processing unit configuration.
    """

    # pylint: disable=abstract-method

    def __init__(
        self,
        x_size: int,
        d_size: int,
        rpu_config: "TorchInferenceRPUConfigIRDropT",
        bias: bool = False,
    ):
        self._phys_input_size = rpu_config.mapping.max_input_size or x_size
        self._g_converter = rpu_config.noise_model.g_converter

        super().__init__(x_size, d_size, rpu_config, bias, analog_mvm=AnalogMVMIRDropT)

    def forward(
        self,
        x_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        is_test: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        if in_trans or out_trans:
            raise TorchTileConfigError("Non-trans MVMs supported only.")

        if not is_test:
            noisy_weights = self.modify_weight(self.weight, self._modifier, x_input.shape[0])
        else:
            noisy_weights = self.weight

        return self._analog_mvm.matmul(
            noisy_weights,
            x_input,
            self._f_io,
            False,
            is_test,
            phys_input_size=self._phys_input_size,
            g_converter=self._g_converter,
            out_noise_values=self.out_noise_values,
        )

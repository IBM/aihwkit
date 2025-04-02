# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# mypy: disable-error-code=attr-defined

"""Tile with quantized periphery and outputs."""

from typing import TYPE_CHECKING, Optional, Tuple

from torch import Tensor

from aihwkit.simulator.digital_low_precision.base_quantized_classes import QuantizedActivation
from aihwkit.simulator.digital_low_precision.config_utils import convert_act_config_to_kwargs_dict
from aihwkit.simulator.tiles import TorchInferenceTile

if TYPE_CHECKING:
    from aihwkit.simulator.configs import QuantizedTorchInferenceRPUConfig


class QuantizedTorchInferenceTile(TorchInferenceTile):
    """InferenceTile using a torch-based simulator tile (and not a
    tile from RPUCuda). It extends the TorchInferenceTile, adding
    support for quantized periphery, which corresponds to the bias
    and the affine scales, as well as output activation quantization
    with configurable parameters. To configure the various quantization
    parameters, use `QuantizedTorchInferenceRPUConfig`.
    """

    def __init__(
        self,
        out_size: int,
        in_size: int,
        rpu_config: Optional["QuantizedTorchInferenceRPUConfig"] = None,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
    ):
        # Initialize the quantized torch inference RPU config if not given
        if not rpu_config:
            # Import dynamically to avoid import cycles.
            # pylint: disable=import-outside-toplevel
            from aihwkit.simulator.configs import QuantizedTorchInferenceRPUConfig

            rpu_config = QuantizedTorchInferenceRPUConfig()

        # Call the constructor of the TorchInferenceTile but with the new rpu config
        super().__init__(out_size, in_size, rpu_config, bias, in_trans, out_trans)

        # Define and enable the out_quantizer if defined appropriately
        if rpu_config.act_quant_config is not None and rpu_config.act_quant_config.n_bits > 0:
            self.out_quantizer = QuantizedActivation(
                **convert_act_config_to_kwargs_dict(rpu_config.act_quant_config)
            )
            self.out_quantizer.quantized_acts()
        else:
            self.out_quantizer = None

    def forward(self, x_input: Tensor, tensor_view: Optional[Tuple] = None) -> Tensor:
        """Torch forward function that calls the analog forward. It is different
        than the TorchInferenceTile in the way it handles the bias addition, affine
        scaling application and the output quantization. See the methods
        `post_forward`, `apply_quant_periphery_scales` and `add_quant_periphery_bias`
        in `TileWithPeriphery` for details"""
        # pylint: disable=arguments-differ

        # Note: this is now called with autograd enabled and thus will
        # not use the BaseTile.backward functionality. It will call
        # the tile.forward pass internally
        self.tile.set_config(self.rpu_config)  # to allow on-the-fly changes
        out = self.joint_forward(x_input, is_test=not self.training)

        if self.digital_bias:
            if tensor_view is None:
                tensor_view = self.get_tensor_view(out.dim())
            # Add normal or quantized bias, based on configuration
            out = self.add_quant_periphery_bias(out, tensor_view, is_test=not self.training)

        # Quantize the output, based on configurationF
        if self.out_quantizer is not None:
            out = self.out_quantizer(out)

        return out

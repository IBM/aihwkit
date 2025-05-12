# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# mypy: disable-error-code=attr-defined

"""Implements analog tile module array ."""

from typing import Any, Optional, Tuple, List, TYPE_CHECKING

from torch import Tensor, cat, split, zeros, full
from torch.nn import ModuleList, Parameter, Module
from torch.autograd import no_grad

from aihwkit.simulator.tiles.base import TileModuleBase
from aihwkit.exceptions import TileModuleError
from aihwkit.simulator.digital_low_precision.base_quantized_classes import QuantizedActivation
from aihwkit.simulator.digital_low_precision.config_utils import convert_act_config_to_kwargs_dict
from aihwkit.simulator.digital_low_precision.base_quantized_classes import QuantizationManager
from aihwkit.simulator.digital_low_precision.quantizers import QMethods
from aihwkit.simulator.digital_low_precision.range_estimators import RangeEstimators

if TYPE_CHECKING:
    from aihwkit.simulator.configs.configs import MappableRPU
    from aihwkit.simulator.configs import QuantizedTorchInferenceRPUConfig


class TileModuleArray(Module, TileModuleBase):
    """Logical array of tile modules.

    Note:

        The bias in the RPUConfig does not have any effect since the
        bias is always concatenated for the logical array and added at
        the end in digital

    """

    supports_indexed = False

    def __init__(
        self,
        out_size: int,
        in_size: int,
        rpu_config: "MappableRPU",
        bias: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # More than one tile may need to be created. If so, divide
        # weight matrix into equal pieces along input dimension with
        # as many tiles as needed

        if bias:
            self.bias = Parameter(zeros(out_size), requires_grad=True)
        else:
            self.bias = None  # type: ignore

        max_input_size = rpu_config.mapping.max_input_size
        max_output_size = rpu_config.mapping.max_output_size

        self.in_size = in_size
        self.out_size = out_size
        self.in_sizes = self.get_split_sizes(in_size, max_input_size)
        self.out_sizes = self.get_split_sizes(out_size, max_output_size)
        self.analog_tile_count = len(self.in_sizes) * len(self.out_sizes)

        self.array = ModuleList()
        for in_tile_size in self.in_sizes:
            in_tiles = ModuleList()
            for out_tile_size in self.out_sizes:
                tile = rpu_config.tile_class(
                    out_tile_size, in_tile_size, rpu_config, bias=False, **kwargs
                )
                in_tiles.append(tile)
            self.array.append(in_tiles)

    @no_grad()
    def get_split_sizes(self, size: int, split_max_size: int) -> List[int]:
        """Computed the split sizes.

        Args:
            size: number of elements of the layer in one dimension
            split_max_size: max size of the split

        Returns:
            List of split sizes
        """
        if split_max_size <= 0:
            return [size]

        n_splits = (size + split_max_size - 1) // split_max_size
        base, extra = divmod(size, n_splits)
        return [base + (i < extra) for i in range(n_splits)]

    @no_grad()
    def set_weights(self, weight: Tensor, bias: Optional[Tensor] = None, **kwargs: Any) -> None:
        """Set the weight (and bias) values with given tensors to the analog crossbar(s).

        Args:
            weight: weight matrix
            bias: bias vector
            **kwargs: see tile level,
                e.g. :meth:`~aihwkit.simulator.tiles.analog.AnalogTile.set_weights`
        """
        shape = [self.out_size, self.in_size]
        weight = weight.detach().reshape(shape)

        in_start = in_end = 0
        for in_size, in_tiles in zip(self.in_sizes, self.array):
            in_end += in_size
            out_start = out_end = 0
            for out_size, analog_tile in zip(self.out_sizes, in_tiles):
                out_end += out_size

                tile_weight = weight[out_start:out_end, in_start:in_end]

                analog_tile.set_weights(tile_weight, None, **kwargs)

                out_start = out_end
            in_start = in_end

        if self.bias is not None and bias is not None:
            self.bias.data = bias.detach().to(self.bias.device)

    @no_grad()
    def get_weights(self, **kwargs: Any) -> Tuple[Tensor, Optional[Tensor]]:
        """Get the (combined) weight (and bias) tensors from the analog crossbar(s).

        Args:
            kwargs: see tile level,
                e.g. :meth:`~aihwkit.simulator.tiles.analog.AnalogTile.get_weights`

        Returns:
            tuple: weight matrix, bias vector
        """

        weight_lst = []
        for in_tiles in self.array:
            in_tile_weight = []
            for analog_tile in in_tiles:
                tile_weight, _ = analog_tile.get_weights(**kwargs)
                in_tile_weight.append(tile_weight)
            weight_lst.append(cat(in_tile_weight, 0))

        weight = cat(weight_lst, 1)

        if self.bias is not None:
            return weight, self.bias.clone().cpu()
        return weight, None

    def forward(self, x_input: Tensor, tensor_view: Optional[Tuple] = None) -> Tensor:
        """Compute the forward pass."""
        # pylint: disable=arguments-differ,arguments-renamed

        if self.analog_tile_count == 1:
            analog_tile = self.array[0][0]  # pylint: disable=unsubscriptable-object
            result = analog_tile(x_input)
        else:
            # mapped version
            last_dim = x_input.ndim - 1
            splits = split(x_input, self.in_sizes, dim=last_dim)
            result = None
            for idx, (x, in_tiles) in enumerate(zip(splits, self.array)):
                out_result = []

                for analog_tile in in_tiles:
                    out_result.append(analog_tile(x, tensor_view=tensor_view))

                if idx == 0:
                    result = cat(out_result, last_dim)
                else:
                    result.add_(cat(out_result, last_dim))

        if self.bias is not None:
            if tensor_view is None:
                tensor_view = analog_tile.get_tensor_view(result.dim())
            return result + self.bias.view(*tensor_view)
        return result

    def _apply_to_tiles(self, method_name: str, *args: Any, **kwargs: Any) -> List[List[Any]]:
        """Applies function to each tile in the array.

        Raises:
            TileModuleError: if method does not exist
        """
        out_values = []
        for in_tiles in self.array:
            out_values_row = []
            for analog_tile in in_tiles:
                if not hasattr(analog_tile, method_name):
                    raise TileModuleError(f"Tile does not have method '{method_name}'")
                out_values_row.append(getattr(analog_tile, method_name)(*args, **kwargs))
            out_values.append(out_values_row)
        return out_values


class QuantizedTileModuleArray(TileModuleArray):
    """Logical array of quantized torch inference tile modules. It extends
    the functionality of `TileModuleArray`, by adding quantization capability
    for the bias (which is applied here instead of the individual tiles) and
    for the final result of the array, after all the partial results from the
    tiles have been accumulated.

    It only overwrites the forward function of the `TileModuleArray`, to add the
    output and bias quantization.
    """

    def __init__(
        self,
        out_size: int,
        in_size: int,
        rpu_config: "QuantizedTorchInferenceRPUConfig",
        bias: bool = False,
        **kwargs: Any,
    ):
        super().__init__(out_size, in_size, rpu_config, bias, **kwargs)
        self.periph_quant = rpu_config.pre_post.periph_quant

        # Quantization in tall and wide layers. The tall layers need to be
        # requantized because they were produced by accumulation
        # of partial results, and as such are no longer quantized in the
        # same bit-precision. The wide layers need to be quantized to equalize
        # the range across the different tiles and keep the precision down
        # in case of bias addition.
        if self.analog_tile_count > 1:
            if rpu_config.act_quant_config is not None and rpu_config.act_quant_config.n_bits > 0:
                self.module_out_quantizer = QuantizedActivation(
                    **convert_act_config_to_kwargs_dict(rpu_config.act_quant_config)
                )
                # Enable the quantization
                self.module_out_quantizer.quantized_acts()
            else:
                self.module_out_quantizer = None

        # Initialize the bias quantizer, if quantized periphery is defined
        if self.bias is not None:
            if rpu_config.pre_post.periph_quant.n_bits > 0:
                periph_quant = rpu_config.pre_post.periph_quant
                self.bias_quantizer = QuantizationManager(
                    qmethod=(
                        QMethods.symmetric_uniform
                        if periph_quant.symmetric
                        else QMethods.asymmetric_uniform
                    ),
                    qparams={"n_bits": periph_quant.n_bits},
                    init=RangeEstimators.current_minmax,
                )
                if periph_quant.learn_quant_params:
                    self.bias_quant_update_idx = Parameter(
                        full((1,), 0.0, device=self.device), requires_grad=False
                    )
            else:
                self.bias_quantizer = None

    def forward(self, x_input: Tensor, tensor_view: Optional[Tuple] = None) -> Tensor:
        """Compute the forward pass, quantizing the final result as appropriate"""
        # pylint: disable=arguments-differ,arguments-renamed,too-many-branches

        # Create the final result. In tall splits, perform the intermediate accumulation
        if self.analog_tile_count == 1:
            analog_tile = self.array[0][0]  # pylint: disable=unsubscriptable-object
            result = analog_tile(x_input)
        else:
            # mapped version
            last_dim = x_input.ndim - 1
            splits = split(x_input, self.in_sizes, dim=last_dim)
            result = None
            for idx, (x, in_tiles) in enumerate(zip(splits, self.array)):
                out_result = []

                for analog_tile in in_tiles:
                    out_result.append(analog_tile(x, tensor_view=tensor_view))

                if idx == 0:
                    result = cat(out_result, last_dim)
                else:
                    result.add_(cat(out_result, last_dim))

        # Add the bias
        if self.bias is not None:
            if tensor_view is None:
                tensor_view = analog_tile.get_tensor_view(result.dim())
            if self.bias_quantizer is None:
                result += self.bias.view(*tensor_view)
            else:
                # In the case of evaluation with uninitialized quantizer, take care of
                # estimating the ranges first and then fixing them
                if not self.training and not self.bias_quantizer.quantizer.is_initialized:
                    self.bias_quantizer.estimate_ranges()
                    q_bias = self.bias_quantizer(self.bias)
                    self.bias_quantizer.fix_ranges()
                    result += q_bias.view(*tensor_view)

                else:
                    if (
                        self.training
                        and self.periph_quant.learn_quant_params
                        and not self.bias_quantizer.is_learning()
                    ):
                        # If learning is enabled, estimate till `init_learning_after`
                        # before switching to learned
                        self.bias_quant_update_idx.data += 1  # count up to the desired batch
                        if self.bias_quant_update_idx > self.periph_quant.init_learning_after:
                            self.bias_quantizer.learn_ranges()  # Switch to learned

                    # Add to the result the quantized bias
                    result += self.bias_quantizer(self.bias).view(*tensor_view)

        # Quantize the final result post accumulation and bias addition
        if self.module_out_quantizer is not None:
            result = self.module_out_quantizer(result)

        return result

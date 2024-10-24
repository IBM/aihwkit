# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Implements analog tile module array ."""
from typing import Any, Optional, Tuple, List, TYPE_CHECKING

from torch import Tensor, cat, split, zeros
from torch.nn import ModuleList, Parameter, Module
from torch.autograd import no_grad

from aihwkit.simulator.tiles.base import TileModuleBase
from aihwkit.exceptions import TileModuleError

if TYPE_CHECKING:
    from aihwkit.simulator.configs.configs import MappableRPU


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

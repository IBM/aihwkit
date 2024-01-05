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

"""Analog Information utility.

This module prints relevant information about the model and its analog
execution.

"""

from functools import reduce
import operator
from typing import Optional, Any, List

# Imports from PyTorch.
from torch import zeros
from torch.nn import Module

from aihwkit.nn.modules.base import AnalogLayerBase
from aihwkit.nn.modules.conv_mapped import _AnalogConvNdMapped
from aihwkit.nn.modules.linear_mapped import AnalogLinearMapped
from aihwkit.nn.modules.conv import _AnalogConvNd
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.tiles.module import TileModule
from aihwkit.simulator.parameters.base import RPUConfigBase


COLUMN_DEFINITIONS = ["Layer Information", "Tile Information"]

COLUMN_NAMES = {
    "name": (0, "Layer Name"),
    "isanalog": (0, "Is Analog"),
    "input_size": (0, "In Shape"),
    "output_size": (0, "Out Shape"),
    "kernel_size": (0, "Kernel Shape"),
    "num_tiles": (0, "# of Tiles"),
    # "macs": "Multi_Adds",
    "log_shape": (1, "Log. tile shape"),
    "phy_shape": (1, "Phys. tile shape"),
    "utilization": (1, "utilization (%)"),
    "reuse_factor": (0, "Reuse Factor"),
}

FORMATTING_WIDTH = 200
COLUMN_WIDTH = 10
FLOAT_FORMAT = "{0:.2f}"


class TileInfo:
    """Class for storing tile statistics and information."""

    log_in_size: Any
    log_out_size: Any
    phy_in_size: Any
    phy_out_size: Any
    utilization: float

    def __init__(self, tile: TileModule, is_mapped: bool):
        self.log_in_size = tile.in_size
        self.log_out_size = tile.out_size
        self.phy_in_size = tile.rpu_config.mapping.max_input_size
        self.phy_out_size = tile.rpu_config.mapping.max_output_size
        self.is_mapped = is_mapped
        max_space = self.phy_in_size * self.phy_out_size
        log_space = self.log_in_size * self.log_out_size
        self.utilization = log_space * 100 / max_space if is_mapped else 100

    def tile_summary_dict(self) -> dict:
        """Return a dictionary with the tile info."""
        phys_shape = "N/A" if not self.is_mapped else (self.phy_out_size, self.phy_in_size)
        return {
            "log_shape": str((self.log_out_size, self.log_in_size)),
            "phys_shape": str(phys_shape),
            "utilization": self.utilization,
        }

    def __repr__(self) -> str:
        """Print Tile's information."""
        tile_info = self.tile_summary_dict().values()
        return "{:<20}{:<20}{:<20}\n".format(*(tile_info))


class LayerInfo:
    """Class for storing layer statistics and information."""

    # pylint: disable=too-many-instance-attributes
    module: Module
    name: str
    isanalog: bool
    num_tiles: int
    tiles_info: List[TileInfo]
    input_size: Any
    output_size: Any
    kernel_size: Any
    reuse_factor: int

    def __init__(
        self,
        module: Module,
        rpu_config: Optional[RPUConfigBase] = None,
        input_size: Any = None,
        output_size: Any = None,
    ):
        self.module = module
        self.name = self.module.__class__.__name__
        self.isanalog = isinstance(self.module, AnalogLayerBase)
        self.num_tiles = 0 if not self.isanalog else len(list(self.module.analog_tiles()))
        self.tiles_info = self.set_tiles_info()
        self.kernel_size = None
        self.reuse_factor = 0
        self.input_size, self.output_size = input_size, output_size
        self.rpu_config = rpu_config
        self.set_kernel_size()
        self.calculate_reuse_factor()

    def __set_reuse_factor(self, reuse_factor: int) -> None:
        """Private method to set layer's reuse factor attribute."""
        self.reuse_factor = reuse_factor

    def set_tiles_info(self) -> List[TileInfo]:
        """Create TileInfo objects for each tile of the layer."""
        tiles_info = []
        is_mapped = isinstance(self.module, AnalogLinearMapped)
        is_mapped = is_mapped or isinstance(self.module, _AnalogConvNdMapped)
        if isinstance(self.module, AnalogLayerBase):
            for tile in self.module.analog_tiles():
                tiles_info.append(TileInfo(tile, is_mapped))
        return tiles_info

    def set_kernel_size(self) -> None:
        """Set kernel size attribute."""
        if hasattr(self.module, "kernel_size"):
            self.kernel_size = self.module.kernel_size

    def calculate_reuse_factor(self) -> None:
        """Compute the reuse factor.

        The reuse factor is the number of vector matrix multiplication
        a layer computes.

        """
        if isinstance(self.module, (AnalogLinear, AnalogLinearMapped)):
            ruf = reduce(operator.mul, (self.input_size), 1) // int(self.input_size[-1])
            self.__set_reuse_factor(ruf)
        elif isinstance(self.module, (_AnalogConvNd, _AnalogConvNdMapped)):
            ruf = reduce(operator.mul, (self.output_size), 1) // self.output_size[1]
            self.__set_reuse_factor(ruf)

    def tiles_summary_dict(self) -> dict:
        """Return a dictionary with all tiles information."""
        tiles_summary = {}
        for tile in self.tiles_info:
            tiles_summary.update(tile.tile_summary_dict())
        return tiles_summary

    def layer_summary_dict(self) -> dict:
        """Return a dictionary with all layer's information."""

        analog = "analog" if self.isanalog else "digital"
        return {
            "name": self.name,
            "isanalog": analog,
            "input_size": str(self.input_size) if self.input_size is not None else "-",
            "output_size": str(self.output_size) if self.output_size is not None else "-",
            "kernel_size": str(self.kernel_size) if self.kernel_size is not None else "-",
            "num_tiles": self.num_tiles,
            "reuse_factor": str(self.reuse_factor) if self.reuse_factor is not None else "-",
            "log_shape": "-",
            "phy_shape": "-",
            "utilization": "-",
        }

    def __repr__(self) -> str:
        """Print layer's information in the summary table."""
        stats = self.layer_summary_dict().values()
        result = ("{:<20}" * len(stats)).format(*stats)
        result += "\n"
        for tile in self.tiles_info:
            tile_info = tile.tile_summary_dict()
            tile_info["utilization"] = FLOAT_FORMAT.format(tile_info["utilization"])
            result += " " * 20 * (len(stats) - 3)
            result += "{:<20}{:<20}{:<20}\n".format(*(tile_info.values()))
        return result


class AnalogInfo:
    """Class for computing and storing results of the analog summary."""

    def __init__(
        self, model: Module, input_size: Any = None, rpu_config: Optional[RPUConfigBase] = None
    ):
        self.model = model
        self.input_size = input_size
        self.rpu_config = rpu_config
        self.layer_summary = self.create_layer_summary()
        self.total_tile_number = self.calculate_num_tiles()
        self.total_nb_analog = self.calculate_num_analog()

    def register_hooks_recursively(self, module: Module, hook: Any) -> None:
        """Hooks the function into all layers with no children (or
        only analog tiles as childrens).
        """

        if len(list(module.children())) == 0:
            module.register_forward_hook(hook)
        elif (
            isinstance(module, AnalogLayerBase)
            and not module.IS_CONTAINER
            and len(
                [ch for ch in module.children() if isinstance(ch, AnalogLayerBase)]  # type: ignore
            )
            == 0
        ):  # type: ignore
            module.register_forward_hook(hook)  # type: ignore
        else:
            for layer in module.children():
                self.register_hooks_recursively(layer, hook)

    def create_layer_summary(self) -> List[LayerInfo]:
        """Create the layer summary list.

        This list contains LayerInfo elements that corresponds to each
        layer of the model.
        """
        layer_summary = []

        def get_size_hook(_: Module, _input: Any, _output: Any) -> None:
            nonlocal layer_summary
            input_size = list(_input[0].size())
            output_size = list(_output.size())
            layer_summary.append(LayerInfo(_, self.rpu_config, input_size, output_size))

        self.register_hooks_recursively(self.model, get_size_hook)
        device = next(self.model.parameters()).device
        dummy_var = zeros(self.input_size).to(device)
        self.model(dummy_var)
        return layer_summary

    def calculate_num_tiles(self) -> int:
        """Calculate the total number of tiles needed by the model."""
        total_tile_number = 0
        for x in self.layer_summary:
            total_tile_number += x.num_tiles
        return total_tile_number

    def calculate_num_analog(self) -> int:
        """Calculate the number of analog layers."""
        total_nb_analog = 0
        for x in self.layer_summary:
            total_nb_analog += 1 if x.isanalog else 0
        return total_nb_analog

    def __repr__(self) -> str:
        """Print summary results."""
        divider = "=" * FORMATTING_WIDTH + "\n"
        name = "Model Name: " + self.model.__class__.__name__ + "\n"
        result = divider + name + divider
        result += "Per-layer Information\n" + divider

        # Add header
        header = [*COLUMN_NAMES.values()]
        for i, category in enumerate(COLUMN_DEFINITIONS):
            header_i = [v for x, v in header if x == i]
            trim_length = COLUMN_WIDTH * len(header_i) - len(category)
            result += category + " " * trim_length
            if i == len(COLUMN_DEFINITIONS) - 1:
                break
            result += "| "
        result += "\n" + divider
        for i, category in enumerate(COLUMN_DEFINITIONS):
            header_i = [v for x, v in header if x == i]
            result += ("{:<20}" * len(header_i)).format(*header_i)
        result += "\n"

        for x in self.layer_summary:
            result += repr(x)

        result += divider
        result += "General Information\n" + divider
        result += "Total number of tiles: " + str(self.total_tile_number) + "\n"
        result += "Total number of analog layers: " + str(self.total_nb_analog) + "\n"
        return result


def analog_summary(
    model: Module, input_size: Optional[Any] = None, rpu_config: Optional[RPUConfigBase] = None
) -> AnalogInfo:
    """Summarize the given PyTorch model.

    Summarized information includes:

        1) Layer names,
        2) input/output shapes,
        3) kernel shape,
        4) # of digital parameters,
        5) # of analog parameters,
        6) # of analog tiles
        7) reuse factor

    Args:
        model: PyTorch model to run on the analog platform.

        input_size: required to run a forward pass of the model.

        rpu_config: resistive processing unit configuration.

    Returns:
        AnalogInfo Object.
    """
    results = AnalogInfo(model, input_size, rpu_config)
    return results

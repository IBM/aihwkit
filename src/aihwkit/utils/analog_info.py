"""Analog Information utility.

This module prints relevant information about the model and its analog execution.
"""

from functools import reduce
import operator
from typing import Optional, Any, Tuple, List

# Imports from PyTorch.
import torch
from torch import Tensor
from torch import nn

from aihwkit.nn.modules.base import AnalogModuleBase, RPUConfigAlias
from aihwkit.nn.modules.conv import _AnalogConvNd
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.tiles import BaseTile

COLUMN_DEFINITIONS = ["Layer Information", "Tile Information"]

COLUMN_NAMES = {
    "name": (0, "Layer Name"),
    "isanalog": (0, "Is Analog"),
    "input_size": (0, "In Shape"),
    "output_size": (0, "Out Shape"),
    "kernel_size": (0, "Kernel Shape"),
    "num_tiles": (0, "# of Tiles"),
    # "macs": "Multi_Adds",
    "log_in_size": (1, "in_size (log)"),
    "log_out_size": (1, "out_size (log)"),
    "utilization": (1, "utilization (%)"),
    "reuse_factor": (0, "Reuse Factor")
}

INPUT_SIZE_TYPE = Any
FORMATTING_WIDTH = 200
COLUMN_WIDTH = 20
FLOAT_FORMAT = "{0:.2f}"


class TileInfo:
    """
    class storing tile statistics and information.
    """
    log_in_size: INPUT_SIZE_TYPE
    log_out_size: INPUT_SIZE_TYPE
    phy_in_size: INPUT_SIZE_TYPE
    phy_out_size: INPUT_SIZE_TYPE
    utilization: float

    def __init__(self, tile: BaseTile):
        self.log_in_size = tile.in_size
        self.log_out_size = tile.out_size
        self.phy_in_size = tile.rpu_config.mapping.max_input_size
        self.phy_out_size = tile.rpu_config.mapping.max_output_size
        max_space = (self.phy_in_size*self.phy_out_size)
        self.utilization = (self.log_in_size*self.log_out_size) / max_space

    def tile_summary_dict(self) -> dict:
        """return a dictionary with the tile info."""
        return {"log_in_size": self.log_in_size,
                "log_out_size": self.log_out_size,
                "utilization": self.utilization,
                "phy_in_size": self.phy_in_size,
                "phy_out_size": self.phy_out_size
                }

    def __repr__(self) -> str:
        """Print Tile's information."""
        tile_info = self.tile_summary_dict().values()
        return "{:<20}{:<20}{:<20}\n".format(*(tile_info))


class LayerInfo:
    """
    class storing layer statistics and information.
    """
    # pylint: disable=too-many-instance-attributes
    module: nn.Module
    name: str
    isanalog: bool
    num_tiles: int
    tiles_info: List[TileInfo]
    input_size: INPUT_SIZE_TYPE
    output_size: INPUT_SIZE_TYPE
    kernel_size: INPUT_SIZE_TYPE
    reuse_factor: int

    def __init__(self, module: nn.Module, rpu_config: Optional[RPUConfigAlias] = None):
        self.module = module
        self.name = self.module.__class__.__name__
        self.isanalog = isinstance(self.module, AnalogModuleBase)
        self.num_tiles = 0 if not self.isanalog else self.module._analog_tile_counter
        self.tiles_info = self.set_tiles_info()
        self.kernel_size = None
        self.reuse_factor = 0
        self.input_size, self.output_size = None, None
        self.rpu_config = rpu_config

    def __set_layer_size(self,
                         input_size: Optional[INPUT_SIZE_TYPE],
                         output_size: Optional[INPUT_SIZE_TYPE]) -> None:
        """Private method to set layer's sizes attributes."""
        self.input_size = input_size
        self.output_size = output_size

    def __set_reuse_factor(self, reuse_factor: int) -> None:
        """Private method to set layer's reuse factor attribute."""
        self.reuse_factor = reuse_factor

    def calculate_size(self,
                       input_size: Optional[INPUT_SIZE_TYPE]
                       ) -> Tuple[INPUT_SIZE_TYPE, INPUT_SIZE_TYPE]:
        """Set input_size or output_size of the layer."""
        rand_tensor = torch.rand(input_size)
        output_size = self.module(rand_tensor).shape
        self.__set_layer_size(input_size, tuple(output_size))
        return input_size, tuple(output_size)

    def set_tiles_info(self) -> List[TileInfo]:
        """Create TileInfo objects for each tile of the layer."""
        tiles_info = []
        if isinstance(self.module, AnalogModuleBase):
            for tile in self.module.analog_tiles():
                tiles_info.append(TileInfo(tile))
        return tiles_info

    def set_kernel_size(self) -> None:
        """Set kernel size attribute."""
        if hasattr(self.module, 'kernel_size'):
            self.kernel_size = self.module.kernel_size

    def calculate_reuse_factor(self) -> None:
        """Compute the reuse factor.
        The reuse factor is the number of vector matrix multiplication a layer computes."""
        if isinstance(self.module, AnalogLinear):
            ruf = reduce(operator.mul, (self.input_size), 1) // int(self.input_size[-1])
            self.__set_reuse_factor(ruf)
        elif isinstance(self.module, _AnalogConvNd):
            # TODO: Extend the formula to ConvNd
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
        return {"name": self.name,
                "isanalog": analog,
                "input_size": str(self.input_size) if self.input_size is not None else "-",
                "output_size": str(self.output_size) if self.output_size is not None else "-",
                "kernel_size": str(self.kernel_size) if self.kernel_size is not None else "-",
                "num_tiles": self.num_tiles,
                "reuse_factor": str(self.reuse_factor) if self.reuse_factor is not None else "-",
                "log_in_size": "-",
                "log_out_size": "-",
                "utilization": "-"}

    def __repr__(self) -> str:
        """Print layer's information in the summary table."""
        stats = self.layer_summary_dict().values()
        result = (("{:<20}"*len(stats)).format(*stats))
        result += "\n"
        for tile in self.tiles_info:
            tile_info = tile.tile_summary_dict()
            tile_info["utilization"] = FLOAT_FORMAT.format(tile_info["utilization"])
            result += (" "*20*(len(stats)-3))
            result += ("{:<20}{:<20}{:<20}\n".format(*(tile_info.values())))
        return result


class AnalogInfo:
    """class for computing and storing results of the analog summary."""
    def __init__(self,
                 model: nn.Module,
                 input_size: INPUT_SIZE_TYPE = None,
                 rpu_config: Optional[RPUConfigAlias] = None):

        self.model = model
        self.input_size = input_size
        self.rpu_config = rpu_config
        self.layer_summary = self.create_layer_summary()
        self.total_tile_number = self.calculate_num_tiles()
        self.set_layer_sizes()

    def set_layer_sizes(self) -> None:
        """Set each layer input, output and kernel sizes."""
        parent_input_size = self.input_size
        for layer in self.layer_summary:
            layer.calculate_size(parent_input_size)
            parent_input_size = layer.output_size
            layer.set_kernel_size()
            layer.calculate_reuse_factor()

    def create_layer_summary(self) -> List[LayerInfo]:
        """Create the layer summary list.

        This list contains LayerInfo elements that corresponds to each layer of the model."""
        layer_summary = []
        if isinstance(self.model, tuple):
            layer_summary.append(LayerInfo(self.model[0], self.rpu_config))
        else:
            for module in self.model.modules():
                if not isinstance(module, nn.Sequential):
                    layer_summary.append(LayerInfo(module, self.rpu_config))

        return layer_summary

    def calculate_num_tiles(self) -> int:
        """Calculate the total number of tiles needed by the model."""
        total_tile_number = 0
        for x in self.layer_summary:
            total_tile_number += x.num_tiles
        return total_tile_number

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
            trim_length = (COLUMN_WIDTH*len(header_i) - len(category))
            result += category + " "*trim_length
            if i == len(COLUMN_DEFINITIONS)-1:
                break
            result += '| '
        result += "\n"+divider
        for i, category in enumerate(COLUMN_DEFINITIONS):
            header_i = [v for x, v in header if x == i]
            result += (("{:<20}"*len(header_i)).format(*header_i))
        result += "\n"

        for x in self.layer_summary:
            result += repr(x)

        return result


def analog_summary(model: nn.Module,
                   input_size: Optional[INPUT_SIZE_TYPE] = None,
                   rpu_config: Optional[RPUConfigAlias] = None) -> AnalogInfo:
    """
    Summarize the given PyTorch model. Summarized information includes:
        1) Layer names,
        2) input/output shapes,
        3) kernel shape,
        4) # of digital parameters,
        5) # of analog parameters,
        6) # of analog tiles
        7) reuse factor

    Args:
        model (nn.Module):
            PyTorch model to run on the analog platform.

        input_size:
            required to run a forward pass of the model.

        rpu_config:
            resistive processing unit configuration.
    Return:
        AnalogInfo Object.
    """
    results = AnalogInfo(model, input_size, rpu_config)
    print(results)
    return results

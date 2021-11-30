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

COLUMN_NAMES = {
    "name": "Layer Name",
    "isanalog": "Is Analog",
    "input_size": "In Shape",
    "output_size": "Out Shape",
    "kernel_size": "Kernel Shape",
    "num_tiles": "# of Tiles",
    #"macs": "Multi_Adds",
    "log_in_size": "in_size (log)", 
    "log_out_size": "out_size (log)", 
    "utilization" : "utilization", 
    "reuse_factor": "Reuse Factor"
}

INPUT_SIZE_TYPE = Any
FORMATTING_WIDTH = 150

class TileInfo: 
    """
    class storing tile statistics and information.
    """
    log_in_size: INPUT_SIZE_TYPE
    log_out_size: INPUT_SIZE_TYPE
    phy_in_size: INPUT_SIZE_TYPE
    phy_out_size: INPUT_SIZE_TYPE
    utilization: float 

    def __init__(self, tile):
        self.log_in_size = tile.in_size
        self.log_out_size = tile.out_size
        self.phy_in_size = tile.rpu_config.mapping.max_input_size
        self.phy_out_size = tile.rpu_config.mapping.max_output_size
        self.utilization = self.phy_in_size*self.phy_out_size - self.log_in_size*self.log_out_size

    def tile_summary_dict(self):
        """ return a dictionary with the tile info.""" 
        return {"log_in_size" : self.log_in_size, 
                "log_out_size": self.log_in_size, 
                #"phy_in_size": self.phy_in_size, 
                #"phy_out_size": self.phy_out_size, 
                "utilization": self.utilization
                }



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

    def set_tiles_info(self): 
        tiles_info = []
        if isinstance(self.module, AnalogModuleBase):
            for tile in self.module.analog_tiles():
                tiles_info.append(TileInfo(tile))
                print(tiles_info[0].log_in_size, tiles_info[0].log_out_size )
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
                "log_in_size": "",
                "log_out_size":"", 
                "utilization": "",
                "reuse_factor": str(self.reuse_factor) if self.reuse_factor is not None else "-"}

    def __repr__(self) -> str:
        """Print layer's information in the summary table."""
        stats = self.layer_summary_dict().values()
        result = ("{:<15} {:<10} {:<15} {:<15} {:<15} {:<10} {:<15} {:<15} {:<15} {:<15}\n".format(*stats))
        for tile in self.tiles_info: 
            tile_info = tile.tile_summary_dict().values()
            print(tile_info)
            result += ("{:>88} {:>15} {:>20}\n".format(*tile_info))
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
        header = COLUMN_NAMES.values()
        result += ("{:<15} {:<10} {:<15} {:<15} {:<15} {:<10} {:<15} {:<15} {:<15} {:<15}\n".format(*header))

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

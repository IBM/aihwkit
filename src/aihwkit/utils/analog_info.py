# Imports from PyTorch.
import torch
from torch import Tensor
from torch import nn

from aihwkit.nn.modules.base import AnalogModuleBase, RPUConfigAlias
from aihwkit.nn.modules.conv import _AnalogConvNd
from aihwkit.nn import AnalogLinear

from functools import reduce
import operator
from typing import Optional, Sequence, Union, Any, Tuple, List

COLUMN_NAMES= {
    "name": "Layer Name", 
    "isanalog" : "Analog/Digital", 
    "input_size" : "Input Shape", 
    "output_size" : "Output Shape", 
    "kernel_size" : "Kernel Shape", 
    "num_tiles" : "Number of Tiles", 
    "macs" : "Multi_Adds", 
    "reuse_factor" : "Reuse Factor"
}

INPUT_SIZE_TYPE = Any
FORMATTING_WIDTH = 150

class LayerInfo:
    """
    class storing layer statistics and information. 
    """

    module: nn.Module
    name: str
    isanalog: bool 
    num_tiles: int
    input_size: INPUT_SIZE_TYPE
    output_size: INPUT_SIZE_TYPE
    kernel_size: INPUT_SIZE_TYPE
    reuse_factor: int

    def __init__(self, module: nn.Module, rpu_config: Optional[RPUConfigAlias] = None):
        self.module = module
        self.name = self.module.__class__.__name__
        self.isanalog = isinstance(self.module, AnalogModuleBase)
        self.num_tiles = 0 if not self.isanalog else self.module._analog_tile_counter
        self.kernel_size = None
        self.reuse_factor = 0 
        self.input_size, self.output_size = None, None

    def __set_layer_size(self, input_size: Optional[INPUT_SIZE_TYPE], output_size: Optional[INPUT_SIZE_TYPE]) -> None: 
        """Private method to set layer's sizes attributes."""
        self.input_size = input_size
        self.output_size = output_size

    def __set_reuse_factor(self, reuse_factor: int) -> None: 
        """Private method to set layer's reuse factor attribute."""
        self.reuse_factor = reuse_factor

    def calculate_size(self, input_size: Optional[INPUT_SIZE_TYPE]) -> Tuple[Optional[INPUT_SIZE_TYPE], Optional[INPUT_SIZE_TYPE]]:
        """Set input_size or output_size of the layer."""
        rand_tensor = torch.rand(input_size)
        output_size = self.module(rand_tensor).shape
        self.__set_layer_size(input_size, tuple(output_size))
        return input_size, tuple(output_size)

    def set_kernel_size(self) -> None: 
        """Set kernel size attribute."""
        if hasattr(self.module, 'kernel_size'): 
            self.kernel_size = self.module.kernel_size

    def calculate_reuse_factor(self) -> None: 
        """Compute the reuse factor. 

        The reuse factor is the number of vector matrix multiplication a layer computes."""
        if isinstance(self.module, AnalogLinear):
            self.__set_reuse_factor(reduce(operator.mul, (self.input_size), 1) // int(self.input_size[-1]))
        elif isinstance(self.module, _AnalogConvNd): 
            # TODO: Extend the formula to ConvNd 
            self.__set_reuse_factor(reduce(operator.mul, (self.output_size), 1)// self.output_size[1])
    

    def layer_summary_dict(self) -> dict: 
        """Return a dictionary with all layer's information."""
        analog = "analog" if self.isanalog else "digital"
        return {"name": self.name, 
                "isanalog":analog, 
                "input_size" : str(self.input_size) if self.input_size != None else "-", 
                "output_size": str(self.output_size) if self.output_size != None else "-", 
                "kernel_size" : str(self.kernel_size) if self.kernel_size != None else "-",
                "num_tiles" : self.num_tiles, 
                "macs": "", 
                "reuse_factor": str(self.reuse_factor) if self.reuse_factor != None else "-"}

    def __repr__(self) -> str:
        """Print layer's information in the summary table."""
        result = ("{:<15} {:<15} {:<15} {:<15} {:<15} {:<20} {:<15} {:<15}\n".format(*self.layer_summary_dict().values()))
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
        result +=("{:<15} {:<15} {:<15} {:<15} {:<15} {:<20} {:<15} {:<15}\n".format(*COLUMN_NAMES.values()))
        
        
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

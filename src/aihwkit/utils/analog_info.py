# Imports from PyTorch.
import torch
from torch import Tensor
from torch import nn

from aihwkit.nn.modules.base import AnalogModuleBase
from aihwkit.nn.modules.conv import _AnalogConvNd
from aihwkit.nn import AnalogLinear

import logging
from math import prod
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

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

INPUT_SIZE_TYPE = Sequence[Union[int, Sequence[Any], torch.Size]]
FORMATTING_WIDTH = 150

class LayerInfo:
    """
    class storing layer statistics and information. 
    """
    def __init__(self, module:nn.Module, rpu_config):
        self.module = module
        self.name = self.module.__class__.__name__
        self.isanalog = isinstance(self.module, AnalogModuleBase)
        self.num_tiles = 0 if not self.isanalog else self.module._analog_tile_counter
        self.input_size, self.output_size = None, None
        self.kernel_size = None 
        self.reuse_factor = None

    def __set_layer_size(self, input_size, output_size): 
        self.input_size = input_size
        self.output_size = output_size

    def __set_reuse_factor(self, reuse_factor): 
        self.reuse_factor = reuse_factor

    def calculate_size(self, input_size):
        """Set input_size or output_size of the layer."""
        rand_tensor = torch.rand(input_size)
        output_size = self.module(rand_tensor).shape
        self.__set_layer_size(input_size, tuple(output_size))
        return input_size, tuple(output_size)

    def set_kernel_size(self): 
        try: 
            self.kernel_size = self.module.kernel_size
        except: 
            logging.info("Not a convolution operator.")

    def calculate_reuse_factor(self): 
        if isinstance(self.module, AnalogLinear):
            self.__set_reuse_factor(prod(self.input_size) // self.input_size[-1])
        elif isinstance(self.module, _AnalogConvNd): 
            # TODO: Extend the formula to ConvNd 
            self.__set_reuse_factor(prod(self.output_size) // self.output_size[1])
    

    def layer_summary_dict(self): 
        analog = "analog" if self.isanalog else "digital"
        return {"name": self.name, 
                "isanalog":analog, 
                "input_size" : str(self.input_size) if self.input_size != None else "-", 
                "output_size": str(self.output_size) if self.output_size != None else "-", 
                "kernel_size" : str(self.kernel_size) if self.kernel_size != None else "-",
                "num_tiles" : self.num_tiles, 
                "macs": "", 
                "reuse_factor": str(self.reuse_factor) if self.reuse_factor != None else "-"}

    def __repr__(self):
        result = ("{:<15} {:<15} {:<15} {:<15} {:<15} {:<20} {:<15} {:<15}\n".format(*self.layer_summary_dict().values()))
        return result
        
class AnalogInfo: 
    """class for computing and storing results of the analog summary."""
    def __init__(self, 
            model:nn.Module, 
            input_size: Optional[INPUT_SIZE_TYPE] = None, 
            rpu_config=None):

        self.model = model 
        self.input_size = input_size
        self.rpu_config = rpu_config
        self.layer_summary = self.create_layer_summary()
        self.total_tile_number = self.calculate_num_tiles()
        self.set_layer_sizes()

    def set_layer_sizes(self):
        parent_input_size = self.input_size
        for layer in self.layer_summary:
            layer.calculate_size(parent_input_size)
            parent_input_size = layer.output_size
            layer.set_kernel_size()
            layer.calculate_reuse_factor()
                    
    def create_layer_summary(self): 
        layer_summary = []
        if isinstance(self.model, tuple): 
            layer_summary.append(LayerInfo(self.model[0], self.rpu_config))
        else: 
            for module in self.model.modules():
                if not isinstance(module, nn.Sequential):
                    layer_summary.append(LayerInfo(module, self.rpu_config))

        return layer_summary

    def calculate_num_tiles(self):
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


def analog_summary(model: nn.Module, input_size: Optional[INPUT_SIZE_TYPE] = None, rpu_config=None):
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

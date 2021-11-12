# Imports from PyTorch.
import torch
from torch import Tensor
from torch import nn

#Imports from AIHWKit 
from aihwkit.nn.modules.base import AnalogModuleBase

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

COLUMN_NAMES = ["Layer Name", "Input Shape", "Output Shape", "Kernel Shape", "Num of Tiles"]

def analog_summary(model, rpu_config): 
    
# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Analog Modules that contain children Modules."""

# pylint: disable=unused-argument, arguments-differ

from typing import Any, Optional, Tuple
from collections import OrderedDict

from torch import Tensor
from torch.nn import Sequential, Module

from aihwkit.nn.modules.base import AnalogLayerBase
from aihwkit.exceptions import ModuleError


class AnalogSequential(AnalogLayerBase, Sequential):
    """An analog-aware sequential container.

    Specialization of torch ``nn.Sequential`` with extra functionality for
    handling analog layers:

    * apply analog-specific functions to all its children (drift and program
      weights).

    Note:
        This class is recommended to be used in place of ``nn.Sequential`` in
        order to correctly propagate the actions to all the children analog
        layers. If using regular containers, please be aware that operations
        need to be applied manually to the children analog layers when needed.
    """

    IS_CONTAINER: bool = True

    @classmethod
    def from_digital(cls, module: Sequential, *args: Any, **kwargs: Any) -> "AnalogSequential":
        """Construct AnalogSequential in-place from Sequential."""
        return cls(OrderedDict(mod for mod in module.named_children()))

    @classmethod
    def to_digital(cls, module: "AnalogSequential", *args: Any, **kwargs: Any) -> Sequential:
        """Construct Sequential in-place from AnalogSequential."""
        return Sequential(OrderedDict(mod for mod in module.named_children()))

    def get_weights(  # type: ignore
        self, **kwargs: Any
    ) -> "OrderedDict[str, Tuple[Tensor, Optional[Tensor]]]":
        """Returns all weights, bias tuples in an ordered dictionary.

        Args:
            kwargs: passed to the TileModule ``get_weights`` call

        Returns:
            All analog weight of all layers
        """

        weights_dic = OrderedDict()
        for name, analog_tile in self.named_analog_tiles():
            weights_dic[name] = analog_tile.get_weights(**kwargs)
        return weights_dic

    def set_weights(  # type: ignore
        self, weights_dic: "OrderedDict[str, Tuple[Tensor, Optional[Tensor]]]", **kwargs: Any
    ) -> None:
        """Returns a zeroed RNN state based on cell type and layer type

        Args:
            weights_dic: Ordered dictionary of weight data
            kwargs: passed to the TileModule ``set_weights`` call

        Raises:
            ModuleError: in case tile name cannot be found
        """
        for name, analog_tile in self.named_analog_tiles():
            if name not in weights_dic:
                raise ModuleError("Cannot find tile weight {} in given dictionary.".format(name))
            analog_tile.set_weights(*weights_dic[name], **kwargs)


class AnalogWrapper(AnalogLayerBase, Module):
    """Light-weight wrapper for analog models.

    This exposes the typical analog methods, such as ``analog_tiles``
    generator etc.

    Args:
        module: (analog) module to wrap
    """

    IS_CONTAINER: bool = True

    def __init__(self, module: Module):
        super().__init__()
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward calling the module forward."""
        return self.module(*args, **kwargs)

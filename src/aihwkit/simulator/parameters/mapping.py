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

# pylint: disable=too-many-instance-attributes

"""Mapping parameters for resistive processing units."""

from typing import ClassVar, Type
from dataclasses import dataclass, fields, field
from .base import RPUConfigBase
from .helpers import _PrintableMixin


@dataclass
class MappingParameter(_PrintableMixin):
    """Parameter related to hardware design and the mapping of logical
    weight matrices to physical tiles.

    Caution:

        Some of these parameters have only an effect for modules that
        support tile mappings.
    """

    digital_bias: bool = True
    """Whether the bias term is handled by the analog tile or kept in
    digital.

    Note:
        Default is having a *digital* bias so that bias values are
        *not* stored onto the analog crossbar. This needs to be
        supported by the chip design. Set to False if the analog bias
        is instead situated on the the crossbar itself (as an extra
        column)

    Note:
        ``digital_bias`` is supported by *all* analog modules.
    """

    weight_scaling_omega: float = 0.0
    """omega_scale is a user defined parameter used to scale the weights
    while remapping these to cover the full range of values allowed.
    By default, no remapping is performed. If values > 0.0 are supplied
    the abs-max of the weight is scaled to that value.
    """

    weight_scaling_columnwise: bool = False
    """Whether the weight matrix will be remapped column-wise over
    the maximum device allowed value."""

    weight_scaling_lr_compensation: bool = False
    """Whether to adjust the LR to compensate for the mapping factors
    that are not learned.

    The learning rate will be divided
    for a tile individually by the mean of the mapping scales that are
    determined by the ``weight_scaling_omega`` setting.

    Otherwise the gradient information will be divided isntead before
    the update.
    """

    learn_out_scaling: bool = False
    """Define (additional) out scales that are learnable parameter
    used to scale the output."""

    out_scaling_columnwise: bool = False
    """Whether the learnable out scaling parameter enabled by
    ``learn_out_scaling`` is a scalar (``False``) or learned for
    each output (``True``).
    """

    max_input_size: int = 512
    """Maximal input size (number of columns) of the weight matrix
    that is handled on a single analog tile.

    If the logical weight matrix size exceeds this size it will be
    split and mapped onto multiple analog tiles.

    Caution:
        Only relevant for ``Mapped`` modules such as
        :class:`aihwkit.nn.modules.linear_mapped.AnalogLinearMapped`.
    """

    max_output_size: int = 512
    """Maximal output size (number of rows) of the weight matrix
    that is handled on a single analog tile.

    If the logical weight matrix size exceeds this size it will be
    split and mapped onto multiple analog tiles.

    Caution:
        Only relevant for ``Mapped`` modules such as
        :class:`aihwkit.nn.modules.linear_mapped.AnalogLinearMapped`.
    """

    def compatible_with(self, mapping: "MappingParameter") -> bool:
        """Checks compatiblity

        Args:
            mapping: param to check

        Returns:
            success:  if compatible
        """
        if mapping == self:
            return True

        for key in fields(mapping):
            if key.name in [
                "weight_scaling_omega",
                "weight_scaling_columnwise",
                "weight_scaling_lr_compensation",
            ]:
                continue

            if mapping.__dict__[key.name] != self.__dict__[key.name]:
                return False
        return True


@dataclass
class MappableRPU(RPUConfigBase, _PrintableMixin):
    """Defines the mapping parameters and utility factories"""

    tile_array_class: ClassVar[Type]
    """Tile array class that correspond to the RPUConfig.

    This is used to build logical arrays of tiles. Needs to be defined
    in the derived class.
    """

    mapping: MappingParameter = field(default_factory=MappingParameter)
    """Parameter related to mapping weights to tiles for supporting modules."""

    def get_default_tile_module_class(self, out_size: int = 0, in_size: int = 0) -> Type:
        """Returns the default TileModule class."""
        if self.mapping.max_input_size == 0 and self.mapping.max_output_size == 0:
            return self.tile_class
        if self.mapping.max_input_size < in_size or self.mapping.max_output_size < out_size:
            return self.tile_array_class
        return self.tile_class

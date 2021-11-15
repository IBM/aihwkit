# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Analog mapped layers."""

from typing import Optional, Tuple, List

from torch import Tensor, cat, split, no_grad
from torch.nn import Linear

from aihwkit.nn.modules.base import RPUConfigAlias
from aihwkit.nn.functions import AnalogFunction
from aihwkit.nn.modules.base import AnalogModuleBase
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.exceptions import ModuleError


class AnalogLinearMapped(AnalogModuleBase, Linear):
    """Linear layer that uses an analog tile.

    Linear layer that uses an analog tile during its forward, backward
    and update passes. In contrast to
    :class:`~aihwkit.bb.modules.linear.Linear` the maximal in and/or
    out dimension can be restricted, in which case the linear layer is
    split into multiple parts and computed on multiple tiles of given
    max sizes.

    In contrast to :class:`~aihwkit.bb.modules.linear.Linear`, the
    bias vector (if requested) is always handled in digital (floating
    point).

    Note:
        Mapping is controlled by the :class:`aihwkit.simulator.configs.utils.MappingParameter`.

    Note:
        The tensor parameters of this layer (``.weight`` and ``.bias``) are not
        guaranteed to contain the same values as the internal weights and biases
        stored in the analog tile. Please use ``set_weights`` and
        ``get_weights`` when attempting to read or modify the weight/bias. This
        read/write process can simulate the (noisy and inexact) analog writing
        and reading of the resistive elements.

    Args:
        in_features: input vector size (number of columns).
        out_features: output vector size (number of rows).
        rpu_config: resistive processing unit configuration.
        bias: whether to use a bias row on the analog tile or not
        realistic_read_write: whether to enable realistic read/write
            for setting initial weights and read out of weights
        weight_scaling_omega: the weight value where the max
            weight will be scaled to. If zero, no weight scaling will
            be performed

    """
    # pylint: disable=abstract-method, too-many-locals, too-many-instance-attributes

    __constants__ = ['in_features', 'out_features', 'realistic_read_write', 'weight_scaling_omega',
                     'digital_bias', 'analog_bias', 'use_bias']
    in_features: int
    out_features: int
    realistic_read_write: bool
    weight_scaling_omega: float
    digital_bias: bool
    analog_bias: bool
    use_bias: bool
    in_sizes: List[int]
    out_sizes: List[int]

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False,
            weight_scaling_omega: float = 0.0,
            digital_bias: bool = True,
    ):

        AnalogModuleBase.__init__(self)

        if bias and not digital_bias:
            raise ModuleError("AnalogMappedLayer only supports digital bias.")

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.analog_bias = False
        self.digital_bias = bias

        self.realistic_read_write = realistic_read_write
        self.weight_scaling_omega = weight_scaling_omega

        if not rpu_config:
            rpu_config = SingleRPUConfig()

        # Call super() after tile creation, including ``reset_parameters``.
        Linear.__init__(self, in_features, out_features, bias=bias)

        # Create the tile(s)
        # More than one tile may need to be created. If so, divide
        # weight matrix into equal pieces along input dimension with
        # as many tiles as needed
        max_input_size = rpu_config.mapping.max_input_size
        max_output_size = rpu_config.mapping.max_output_size
        self.in_sizes = self.get_split_sizes(in_features, max_input_size)
        self.out_sizes = self.get_split_sizes(out_features, max_output_size)

        self.analog_tile_array = []
        for i, in_tile_size in enumerate(self.in_sizes):
            in_tiles = []
            for j, out_tile_size in enumerate(self.out_sizes):
                tile = rpu_config.tile_class(out_tile_size,
                                             in_tile_size,
                                             rpu_config,
                                             bias=self.analog_bias)
                self.register_analog_tile(tile, name=f"{i}_{j}")
                in_tiles.append(tile)
            self.analog_tile_array.append(in_tiles)

        # Set weights from the reset_parameters call (since only now the
        # analog_tiles are registered)
        self.set_weights(self.weight, self.bias)

        # Unregister weight/bias as a parameter but keep it as a
        # field (needed for syncing still)
        self.unregister_parameter('weight')

        if self.analog_bias:
            self.unregister_parameter('bias')

    def get_split_sizes(self, size: int, split_max_size: int) -> List[int]:
        """ Computed the split sizes.

        Args:
            size: number of elements of the layer in one dimension
            split_max_size: max size of the split

        Returns:
            List of split sizes
        """
        n_splits = (size + split_max_size - 1) // split_max_size
        base, extra = divmod(size, n_splits)
        return [base + (i < extra) for i in range(n_splits)]

    def set_weights(
            self,
            weight: Tensor,
            bias: Optional[Tensor] = None,
            force_exact: bool = False
    ) -> None:
        """Set the weight (and bias) with given Tensors.

        This uses an realistic write if the property ``realistic_read_write``
        of the layer is set, unless it is overwritten by ``force_exact``. It
        uses a scaled write if ``weight_scaling_omega`` is positive (see
        :meth:`~aihwkit.simulator.tiles.base.BaseTile.set_weights_scaled`).

        Note:
            This is the recommended way for setting the weight/bias matrix of
            the analog tile, as it will correctly store the weights into the
            internal memory. Directly writing to ``self.weight`` and
            ``self.bias`` might yield wrong results as they are not always in
            sync with the analog tile Parameters, for performance reasons.

        Args:
            weight: weight matrix
            bias: bias vector
            force_exact: forces an exact write to the analog tiles

        """
        shape = [self.out_features, self.in_features]
        weight = weight.clone().reshape(shape)

        realistic = self.realistic_read_write and not force_exact

        in_start = in_end = 0
        for in_size, in_tiles in zip(self.in_sizes, self.analog_tile_array):
            in_end += in_size
            out_start = out_end = 0
            for out_size, analog_tile in zip(self.out_sizes, in_tiles):
                out_end += out_size

                tile_weight = weight[out_start:out_end, in_start:in_end]

                if self.weight_scaling_omega > 0.0:
                    analog_tile.set_weights_scaled(tile_weight, None,
                                                   realistic=realistic,
                                                   omega=self.weight_scaling_omega)
                else:
                    analog_tile.set_weights(tile_weight, None, realistic=realistic)

                out_start = out_end
            in_start = in_end

        if self.use_bias and bias is not None:
            with no_grad():
                self.bias.data[:] = bias[:]

        self._sync_weights_from_tile()

    def get_weights(self, force_exact: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """Get the weight (and bias) tensors.

        This uses an realistic read if the property ``realistic_read_write`` of
        the layer is set, unless it is overwritten by ``force_exact``. It
        scales the analog weights by the digital alpha scale if
        ``weight_scaling_omega`` is positive (see
        :meth:`~aihwkit.simulator.tiles.base.BaseTile.get_weights_scaled`).

        Note:
            This is the recommended way for setting the weight/bias matrix from
            the analog tile, as it will correctly fetch the weights from the
            internal memory. Accessing ``self.weight`` and ``self.bias`` might
            yield wrong results as they are not always in sync with the
            analog tile library, for performance reasons.

        Args:
            force_exact: forces an exact read to the analog tiles

        Returns:
            tuple: weight matrix, bias vector

        """

        realistic = self.realistic_read_write and not force_exact

        weight_lst = []
        for in_tiles in self.analog_tile_array:
            in_tile_weight = []
            for analog_tile in in_tiles:
                if self.weight_scaling_omega > 0.0:
                    tile_weight, _ = analog_tile.get_weights_scaled(realistic=realistic)
                else:
                    tile_weight, _ = analog_tile.get_weights(realistic=realistic)
                in_tile_weight.append(tile_weight)
            weight_lst.append(cat(in_tile_weight, 0))

        weight = cat(weight_lst, 1)

        if self.digital_bias:
            with no_grad():
                return weight, self.bias.data.detach().cpu()
        return weight, None

    def reset_parameters(self) -> None:
        """Reset the parameters (weight and bias)."""
        super().reset_parameters()
        if self.analog_tile_count():
            self.set_weights(self.weight, self.bias)

    def forward(self, x_input: Tensor) -> Tensor:
        """Compute the forward pass."""
        # pylint: disable=arguments-differ,arguments-renamed

        if self.analog_tile_count() == 1:
            out = AnalogFunction.apply(
                self.analog_tile_array[0][0].get_analog_ctx(), x_input,
                self.analog_tile_array[0][0].shared_weights, not self.training)

            if self.digital_bias:
                return out + self.bias
            return out

        # mapped version
        last_dim = x_input.ndim - 1
        splits = split(x_input, self.in_sizes, dim=last_dim)
        result = None  # type: Tensor
        for idx, (x, in_tiles) in enumerate(zip(splits, self.analog_tile_array)):
            out_result = []

            for analog_tile in in_tiles:
                output = AnalogFunction.apply(
                    analog_tile.get_analog_ctx(), x,
                    analog_tile.shared_weights, not self.training)
                out_result.append(output)

            if idx == 0:
                result = cat(out_result, last_dim)
            else:
                result.add_(cat(out_result, last_dim))

        # add bias to final result
        if self.digital_bias:
            return result.add_(self.bias)
        return result

    def extra_repr(self) -> str:
        """Set the extra representation of the module.

        Returns:
            A string with the extra representation.
        """
        output = AnalogModuleBase.extra_repr(self)[:-1]
        output += ', mapping={}'.format((len(self.in_sizes), len(self.out_sizes)))

        return output

    @classmethod
    def from_digital(
            cls,
            module: Linear,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False,
            weight_scaling_omega: float = 0.0,
            digital_bias: bool = False,
    ) -> 'AnalogLinearMapped':
        """Return an AnalogLinearMapped layer from a torch Linear layer.

        Args:
            module: The torch module to convert. All layers that are
                defined in the ``conversion_map``.
            rpu_config: RPU config to apply to all converted tiles.
                Applied to all converted tiles.
            realistic_read_write: Whether to use closed-loop programming
                when setting the weights. Applied to all converted tiles.
            weight_scaling_omega: If non-zero, applied weights of analog
                layers will be scaled by ``weight_scaling_omega`` divided by
                the absolute maximum value of the original weight matrix.
            digital_bias: decide whether the bias term is handled by the analog tile
                or kept in digital.

                Note:
                    Make sure that the weight max and min setting of the
                    device support the desired analog weight range.

        Returns:
            an AnalogLinearMapped layer based on the digital Linear ``module``.
        """
        analog_module = cls(module.in_features,
                            module.out_features,
                            module.bias is not None,
                            rpu_config,
                            realistic_read_write,
                            weight_scaling_omega,
                            digital_bias)

        analog_module.set_weights(module.weight, module.bias)
        return analog_module

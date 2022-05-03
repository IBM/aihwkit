# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
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

from aihwkit.nn.functions import AnalogFunction
from aihwkit.nn.modules.base import AnalogModuleBase, RPUConfigAlias
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
            weight_scaling_omega: Optional[float] = None,
    ):

        # Call super() after tile creation, including ``reset_parameters``.
        Linear.__init__(self, in_features, out_features, bias=bias)

        # Create tiles
        if rpu_config is None:
            rpu_config = SingleRPUConfig()

        AnalogModuleBase.__init__(
            self,
            in_features,
            out_features,
            bias,
            realistic_read_write,
            rpu_config.mapping
        )
        if self.analog_bias:
            raise ModuleError("AnalogLinearMapped only supports digital bias.")

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

        # Set weights from the reset_parameters
        self.set_weights(self.weight, self.bias, remap_weights=True,
                         weight_scaling_omega=weight_scaling_omega)

        # Unregister weight/bias as a parameter but keep for sync
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
        if split_max_size <= 0:
            return [size]

        n_splits = (size + split_max_size - 1) // split_max_size
        base, extra = divmod(size, n_splits)
        return [base + (i < extra) for i in range(n_splits)]

    def set_weights(
            self,
            weight: Tensor,
            bias: Optional[Tensor] = None,
            force_exact: bool = False,
            remap_weights: bool = True,
            weight_scaling_omega: float = None
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
            remap_weights: Whether to rescale the given weight matrix
                and populate the digital output scaling factors as
                specified in the configuration
                :class:`~aihwkit.configs.utils.MappingParameter`. A
                new ``weight_scaling_omega`` can be given. Note that
                this will overwrite the existing digital out scaling
                factors.

                Note that each tile (in case of multiple mapped tiles)
                has it separate out scaling factors.

            weight_scaling_omega: The weight scaling omega factor (see
                :class:`~aihwkit.configs.utils.MappingParameter`). If
                given explicitly here, it will overwrite the value in
                the mapping field.

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

                if remap_weights:

                    omega = weight_scaling_omega
                    if omega is None:
                        omega = analog_tile.rpu_config.mapping.weight_scaling_omega

                    analog_tile.set_weights_scaled(
                        tile_weight, None,
                        realistic=realistic,
                        weight_scaling_omega=omega
                    )
                else:
                    analog_tile.set_weights(tile_weight, None, realistic=realistic)

                out_start = out_end
            in_start = in_end

        if self.digital_bias and bias is not None:
            with no_grad():
                self.bias.data[:] = bias[:]

        self._sync_weights_from_tile()

    def get_weights(self, force_exact: bool = False,
                    apply_out_scales: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
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
            apply_out_scales: Whether to return the weights with the
                (digital) output scaling factors applied. Note the
                "logical" weights of the layer which the DNN is
                effectively using are those with the output scales
                applied. If ``apply_out_scales`` is set to False, then
                only the weight values that is programmed onto the
                crossbar array are returned, without applying the
                digital scales.

        Returns:
            tuple: weight matrix, bias vector

        """

        realistic = self.realistic_read_write and not force_exact

        weight_lst = []
        for in_tiles in self.analog_tile_array:
            in_tile_weight = []
            for analog_tile in in_tiles:
                if apply_out_scales:
                    tile_weight, _ = analog_tile.get_weights_scaled(realistic=realistic)
                else:
                    tile_weight, _ = analog_tile.get_weights(realistic=realistic)
                in_tile_weight.append(tile_weight)
            weight_lst.append(cat(in_tile_weight, 0))

        weight = cat(weight_lst, 1)

        if self.digital_bias:
            with no_grad():
                return weight, self.bias.data.clone().detach().cpu()
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
            analog_tile = self.analog_tile_array[0][0]
            out = AnalogFunction.apply(
                analog_tile.get_analog_ctx(), x_input,
                analog_tile.shared_weights, not self.training)

            out = analog_tile.apply_out_scaling(out, (-1, ))

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

                output = analog_tile.apply_out_scaling(output, (-1, ))
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
        output = AnalogModuleBase.extra_repr(self)
        output += ', mapping={}'.format((len(self.in_sizes), len(self.out_sizes)))

        return output

    @classmethod
    def from_digital(
            cls,
            module: Linear,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False,
    ) -> 'AnalogLinearMapped':
        """Return an AnalogLinearMapped layer from a torch Linear layer.

        Args:
            module: The torch module to convert. All layers that are
                defined in the ``conversion_map``.
            rpu_config: RPU config to apply to all converted tiles.
                Applied to all converted tiles.
            realistic_read_write: Whether to use closed-loop programming
                when setting the weights. Applied to all converted tiles.

                Note:
                    Make sure that the weight max and min settings of the
                    device support the desired analog weight range.

        Returns:
            an AnalogLinearMapped layer based on the digital Linear ``module``.
        """
        analog_module = cls(module.in_features,
                            module.out_features,
                            module.bias is not None,
                            rpu_config,
                            realistic_read_write,
                            )

        analog_module.set_weights(module.weight, module.bias)
        return analog_module

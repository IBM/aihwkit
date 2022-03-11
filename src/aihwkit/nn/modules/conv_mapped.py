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

"""Convolution layers."""

from typing import Optional, Tuple, Union, List

from torch import Tensor, arange, cat, float64, int32, ones, split, no_grad
from torch.nn import Unfold
from torch.nn.functional import pad
from torch.nn.modules.conv import _ConvNd, Conv1d, Conv2d, Conv3d
from torch.nn.modules.utils import _single, _pair, _triple

from aihwkit.nn.functions import AnalogIndexedFunction
from aihwkit.nn.modules.base import AnalogModuleBase, RPUConfigAlias
from aihwkit.exceptions import ModuleError
from aihwkit.simulator.configs import SingleRPUConfig


class _AnalogConvNdMapped(AnalogModuleBase, _ConvNd):
    """Base class for convolution layers with tile mapping.

    """

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'in_features', 'out_features',
                     'realistic_read_write', 'weight_scaling_omega',
                     'digital_bias', 'analog_bias', 'use_bias']
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    realistic_read_write: bool
    weight_scaling_omega: float
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    fold_indices: Tensor
    input_size: float
    in_features: int
    out_features: int
    digital_bias: bool
    analog_bias: bool
    use_bias: bool

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, ...],
            stride: Tuple[int, ...],
            padding: Tuple[int, ...],
            dilation: Tuple[int, ...],
            transposed: bool,
            output_padding: Tuple[int, ...],
            groups: int,
            bias: bool,
            padding_mode: str,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False,
            weight_scaling_omega: Optional[float] = None,
    ):
        # pylint: disable=too-many-arguments, too-many-locals
        if groups != 1:
            raise ValueError('Only one group is supported')
        if padding_mode != 'zeros':
            raise ValueError('Only "zeros" padding mode is supported')

        # Call super() after tile creation, including ``reset_parameters``.
        _ConvNd.__init__(self, in_channels, out_channels, kernel_size, stride,
                         padding, dilation, transposed, output_padding, groups, bias,
                         padding_mode)

        # Create tiles
        if rpu_config is None:
            rpu_config = SingleRPUConfig()

        AnalogModuleBase.__init__(
            self,
            self.get_tile_size(in_channels, groups, kernel_size),
            out_channels,
            bias,
            realistic_read_write,
            rpu_config.mapping
        )

        if self.analog_bias:
            raise ModuleError("AnalogConvNdMapped only supports digital bias.")

        if not rpu_config:
            rpu_config = SingleRPUConfig()

        max_input_size = rpu_config.mapping.max_input_size
        max_output_size = rpu_config.mapping.max_output_size
        kernel_elem = self.in_features // self.in_channels
        self.in_sizes = self.get_split_sizes(self.in_features, max_input_size, kernel_elem)
        self.out_sizes = self.get_split_sizes(self.out_features, max_output_size)

        self.analog_tile_array = []
        for i, in_tile_size in enumerate(self.in_sizes):
            in_tiles = []
            for j, out_tile_size in enumerate(self.out_sizes):
                tile = rpu_config.tile_class(out_tile_size,
                                             in_tile_size * kernel_elem,
                                             rpu_config,
                                             bias=self.analog_bias)
                self.register_analog_tile(tile, name=f"{i}_{j}")
                in_tiles.append(tile)
            self.analog_tile_array.append(in_tiles)

        # Set weights from the reset_parameters (since now the
        # analog_tiles are registered)
        self.set_weights(self.weight, self.bias, remap_weights=True,
                         weight_scaling_omega=weight_scaling_omega)

        # Set the index matrices.
        self.input_size = 0
        self.fold_indices_lst = []  # type: List[Tensor]
        self.tensor_view = (-1,)  # type: Tuple[int, ...]

        # Unregister weight/bias as a parameter but keep it as a
        # field (needed for syncing still)
        self.unregister_parameter('weight')
        if self.analog_bias:
            self.unregister_parameter('bias')

    def get_split_sizes(self, size: int, split_max_size: int, group_size: int = 1) -> List[int]:
        """ Computed the split sizes across channels.

        Args:
            size: number of elements of the layer in one dimension
            split_max_size: max size of the split
            group_size: minimal size of features that needs to stay on one tile

        Returns:
            List of split sizes (in split groups)

        Raises:
            ModuleError: Tiling weight matrices is always done across channels
                only. If the group_size is larger than the
                maximal tile size, mapping cannot be done
        """
        if split_max_size <= 0:
            return [size // group_size]

        if group_size > split_max_size:
            raise ModuleError("Tile size too small to fit a single group (kernel): " +
                              f"{group_size} > {split_max_size}")

        size_per_group = size // group_size
        split_max_per_group = split_max_size // group_size

        n_splits = (size_per_group + split_max_per_group - 1) // split_max_per_group
        base, extra = divmod(size_per_group, n_splits)
        return [(base + (i < extra)) for i in range(n_splits)]

    def get_tile_size(
            self,
            in_channels: int,
            groups: int,
            kernel_size: Tuple[int, ...]
    ) -> int:
        """Calculate the tile size."""
        raise NotImplementedError

    def get_image_size(self, size: int, i: int) -> int:
        """Calculate the output image sizes."""
        nom = (size + 2 * self.padding[i] - self.dilation[i] * (self.kernel_size[i] - 1) - 1)
        return nom // self.stride[i] + 1

    def reset_parameters(self) -> None:
        """Reset the parameters (weight and bias)."""
        super().reset_parameters()
        if self.analog_tile_count():
            self.set_weights(self.weight, self.bias)

    def _calculate_indexes(self, x_input: Tensor,
                           in_channels: int) -> Tuple[Tensor, List[int], int]:
        """Calculate and return the fold indexes and sizes.

        Args:
            x_input: input matrix
            in_channels: number of input channel

        Returns:
            fold_indices: indices for the analog tile
            image_sizes: image sizes for the analog tile
            input_size: size of the current input
        """
        raise NotImplementedError

    def recalculate_indexes(self, x_input: Tensor) -> None:
        """Calculate and set the indexes of the analog tile.

        Args:
            x_input: the input tensor.

        Raises:
            ModuleError: in case the input is not at least 3
            dimensional

        """
        self.input_size = x_input.numel() / x_input.size(0)
        if x_input.ndim < 3:
            raise ModuleError("Expect >2-dim inputs to convolutions")
        channel_dim = 1
        self.fold_indices_lst = []
        splits = split(x_input, self.in_sizes, dim=channel_dim)
        for x, in_channels, in_tiles in zip(splits, self.in_sizes, self.analog_tile_array):
            fold_indices, image_sizes, _ = self._calculate_indexes(x, in_channels)
            self.fold_indices_lst.append(fold_indices)

            for analog_tile in in_tiles:
                analog_tile.set_indexed(fold_indices, image_sizes)

    def forward(self, x_input: Tensor) -> Tensor:
        """Compute the forward pass."""
        # pylint: disable=arguments-differ,arguments-renamed

        input_size = x_input.numel() / x_input.size(0)
        if self.input_size != input_size:
            self.recalculate_indexes(x_input)

        if self.analog_tile_count() == 1:
            analog_tile = self.analog_tile_array[0][0]
            output = AnalogIndexedFunction.apply(
                analog_tile.get_analog_ctx(), x_input,
                analog_tile.shared_weights, not self.training)

            output = analog_tile.apply_out_scaling(output, self.tensor_view)

            if self.digital_bias:
                return output + self.bias.view(*self.tensor_view)
            return output

        # mapped version
        channel_dim = 1
        splits = split(x_input, self.in_sizes, dim=channel_dim)
        result = None  # type: Tensor
        for idx, (x, in_tiles) in enumerate(zip(splits, self.analog_tile_array)):
            out_result = []
            input_size = x.numel() / x.size(0)

            for analog_tile in in_tiles:
                output = AnalogIndexedFunction.apply(
                    analog_tile.get_analog_ctx(), x,
                    analog_tile.shared_weights, not self.training)

                output = analog_tile.apply_out_scaling(output, self.tensor_view)
                out_result.append(output)

            if idx == 0:
                result = cat(out_result, channel_dim)
            else:
                result.add_(cat(out_result, channel_dim))

        # add bias to final result
        if self.digital_bias:
            return result + self.bias.view(*self.tensor_view)
        return result

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
        # pylint: disable=too-many-locals
        realistic = self.realistic_read_write and not force_exact

        shape = [self.out_features, self.in_channels, self.in_features // self.in_channels]
        weight = weight.clone().reshape(shape)

        weight_splits = split(weight, self.in_sizes, dim=1)

        for in_tiles, in_weight in zip(self.analog_tile_array, weight_splits):
            out_start = out_end = 0
            in_weight = in_weight.reshape([self.out_features, -1])
            for out_size, analog_tile in zip(self.out_sizes, in_tiles):
                out_end += out_size
                tile_weight = in_weight[out_start:out_end, :]
                out_start = out_end

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

        self._sync_weights_from_tile()

        if self.digital_bias and bias is not None:
            with no_grad():
                self.bias.data[:] = bias[:]

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

    def extra_repr(self) -> str:
        """Set the extra representation of the module.

        Returns:
            A string with the extra representation.
        """
        output = AnalogModuleBase.extra_repr(self)[:-1]
        output += ', mapping={}'.format((len(self.in_sizes), len(self.out_sizes)))

        return output


class AnalogConv1dMapped(_AnalogConvNdMapped):
    """1D convolution layer that maps to analog tiles.

    Applies a 1D convolution over an input signal composed of several input
    planes, using an analog tile for its forward, backward and update passes.

    The module will split the weight matrix onto multiple tiles if
    necessary. Physical max tile sizes are specified with
    :class:`~aihwkit.simulator.configs.utils.MappingParameter` in the
    RPU configuration, see
    :class:`~aihwkit.simulator.configs.configs.RPUConfigAlias`.

    Note:
        The tensor parameters of this layer (``.weight`` and ``.bias``) are not
        guaranteed to contain the same values as the internal weights and biases
        stored in the analog tile. Please use ``set_weights`` and
        ``get_weights`` when attempting to read or modify the weight/bias. This
        read/write process can simulate the (noisy and inexact) analog writing
        and reading of the resistive elements.

    Args:
        in_channels: number of channels in the input image.
        out_channels: number of channels produced by the convolution.
        kernel_size: size of the convolving kernel.
        stride: stride of the convolution.
        padding: zero-padding added to both sides of the input.
        dilation: spacing between kernel elements.
        groups: number of blocked connections from input channels to output
            channels.
        bias: whether to use a bias row on the analog tile or not.
        padding_mode: padding strategy. Only ``'zeros'`` is supported.
        rpu_config: resistive processing unit configuration.
        realistic_read_write: whether to enable realistic read/write for
            setting initial weights and read out of weights.
        weight_scaling_omega: the weight value where the max weight will be
            scaled to. If zero, no weight scaling will be performed.
    """
    # pylint: disable=abstract-method

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple],
            stride: Union[int, Tuple] = 1,
            padding: Union[int, Tuple] = 0,
            dilation: Union[int, Tuple] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False,
            weight_scaling_omega: Optional[float] = None,
    ):
        # pylint: disable=too-many-arguments
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)

        if dilation != _single(1):
            raise ValueError('Only dilation = 1 is supported')

        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,  # type: ignore
            False, _single(0), groups, bias, padding_mode,
            rpu_config, realistic_read_write, weight_scaling_omega
        )

        self.tensor_view = (-1, 1)

    @classmethod
    def from_digital(
            cls,
            module: Conv1d,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False,
    ) -> 'AnalogConv1dMapped':
        """Return an AnalogConv1dMapped layer from a torch Conv1d layer.

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
            an AnalogConv1d layer based on the digital Conv1d ``module``.
        """
        analog_module = cls(module.in_channels,
                            module.out_channels,
                            module.kernel_size,
                            module.stride,
                            module.padding,
                            module.dilation,
                            module.groups,
                            module.bias is not None,
                            module.padding_mode,
                            rpu_config,
                            realistic_read_write,
                            )

        analog_module.set_weights(module.weight, module.bias)
        return analog_module

    def get_tile_size(
            self,
            in_channels: int,
            groups: int,
            kernel_size: Tuple[int, ...]
    ) -> int:
        """Calculate the tile size."""
        return (in_channels // groups) * kernel_size[0]

    def _calculate_indexes(self, x_input: Tensor,
                           in_channels: int) -> Tuple[Tensor, List[int], int]:
        """Calculate and return the fold indexes and sizes.

        Args:
            x_input: input matrix
            in_channels: number of input channel

        Returns:
            fold_indices: indices for the analog tile
            image_sizes: image sizes for the analog tile
            input_size: size of the current input
        """
        input_size = x_input.numel() / x_input.size(0)

        # pytorch just always uses NCHW order?
        fold_indices = arange(2, x_input.size(2) + 2, dtype=float64).detach()
        shape = [1] + [1] + list(x_input.shape[2:])
        fold_indices = fold_indices.reshape(*shape)
        if not all(item == 0 for item in self.padding):
            fold_indices = pad(fold_indices, pad=[self.padding[0], self.padding[0]],
                               mode='constant', value=0)
        unfold = fold_indices.unfold(2, self.kernel_size[0], self.stride[0]).clone()

        fold_indices = unfold.reshape(-1, self.kernel_size[0]).transpose(0, 1).flatten().round()

        # concatenate the matrix index for different channels
        fold_indices_orig = fold_indices.clone()
        for i in range(in_channels - 1):
            fold_indices_tmp = fold_indices_orig.clone()
            for j in range(fold_indices_orig.size(0)):
                if fold_indices_orig[j] != 0:
                    fold_indices_tmp[j] += (input_size / in_channels) * (i + 1)

            fold_indices = cat([fold_indices, fold_indices_tmp], dim=0).clone()

        fold_indices = fold_indices.to(dtype=int32)

        if self.analog_bias:
            out_image_size = fold_indices.numel() // (self.kernel_size[0])
            fold_indices = cat((fold_indices, ones(out_image_size, dtype=int32)), 0)

        fold_indices = fold_indices.to(x_input.device)

        x_height = x_input.size(2)
        d_height = self.get_image_size(x_height, 0)

        image_sizes = [in_channels, x_height, d_height]
        return (fold_indices, image_sizes, input_size)


class AnalogConv2dMapped(_AnalogConvNdMapped):
    """2D convolution layer that maps to analog tiles.

    Applies a 2D convolution over an input signal composed of several input
    planes, using an analog tile for its forward, backward and update passes.

    The module will split the weight matrix onto multiple tiles if
    necessary. Physical max tile sizes are specified with
    :class:`~aihwkit.simulator.configs.utils.MappingParameter` in the
    RPU configuration, see
    :class:`~aihwkit.simulator.configs.configs.RPUConfigAlias`.

    Note:
        The tensor parameters of this layer (``.weight`` and ``.bias``) are not
        guaranteed to contain the same values as the internal weights and biases
        stored in the analog tile. Please use ``set_weights`` and
        ``get_weights`` when attempting to read or modify the weight/bias. This
        read/write process can simulate the (noisy and inexact) analog writing
        and reading of the resistive elements.

    Args:
        in_channels: number of channels in the input image.
        out_channels: number of channels produced by the convolution.
        kernel_size: size of the convolving kernel.
        stride: stride of the convolution.
        padding: zero-padding added to both sides of the input.
        dilation: spacing between kernel elements.
        groups: number of blocked connections from input channels to output
            channels.
        bias: whether to use a bias row on the analog tile or not.
        padding_mode: padding strategy. Only ``'zeros'`` is supported.
        rpu_config: resistive processing unit configuration.
        realistic_read_write: whether to enable realistic read/write
            for setting initial weights and read out of weights.
        weight_scaling_omega: the weight value where the max weight will be
            scaled to. If zero, no weight scaling will be performed.
    """
    # pylint: disable=abstract-method

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple],
            stride: Union[int, Tuple] = 1,
            padding: Union[int, Tuple] = 0,
            dilation: Union[int, Tuple] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False,
            weight_scaling_omega: Optional[float] = None,
    ):
        # pylint: disable=too-many-arguments
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,  # type: ignore
            False, _pair(0), groups, bias, padding_mode,
            rpu_config, realistic_read_write, weight_scaling_omega
        )

        self.tensor_view = (-1, 1, 1)

    @classmethod
    def from_digital(
            cls,
            module: Conv2d,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False,
    ) -> 'AnalogConv2dMapped':
        """Return an AnalogConv2dMapped layer from a torch Conv2d layer.

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
            an AnalogConv2dMapped layer based on the digital Conv2d ``module``.
        """
        analog_module = cls(module.in_channels,
                            module.out_channels,
                            module.kernel_size,
                            module.stride,
                            module.padding,
                            module.dilation,
                            module.groups,
                            module.bias is not None,
                            module.padding_mode,
                            rpu_config,
                            realistic_read_write)

        analog_module.set_weights(module.weight, module.bias)
        return analog_module

    def get_tile_size(
            self,
            in_channels: int,
            groups: int,
            kernel_size: Tuple[int, ...]
    ) -> int:
        """Calculate the tile size."""
        return (in_channels // groups) * kernel_size[0] * kernel_size[1]

    def _calculate_indexes(self, x_input: Tensor,
                           in_channels: int) -> Tuple[Tensor, List[int], int]:
        """Calculate and return the fold indexes and sizes.

        Args:
            x_input: input matrix
            in_channels: number of input channel

        Returns:
            fold_indices: indices for the analog tile
            image_sizes: image sizes for the analog tile
            input_size: size of the current input
        """
        input_size = x_input.numel() / x_input.size(0)

        # pytorch just always uses NCHW order
        fold_indices = arange(2, input_size + 2, dtype=float64).detach()
        shape = [1] + list(x_input.shape[1:])
        fold_indices = fold_indices.reshape(*shape)
        unfold = Unfold(kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation)
        fold_indices = unfold(fold_indices).flatten().round().to(dtype=int32)

        if self.analog_bias:
            out_image_size = fold_indices.numel() // (self.kernel_size[0] * self.kernel_size[1])
            fold_indices = cat((fold_indices, ones(out_image_size, dtype=int32)), 0)

        fold_indices = fold_indices.to(x_input.device)

        x_height = x_input.size(2)
        x_width = x_input.size(3)

        d_height = self.get_image_size(x_height, 0)
        d_width = self.get_image_size(x_width, 1)

        image_sizes = [in_channels, x_height, x_width, d_height, d_width]
        return (fold_indices, image_sizes, input_size)


class AnalogConv3dMapped(_AnalogConvNdMapped):
    """3D convolution layer that maps to analog tiles.

    Applies a 3D convolution over an input signal composed of several input
    planes, using an analog tile for its forward, backward and update passes.

    The module will split the weight matrix onto multiple tiles if
    necessary. Physical max tile sizes are specified with
    :class:`~aihwkit.simulator.configs.utils.MappingParameter` in the
    RPU configuration, see
    :class:`~aihwkit.simulator.configs.configs.RPUConfigAlias`.

    Note:
        The tensor parameters of this layer (``.weight`` and ``.bias``) are not
        guaranteed to contain the same values as the internal weights and biases
        stored in the analog tile. Please use ``set_weights`` and
        ``get_weights`` when attempting to read or modify the weight/bias. This
        read/write process can simulate the (noisy and inexact) analog writing
        and reading of the resistive elements.

    Args:
        in_channels: number of channels in the input image.
        out_channels: number of channels produced by the convolution.
        kernel_size: size of the convolving kernel.
        stride: stride of the convolution.
        padding: zero-padding added to both sides of the input.
        dilation: spacing between kernel elements.
        groups: number of blocked connections from input channels to output
            channels.
        bias: whether to use a bias row on the analog tile or not.
        padding_mode: padding strategy. Only ``'zeros'`` is supported.
        rpu_config: resistive processing unit configuration.
        realistic_read_write: whether to enable realistic read/write
            for setting initial weights and read out of weights.
        weight_scaling_omega: the weight value where the max weight will be
            scaled to. If zero, no weight scaling will be performed.

    Raises:
        ModuleError: Tiling weight matrices is always done across channels
            only. If the kernel number of elements is larger than the
            maximal tile size, mapping cannot be done

    """
    # pylint: disable=abstract-method

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple],
            stride: Union[int, Tuple] = 1,
            padding: Union[int, Tuple] = 0,
            dilation: Union[int, Tuple] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False,
            weight_scaling_omega: Optional[float] = None,
    ):
        # pylint: disable=too-many-arguments
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        if dilation != _triple(1):
            raise ValueError('Only dilation = 1 is supported')

        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,  # type: ignore
            False, _triple(0), groups, bias, padding_mode,
            rpu_config, realistic_read_write, weight_scaling_omega
        )

        self.tensor_view = (-1, 1, 1, 1)

    @classmethod
    def from_digital(
            cls,
            module: Conv3d,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False,
    ) -> 'AnalogConv3dMapped':
        """Return an AnalogConv3dMapped layer from a torch Conv3d layer.

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
            an AnalogConv3d layer based on the digital Conv3d ``module``.
        """
        analog_module = cls(module.in_channels,
                            module.out_channels,
                            module.kernel_size,
                            module.stride,
                            module.padding,
                            module.dilation,
                            module.groups,
                            module.bias is not None,
                            module.padding_mode,
                            rpu_config,
                            realistic_read_write)

        analog_module.set_weights(module.weight, module.bias)
        return analog_module

    def get_tile_size(
            self,
            in_channels: int,
            groups: int,
            kernel_size: Tuple[int, ...]
    ) -> int:
        """Calculate the tile size."""
        return (in_channels // groups) * (
                kernel_size[0] * kernel_size[1] * kernel_size[2])

    def _calculate_indexes(self, x_input: Tensor,
                           in_channels: int) -> Tuple[Tensor, List[int], int]:
        """Calculate and return the fold indexes and sizes.

        Args:
            x_input: input matrix
            in_channels: number of input channel

        Returns:
            fold_indices: indices for the analog tile
            image_sizes: image sizes for the analog tile
            input_size: size of the current input
        """
        # pylint: disable=too-many-locals
        input_size = x_input.numel() / x_input.size(0)

        # pytorch just always uses NCDHW order
        fold_indices = arange(2, x_input.size(2) * x_input.size(3) * x_input.size(4) + 2,
                              dtype=float64).detach()
        shape = [1] + [1] + list(x_input.shape[2:])
        fold_indices = fold_indices.reshape(*shape)
        if not all(item == 0 for item in self.padding):
            fold_indices = pad(fold_indices, pad=[
                self.padding[2], self.padding[2],
                self.padding[1], self.padding[1],
                self.padding[0], self.padding[0]], mode='constant', value=0)
        unfold = fold_indices.unfold(2, self.kernel_size[0], self.stride[0]). \
            unfold(3, self.kernel_size[1], self.stride[1]). \
            unfold(4, self.kernel_size[2], self.stride[2]).clone()

        fold_indices = unfold.reshape(-1, self.kernel_size[0] * self.kernel_size[1] *
                                      self.kernel_size[2]).transpose(0, 1).flatten().round()

        # concatenate the matrix index for different channels
        fold_indices_orig = fold_indices.clone()
        for i in range(in_channels - 1):
            fold_indices_tmp = fold_indices_orig.clone()
            for j in range(fold_indices_orig.size(0)):
                if fold_indices_orig[j] != 0:
                    fold_indices_tmp[j] += (input_size / in_channels) * (i + 1)

            fold_indices = cat([fold_indices, fold_indices_tmp], dim=0).clone()

        fold_indices = fold_indices.to(dtype=int32)

        if self.analog_bias:
            out_image_size = fold_indices.numel() // (self.kernel_size[0] *
                                                      self.kernel_size[1] *
                                                      self.kernel_size[2])
            fold_indices = cat((fold_indices, ones(out_image_size, dtype=int32)), 0)

        fold_indices = fold_indices.to(x_input.device)

        x_depth = x_input.size(2)
        x_height = x_input.size(3)
        x_width = x_input.size(4)

        d_depth = self.get_image_size(x_depth, 0)
        d_height = self.get_image_size(x_height, 1)
        d_width = self.get_image_size(x_width, 2)

        image_sizes = [in_channels, x_depth, x_height, x_width, d_depth, d_height, d_width]
        return (fold_indices, image_sizes, input_size)

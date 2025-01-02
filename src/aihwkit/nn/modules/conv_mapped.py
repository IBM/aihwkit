# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Mapped convolution layers."""

# pylint: disable=too-many-arguments, too-many-locals, too-many-instance-attributes, too-many-lines

from typing import Optional, Tuple, Union, List, Type, Any

from torch import Tensor, arange, cat, float64, int32, split, no_grad
from torch.nn.functional import pad, unfold
from torch.nn.modules.conv import _ConvNd, Conv1d, Conv2d, Conv3d
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn import ModuleList

from aihwkit.nn.modules.base import AnalogLayerBase
from aihwkit.simulator.tiles.module import TileModule
from aihwkit.exceptions import AnalogBiasConfigError, ModuleError, ConfigError
from aihwkit.simulator.parameters.base import RPUConfigBase
from aihwkit.simulator.parameters.mapping import MappableRPU


class _AnalogConvNdMapped(AnalogLayerBase, _ConvNd):
    """Base class for convolution layers with tile mapping."""

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
        rpu_config: Optional[MappableRPU] = None,
        tile_module_class: Optional[Type] = None,
        use_indexed: Optional[bool] = None,
    ):
        if groups != 1:
            raise ValueError("Only one group is supported")
        if padding_mode != "zeros":
            raise ValueError("Only 'zeros' padding mode is supported")

        # Call super()
        _ConvNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias,
            padding_mode,
        )

        # Create tiles
        AnalogLayerBase.__init__(self)

        if rpu_config is None:
            # pylint: disable=import-outside-toplevel
            from aihwkit.simulator.configs.configs import SingleRPUConfig

            rpu_config = SingleRPUConfig()

        if tile_module_class is None:
            # Array might not make sense here since currently the
            # logical tiling is done in this module
            tile_module_class = rpu_config.tile_class

        if bias and not rpu_config.mapping.digital_bias:
            raise AnalogBiasConfigError("AnalogConvNdMapped only supports digital bias.")

        max_input_size = rpu_config.mapping.max_input_size  # type: ignore
        max_output_size = rpu_config.mapping.max_output_size  # type: ignore
        self.in_features = self.get_tile_size(in_channels, groups, kernel_size)
        self.out_features = out_channels
        kernel_elem = self.in_features // self.in_channels
        self.in_sizes = self.get_split_sizes(self.in_features, max_input_size, kernel_elem)

        self.out_sizes = self.get_split_sizes(out_channels, max_output_size)
        self.array = ModuleList()
        for in_tile_size in self.in_sizes:
            in_tiles = ModuleList()
            for out_tile_size in self.out_sizes:
                analog_tile = tile_module_class(
                    out_tile_size, in_tile_size * kernel_elem, rpu_config, bias=False
                )
                in_tiles.append(analog_tile)
            self.array.append(in_tiles)

        # Set weights from the reset_parameters (since now the
        # analog_tiles are registered)
        self.set_weights(self.weight, self.bias)

        # Set the index matrices.
        self.use_indexed = use_indexed
        if use_indexed is None:
            self.use_indexed = True
        self.use_indexed = self.use_indexed and analog_tile.supports_indexed

        self.input_size = 0
        self.fold_indices_lst = []  # type: List[Tensor]
        self.tensor_view = (-1,)  # type: Tuple[int, ...]

        # Unregister weight/bias as a parameter but keep it as a
        # field (needed for syncing still)
        self.unregister_parameter("weight")
        self.reset_parameters()

    def get_split_sizes(self, size: int, split_max_size: int, group_size: int = 1) -> List[int]:
        """Computed the split sizes across channels.

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
            raise ModuleError(
                "Tile size too small to fit a single group (kernel): "
                + f"{group_size} > {split_max_size}"
            )

        size_per_group = size // group_size
        split_max_per_group = split_max_size // group_size

        n_splits = (size_per_group + split_max_per_group - 1) // split_max_per_group
        base, extra = divmod(size_per_group, n_splits)
        return [(base + (i < extra)) for i in range(n_splits)]

    def get_tile_size(self, in_channels: int, groups: int, kernel_size: Tuple[int, ...]) -> int:
        """Calculate the tile size."""
        raise NotImplementedError

    def get_image_size(self, size: int, i: int) -> int:
        """Calculate the output image sizes."""
        nom = size + 2 * self.padding[i] - self.dilation[i] * (self.kernel_size[i] - 1) - 1
        return nom // self.stride[i] + 1

    def reset_parameters(self) -> None:
        """Reset the parameters (weight and bias)."""
        if hasattr(self, "array"):
            self.weight, _ = self.get_weights()
            super().reset_parameters()
            self.set_weights(self.weight, self.bias)
            self.weight = None

    def _calculate_indexes(
        self, x_input: Tensor, in_channels: int
    ) -> Tuple[Tensor, List[int], int]:
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

    def _recalculate_indexes(self, x_input: Tensor) -> None:
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
        for x, in_channels, in_tiles in zip(splits, self.in_sizes, self.array):
            fold_indices, image_sizes, _ = self._calculate_indexes(x, in_channels)
            self.fold_indices_lst.append(fold_indices)

            for analog_tile in in_tiles:
                analog_tile.set_indexed(fold_indices, image_sizes)

    def _single_unfold(self, analog_tile: "TileModule", x_input: Tensor) -> Tensor:
        """Forward using explicit unfolding (more suitable for CPUs)"""
        im_shape = x_input.shape

        x_input_ = unfold(
            x_input,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        ).transpose(1, 2)

        out = analog_tile(x_input_).transpose(1, 2)

        out_im_size = (
            im_shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1
        ) // self.stride[0] + 1
        return out.view(im_shape[0], analog_tile.out_size, out_im_size, -1)

    def forward(self, x_input: Tensor) -> Tensor:
        """Compute the forward pass."""
        # pylint: disable=arguments-differ, arguments-renamed, too-many-branches

        analog_tile = self.array[0][0]
        use_indexed = self.use_indexed and analog_tile.supports_indexed

        if use_indexed:
            input_size = x_input.numel() / x_input.size(0)
            if self.input_size != input_size or not analog_tile.is_indexed():
                self._recalculate_indexes(x_input)

        if self.analog_tile_count() == 1:
            if use_indexed:
                output = analog_tile(x_input)
            else:
                output = self._single_unfold(analog_tile, x_input)

            if self.bias is not None:
                return output + self.bias.view(*self.tensor_view)
            return output

        # Mapped version.
        channel_dim = 1
        splits = split(x_input, self.in_sizes, dim=channel_dim)
        result = None  # type: Tensor
        for idx, (x, in_tiles) in enumerate(zip(splits, self.array)):
            out_result = []

            for analog_tile in in_tiles:
                if use_indexed:
                    output = analog_tile(x)
                else:
                    output = self._single_unfold(analog_tile, x)
                out_result.append(output)

            if idx == 0:
                result = cat(out_result, channel_dim)
            else:
                result.add_(cat(out_result, channel_dim))

        # Add the bias (conditionally) to the final result.
        if self.bias is not None:
            return result + self.bias.view(*self.tensor_view)
        return result

    @no_grad()
    def set_weights(self, weight: Tensor, bias: Optional[Tensor] = None, **kwargs: Any) -> None:
        """Set the weight (and bias) tensors to the analog crossbar.

        Args:
            weight: weight matrix
            bias: bias vector
            **kwargs: see tile level,
                e.g. :meth:`~aihwkit.simulator.tiles.analog.AnalogTile.set_weights`
        """

        shape = [self.out_channels, self.in_channels, self.in_features // self.in_channels]
        weight = weight.clone().reshape(shape)

        weight_splits = split(weight, self.in_sizes, dim=1)

        for in_tiles, in_weight in zip(self.array, weight_splits):
            out_start = out_end = 0
            in_weight = in_weight.reshape([self.out_channels, -1])
            for out_size, analog_tile in zip(self.out_sizes, in_tiles):
                out_end += out_size
                tile_weight = in_weight[out_start:out_end, :]
                out_start = out_end

                analog_tile.set_weights(tile_weight, None, **kwargs)

        if self.bias is not None and bias is not None:
            with no_grad():
                self.bias.data[:] = bias[:]

    @no_grad()
    def get_weights(self, **kwargs: Any) -> Tuple[Tensor, Optional[Tensor]]:
        """Get the (analog) weight (and bias) tensors from the crossbar(s).

        Args:
            **kwargs: see tile level,
            e.g. :meth:`~aihwkit.simulator.tiles.analog.AnalogTile.get_weights`

        Returns:
            tuple: weight matrix, bias vector
        """

        weight_lst = []
        for in_tiles in self.array:
            in_tile_weight = []
            for analog_tile in in_tiles:
                tile_weight, _ = analog_tile.get_weights(**kwargs)
                in_tile_weight.append(tile_weight)
            weight_lst.append(cat(in_tile_weight, 0))

        weight = cat(weight_lst, 1)

        if self.bias is not None:
            return weight, self.bias.data.clone().detach().cpu()
        return weight, None

    def extra_repr(self) -> str:
        """Set the extra representation of the module.

        Returns:
            A string with the extra representation.
        """
        return AnalogLayerBase.extra_repr(self)


class AnalogConv1dMapped(_AnalogConvNdMapped):
    """1D convolution layer that maps to analog tiles.

    Applies a 1D convolution over an input signal composed of several input
    planes, using an analog tile for its forward, backward and update passes.

    The module will split the weight matrix onto multiple tiles if
    necessary. Physical max tile sizes are specified with
    :class:`~aihwkit.simulator.parameters.mapping.MappingParameter` in the
    RPU configuration, see
    :class:`~aihwkit.simulator.configs.configs.RPUConfigBase`.

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
        tile_module_class: Class for the tile module (default
            will be specified from the ``RPUConfig``).
    """

    # pylint: disable=abstract-method

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        rpu_config: Optional[MappableRPU] = None,
        tile_module_class: Optional[Type] = None,
    ):
        # pylint: disable=too-many-arguments
        kernel_size = _single(kernel_size)  # type: ignore
        stride = _single(stride)  # type: ignore
        padding = _single(padding)  # type: ignore
        dilation = _single(dilation)  # type: ignore

        if dilation != _single(1):
            raise ValueError("Only dilation = 1 is supported")

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,  # type: ignore
            stride,  # type: ignore
            padding,  # type: ignore
            dilation,  # type: ignore
            False,
            _single(0),
            groups,
            bias,
            padding_mode,
            rpu_config,
            tile_module_class,
            True,
        )

        self.tensor_view = (-1, 1)

    @classmethod
    def from_digital(
        cls, module: Conv1d, rpu_config: RPUConfigBase, tile_module_class: Optional[Type] = None
    ) -> "AnalogConv1dMapped":
        """Return an AnalogConv1dMapped layer from a torch Conv1d layer.

        Args:
            module: The torch module to convert. All layers that are
                defined in the ``conversion_map``.
            rpu_config: RPU config to apply to all converted tiles.
                Applied to all converted tiles.
            tile_module_class: Class for the tile module (default
                will be specified from the ``RPUConfig``).

        Returns:
            an AnalogConv1d layer based on the digital Conv1d ``module``.

        Raises:
            ConfigError: In case the ``RPUConfig`` is not of type ``MappableRPU``
        """

        if not isinstance(rpu_config, MappableRPU):
            raise ConfigError("Only mappable RPUConfigs are supported.")

        analog_layer = cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            module.padding_mode,
            rpu_config,
            tile_module_class,
        )

        analog_layer.set_weights(module.weight, module.bias)
        return analog_layer.to(module.weight.device)

    @classmethod
    def to_digital(cls, module: "AnalogConv1dMapped", realistic: bool = False) -> Conv1d:
        """Return an nn.Conv1d layer from an AnalogConv1dMapped layer.

        Args:
            module: The analog module to convert.
            realistic: whehter to estimate the weights with the
                non-ideal forward pass. If not set, analog weights are
                (unrealistically) copies exactly

        Returns:
            an torch Linear layer with the same dimension and weights
            as the analog linear layer.
        """
        weight, bias = module.get_weights(realistic=realistic)
        digital_layer = Conv1d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            bias is not None,
            module.padding_mode,
        )
        digital_layer.weight.data = weight.data.view(-1, module.in_channels, *module.kernel_size)
        if bias is not None:
            digital_layer.bias.data = bias.data
        return digital_layer

    def get_tile_size(self, in_channels: int, groups: int, kernel_size: Tuple[int, ...]) -> int:
        """Calculate the tile size."""
        return (in_channels // groups) * kernel_size[0]

    def _calculate_indexes(
        self, x_input: Tensor, in_channels: int
    ) -> Tuple[Tensor, List[int], int]:
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
            fold_indices = pad(
                fold_indices, pad=[self.padding[0], self.padding[0]], mode="constant", value=0
            )
        unfolded = fold_indices.unfold(2, self.kernel_size[0], self.stride[0]).clone()

        fold_indices = unfolded.reshape(-1, self.kernel_size[0]).transpose(0, 1).flatten().round()

        # concatenate the matrix index for different channels
        fold_indices_orig = fold_indices.clone()
        for i in range(in_channels - 1):
            fold_indices_tmp = fold_indices_orig.clone()
            for j in range(fold_indices_orig.size(0)):
                if fold_indices_orig[j] != 0:
                    fold_indices_tmp[j] += (input_size / in_channels) * (i + 1)

            fold_indices = cat([fold_indices, fold_indices_tmp], dim=0).clone()

        fold_indices = fold_indices.to(dtype=int32, device=x_input.device)
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
    :class:`~aihwkit.simulator.parameters.mapping.MappingParameter` in the
    RPU configuration, see
    :class:`~aihwkit.simulator.configs.configs.RPUConfigBase`.

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
        tile_module_class: Class for the tile module (default
            will be specified from the ``RPUConfig``).
        use_indexed: Whether to use explicit unfolding or implicit indexing. If
            None (default), it will use implicit indexing for CUDA and
            explicit unfolding for CPU
    """

    # pylint: disable=abstract-method

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        rpu_config: Optional[MappableRPU] = None,
        tile_module_class: Optional[Type] = None,
        use_indexed: Optional[bool] = None,
    ):
        # pylint: disable=too-many-arguments
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,  # type: ignore
            stride,  # type: ignore
            padding,  # type: ignore
            dilation,  # type: ignore
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            rpu_config,
            tile_module_class,
            use_indexed,
        )

        self.tensor_view = (-1, 1, 1)

    @classmethod
    def from_digital(
        cls, module: Conv2d, rpu_config: RPUConfigBase, tile_module_class: Optional[Type] = None
    ) -> "AnalogConv2dMapped":
        """Return an AnalogConv2dMapped layer from a torch Conv2d layer.

        Args:
            module: The torch module to convert. All layers that are
                defined in the ``conversion_map``.
            rpu_config: RPU config to apply to all converted tiles.
                Applied to all converted tiles.
            tile_module_class: Class for the tile module (default
                will be specified from the ``RPUConfig``).

        Returns:
            an AnalogConv2dMapped layer based on the digital Conv2d ``module``.

        Raises:
            ConfigError: In case the ``RPUConfig`` is not of type ``MappableRPU``
        """

        if not isinstance(rpu_config, MappableRPU):
            raise ConfigError("Only mappable RPUConfigs are supported.")

        analog_layer = cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            module.padding_mode,
            rpu_config,
            tile_module_class,
        )

        analog_layer.set_weights(module.weight, module.bias)
        return analog_layer.to(module.weight.device)

    @classmethod
    def to_digital(cls, module: "AnalogConv2dMapped", realistic: bool = False) -> Conv2d:
        """Return an nn.Conv2d layer from an AnalogConv2dMapped layer.

        Args:
            module: The analog module to convert.
            realistic: whehter to estimate the weights with the
                non-ideal forward pass. If not set, analog weights are
                (unrealistically) copies exactly

        Returns:
            an torch Linear layer with the same dimension and weights
            as the analog linear layer.
        """
        weight, bias = module.get_weights(realistic=realistic)
        digital_layer = Conv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            bias is not None,
            module.padding_mode,
        )
        digital_layer.weight.data = weight.data.view(-1, module.in_channels, *module.kernel_size)
        if bias is not None:
            digital_layer.bias.data = bias.data
        return digital_layer

    def get_tile_size(self, in_channels: int, groups: int, kernel_size: Tuple[int, ...]) -> int:
        """Calculate the tile size."""
        return (in_channels // groups) * kernel_size[0] * kernel_size[1]

    def _calculate_indexes(
        self, x_input: Tensor, in_channels: int
    ) -> Tuple[Tensor, List[int], int]:
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
        fold_indices = (
            unfold(
                fold_indices,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            .flatten()
            .round()
            .to(dtype=int32)
            .to(device=x_input.device)
        )

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
    :class:`~aihwkit.simulator.parameters.mapping.MappingParameter` in the
    RPU configuration, see
    :class:`~aihwkit.simulator.configs.configs.RPUConfigBase`.

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
        tile_module_class: Class for the tile module (default
            will be specified from the ``RPUConfig``).

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
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        rpu_config: Optional[MappableRPU] = None,
        tile_module_class: Optional[Type] = None,
    ):
        # pylint: disable=too-many-arguments
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        if dilation != _triple(1):
            raise ValueError("Only dilation = 1 is supported")

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,  # type: ignore
            stride,  # type: ignore
            padding,  # type: ignore
            dilation,  # type: ignore
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            rpu_config,
            tile_module_class,
            True,
        )

        self.tensor_view = (-1, 1, 1, 1)

    @classmethod
    def from_digital(
        cls, module: Conv3d, rpu_config: RPUConfigBase, tile_module_class: Optional[Type] = None
    ) -> "AnalogConv3dMapped":
        """Return an AnalogConv3dMapped layer from a torch Conv3d layer.

        Args:
            module: The torch module to convert. All layers that are
                defined in the ``conversion_map``.
            rpu_config: RPU config to apply to all converted tiles.
                Applied to all converted tiles.
            tile_module_class: Class for the tile module (default
                will be specified from the ``RPUConfig``).

        Returns:
            an AnalogConv3d layer based on the digital Conv3d ``module``.

        Raises:
            ConfigError: In case the ``RPUConfig`` is not of type ``MappableRPU``

        """

        if not isinstance(rpu_config, MappableRPU):
            raise ConfigError("Only mappable RPUConfigs are supported.")

        analog_layer = cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            module.padding_mode,
            rpu_config,
            tile_module_class,
        )

        analog_layer.set_weights(module.weight, module.bias)
        return analog_layer.to(module.weight.device)

    @classmethod
    def to_digital(cls, module: "AnalogConv3dMapped", realistic: bool = False) -> Conv3d:
        """Return an nn.Conv3d layer from an AnalogConv3dMapped layer.

        Args:
            module: The analog module to convert.
            realistic: whehter to estimate the weights with the
                non-ideal forward pass. If not set, analog weights are
                (unrealistically) copies exactly

        Returns:
            an torch Linear layer with the same dimension and weights
            as the analog linear layer.
        """
        weight, bias = module.get_weights(realistic=realistic)
        digital_layer = Conv3d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            bias is not None,
            module.padding_mode,
        )
        digital_layer.weight.data = weight.data.view(-1, module.in_channels, *module.kernel_size)
        if bias is not None:
            digital_layer.bias.data = bias.data
        return digital_layer

    def get_tile_size(self, in_channels: int, groups: int, kernel_size: Tuple[int, ...]) -> int:
        """Calculate the tile size."""
        return (in_channels // groups) * (kernel_size[0] * kernel_size[1] * kernel_size[2])

    def _calculate_indexes(
        self, x_input: Tensor, in_channels: int
    ) -> Tuple[Tensor, List[int], int]:
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
        fold_indices = arange(
            2, x_input.size(2) * x_input.size(3) * x_input.size(4) + 2, dtype=float64
        ).detach()
        shape = [1] + [1] + list(x_input.shape[2:])
        fold_indices = fold_indices.reshape(*shape)
        if not all(item == 0 for item in self.padding):
            fold_indices = pad(
                fold_indices,
                pad=[
                    self.padding[2],
                    self.padding[2],
                    self.padding[1],
                    self.padding[1],
                    self.padding[0],
                    self.padding[0],
                ],
                mode="constant",
                value=0,
            )
        unfolded = (
            fold_indices.unfold(2, self.kernel_size[0], self.stride[0])
            .unfold(3, self.kernel_size[1], self.stride[1])
            .unfold(4, self.kernel_size[2], self.stride[2])
            .clone()
        )

        fold_indices = (
            unfolded.reshape(-1, self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2])
            .transpose(0, 1)
            .flatten()
            .round()
        )

        # concatenate the matrix index for different channels
        fold_indices_orig = fold_indices.clone()
        for i in range(in_channels - 1):
            fold_indices_tmp = fold_indices_orig.clone()
            for j in range(fold_indices_orig.size(0)):
                if fold_indices_orig[j] != 0:
                    fold_indices_tmp[j] += (input_size / in_channels) * (i + 1)

            fold_indices = cat([fold_indices, fold_indices_tmp], dim=0).clone()

        fold_indices = fold_indices.to(device=x_input.device, dtype=int32)

        x_depth = x_input.size(2)
        x_height = x_input.size(3)
        x_width = x_input.size(4)

        d_depth = self.get_image_size(x_depth, 0)
        d_height = self.get_image_size(x_height, 1)
        d_width = self.get_image_size(x_width, 2)

        image_sizes = [in_channels, x_depth, x_height, x_width, d_depth, d_height, d_width]
        return (fold_indices, image_sizes, input_size)

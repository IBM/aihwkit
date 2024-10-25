# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Convolution layers."""

# pylint: disable=too-many-arguments, too-many-locals, too-many-instance-attributes

from typing import Optional, Tuple, Union, List, Type

from torch import Tensor, arange, cat, float64, int32, ones
from torch.autograd import no_grad
from torch.nn.functional import pad, unfold
from torch.nn.modules.conv import _ConvNd, Conv1d, Conv2d, Conv3d
from torch.nn.modules.utils import _single, _pair, _triple

from aihwkit.exceptions import ModuleError
from aihwkit.nn.modules.base import AnalogLayerBase
from aihwkit.simulator.parameters.base import RPUConfigBase


class _AnalogConvNd(AnalogLayerBase, _ConvNd):
    """Base class for convolution layers."""

    NEEDS_INDEXED = False

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
        rpu_config: Optional[RPUConfigBase] = None,
        tile_module_class: Optional[Type] = None,
        use_indexed: Optional[bool] = None,
    ):
        if groups != 1:
            raise ValueError("Only one group is supported")
        if padding_mode != "zeros":
            raise ValueError('Only "zeros" padding mode is supported')

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

        # Create the tile and set the analog.
        AnalogLayerBase.__init__(self)

        if rpu_config is None:
            # pylint: disable=import-outside-toplevel
            from aihwkit.simulator.configs.configs import SingleRPUConfig

            rpu_config = SingleRPUConfig()

        if tile_module_class is None:
            tile_module_class = rpu_config.get_default_tile_module_class()
        self.in_features = self.get_tile_size(in_channels, groups, kernel_size)
        self.out_features = out_channels
        self.analog_module = tile_module_class(
            self.out_features, self.in_features, rpu_config, bias
        )

        # Set the index matrices.
        self.use_indexed = use_indexed
        if not self.analog_module.supports_indexed:
            self.use_indexed = False

        self.fold_indices = Tensor().detach()
        self.input_size = 0
        self.tensor_view = (-1,)  # type: Tuple[int, ...]

        # Unregister weight/bias as a parameter but keep it for syncs
        self.unregister_parameter("weight")
        if bias:
            self.unregister_parameter("bias")
        else:
            # seems to be a torch bug
            self._parameters.pop("bias", None)
        self.bias = bias

        self.reset_parameters()

    def get_tile_size(self, in_channels: int, groups: int, kernel_size: Tuple[int, ...]) -> int:
        """Calculate the tile size."""
        raise NotImplementedError

    def get_image_size(self, size: int, i: int) -> int:
        """Calculate the output image sizes."""
        # pylint: disable=superfluous-parens
        nom = size + 2 * self.padding[i] - self.dilation[i] * (self.kernel_size[i] - 1) - 1
        return nom // self.stride[i] + 1

    def reset_parameters(self) -> None:
        """Reset the parameters (weight and bias)."""
        if hasattr(self, "analog_module"):
            bias = self.bias
            self.weight, self.bias = self.get_weights()  # type: ignore
            super().reset_parameters()
            self.set_weights(self.weight, self.bias)
            self.weight, self.bias = None, bias

    @no_grad()
    def _recalculate_indexes(self, x_input: Tensor) -> None:
        """Calculate and set the indexes of the analog tile."""

        self.fold_indices, image_sizes, self.input_size = self._calculate_indexes(
            x_input, self.in_channels
        )
        self.analog_module.set_indexed(self.fold_indices, image_sizes)

    @no_grad()
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

    def forward(self, x_input: Tensor) -> Tensor:
        """Compute the forward pass.

        Raises:
            ModuleError: in case indexed convolution is needed but not supported by the TileModule.
        """

        # Use indexed only in case of cuda.
        use_indexed = self.use_indexed
        if use_indexed is None and not self.NEEDS_INDEXED:
            use_indexed = self.analog_module.is_cuda
        if not use_indexed and self.NEEDS_INDEXED:
            raise ModuleError("Tile module does not support indexed computation.")
        if use_indexed:
            input_size = x_input.numel() / x_input.size(0)
            if self.input_size != input_size or not self.analog_module.is_indexed():
                self._recalculate_indexes(x_input)

            return self.analog_module(x_input, tensor_view=self.tensor_view)

        # Brute-force unfold.
        im_shape = x_input.shape
        x_input_ = unfold(
            x_input,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        ).transpose(-1, -2)

        out = self.analog_module(x_input_).transpose(-1, -2)
        out_size = (
            im_shape[-2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1
        ) // self.stride[0] + 1
        if len(im_shape) == 3:
            return out.view(self.out_channels, out_size, -1)
        return out.view(im_shape[0], self.out_channels, out_size, -1)


class AnalogConv1d(_AnalogConvNd):
    """1D convolution layer that uses an analog tile.

    Applies a 1D convolution over an input signal composed of several input
    planes, using an analog tile for its forward, backward and update passes.

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

    NEEDS_INDEXED = True

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
        padding_mode: str = "zeros",
        rpu_config: Optional["RPUConfigBase"] = None,
        tile_module_class: Optional[Type] = None,
    ):
        # pylint: disable=too-many-arguments
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)

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
        cls, module: Conv1d, rpu_config: "RPUConfigBase", tile_module_class: Optional[Type] = None
    ) -> "AnalogConv1d":
        """Return an AnalogConv1d layer from a torch Conv1d layer.

        Args:
            module: The torch module to convert. All layers that are
                defined in the ``conversion_map``.
            rpu_config: RPU config to apply to all converted tiles.
                Applied to all converted tiles.
            tile_module_class: Class for the tile module (default
                will be specified from the ``RPUConfig``).
        Returns:
            an AnalogConv1d layer based on the digital Conv1d ``module``.
        """
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
    def to_digital(cls, module: "AnalogConv1d", realistic: bool = False) -> Conv1d:
        """Return an nn.Conv1d layer from an AnalogConv1d layer.

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
        analog_tile = next(module.analog_tiles())
        return digital_layer.to(device=analog_tile.device, dtype=analog_tile.get_dtype())

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

        # pytorch just always uses NCHW order
        fold_indices = arange(2, x_input.size(2) + 2, dtype=float64).detach()
        shape = [1, 1] + list(x_input.shape[2:])
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

        fold_indices = fold_indices.to(dtype=int32)

        if self.analog_module.analog_bias:
            out_image_size = fold_indices.numel() // (self.kernel_size[0])
            fold_indices = cat((fold_indices, ones(out_image_size, dtype=int32)), 0)

        fold_indices = fold_indices.to(x_input.device)

        x_height = x_input.size(2)
        d_height = self.get_image_size(x_height, 0)

        image_sizes = [in_channels, x_height, d_height]
        return (fold_indices, image_sizes, input_size)


class AnalogConv2d(_AnalogConvNd):
    """2D convolution layer that uses an analog tile.

    Applies a 2D convolution over an input signal composed of several input
    planes, using an analog tile for its forward, backward and update passes.

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
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        padding: Union[int, Tuple] = 0,
        dilation: Union[int, Tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        rpu_config: Optional["RPUConfigBase"] = None,
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
        cls, module: Conv2d, rpu_config: "RPUConfigBase", tile_module_class: Optional[Type] = None
    ) -> "AnalogConv2d":
        """Return an AnalogConv2d layer from a torch Conv2d layer.

        Args:
            module: The torch module to convert. All layers that are
                defined in the ``conversion_map``.
            rpu_config: RPU config to apply to all converted tiles.
                Applied to all converted tiles.
            tile_module_class: Class for the tile module (default
                will be specified from the ``RPUConfig``).

        Returns:
            an AnalogConv2d layer based on the digital Conv2d ``module``.
        """
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
    def to_digital(cls, module: "AnalogConv2d", realistic: bool = False) -> Conv2d:
        """Return an nn.Conv2d layer from an AnalogConv2d layer.

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
        analog_tile = next(module.analog_tiles())
        return digital_layer.to(device=analog_tile.device, dtype=analog_tile.get_dtype())

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
        )

        if self.analog_module.analog_bias:
            out_image_size = fold_indices.numel() // (self.kernel_size[0] * self.kernel_size[1])
            fold_indices = cat((fold_indices, ones(out_image_size, dtype=int32)), 0)

        fold_indices = fold_indices.to(x_input.device)

        x_height = x_input.size(2)
        x_width = x_input.size(3)

        d_height = self.get_image_size(x_height, 0)
        d_width = self.get_image_size(x_width, 1)

        image_sizes = [in_channels, x_height, x_width, d_height, d_width]
        return (fold_indices, image_sizes, input_size)


class AnalogConv3d(_AnalogConvNd):
    """3D convolution layer that uses an analog tile.

    Applies a 3D convolution over an input signal composed of several input
    planes, using an analog tile for its forward, backward and update passes.

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

    NEEDS_INDEXED = True

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
        padding_mode: str = "zeros",
        rpu_config: Optional["RPUConfigBase"] = None,
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
        cls, module: Conv3d, rpu_config: "RPUConfigBase", tile_module_class: Optional[Type] = None
    ) -> "AnalogConv3d":
        """Return an AnalogConv3d layer from a torch Conv3d layer.

        Args:
            module: The torch module to convert. All layers that are
                defined in the ``conversion_map``.
            rpu_config: RPU config to apply to all converted tiles.
                Applied to all converted tiles.
            tile_module_class: Class for the tile module (default
                will be specified from the ``RPUConfig``).

        Returns:
            an AnalogConv3d layer based on the digital Conv3d ``module``.
        """
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
    def to_digital(cls, module: "AnalogConv3d", realistic: bool = False) -> Conv3d:
        """Return an nn.Conv3d layer from an AnalogConv3d layer.

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
        analog_tile = next(module.analog_tiles())
        return digital_layer.to(device=analog_tile.device, dtype=analog_tile.get_dtype())

    def get_tile_size(self, in_channels: int, groups: int, kernel_size: Tuple[int, ...]) -> int:
        """Calculate the tile size."""
        return (in_channels // groups) * (kernel_size[0] * kernel_size[1] * kernel_size[2])

    def _calculate_indexes(
        self, x_input: Tensor, in_channels: int
    ) -> Tuple[Tensor, List[int], int]:
        """Calculate and return the fold indexes and sizes.

        Args:
            x_input: input matrix
            in_channels: then number of in channels

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

        fold_indices = fold_indices.to(dtype=int32)

        if self.analog_module.analog_bias:
            out_image_size = fold_indices.numel() // (
                self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
            )
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

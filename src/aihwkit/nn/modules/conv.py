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

from torch import Tensor, arange, cat, float64, int32, ones
from torch.nn import Unfold
from torch.nn.functional import pad
from torch.nn.modules.conv import _ConvNd, Conv1d, Conv2d, Conv3d
from torch.nn.modules.utils import _single, _pair, _triple

from aihwkit.nn.functions import AnalogIndexedFunction
from aihwkit.nn.modules.base import AnalogModuleBase, RPUConfigAlias
from aihwkit.simulator.configs import SingleRPUConfig


class _AnalogConvNd(AnalogModuleBase, _ConvNd):
    """Base class for convolution layers."""

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

        # Create the tile and set the analog.
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
        self.analog_tile = self._setup_tile(rpu_config)

        # Register analog tile
        self.register_analog_tile(self.analog_tile)

        # Set weights from the reset_parameters
        self.set_weights(self.weight, self.bias, remap_weights=True,
                         weight_scaling_omega=weight_scaling_omega)

        # Set the index matrices.
        self.fold_indices = Tensor().detach()
        self.input_size = 0
        self.tensor_view = (-1,)  # type: Tuple[int, ...]

        # Unregister weight/bias as a parameter but keep it for syncs
        self.unregister_parameter('weight')
        if self.analog_bias:
            self.unregister_parameter('bias')

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

    def recalculate_indexes(self, x_input: Tensor) -> None:
        """Calculate and set the indexes of the analog tile."""

        self.fold_indices, image_sizes, self.input_size = \
            self._calculate_indexes(x_input, self.in_channels)
        self.analog_tile.set_indexed(self.fold_indices, image_sizes)

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

    def forward(self, x_input: Tensor) -> Tensor:
        """Compute the forward pass."""
        input_size = x_input.numel() / x_input.size(0)
        if not self.fold_indices.numel() or self.input_size != input_size:
            self.recalculate_indexes(x_input)

        out = AnalogIndexedFunction.apply(
            self.analog_tile.get_analog_ctx(), x_input,
            self.analog_tile.shared_weights, not self.training)

        out = self.analog_tile.apply_out_scaling(out, self.tensor_view)

        if self.digital_bias:
            return out + self.bias.view(*self.tensor_view)
        return out


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
        realistic_read_write: whether to enable realistic read/write
            for setting initial weights and during reading of the weights.
        weight_scaling_omega: If non-zero, the analog weights will be
            scaled by ``weight_scaling_omega`` divided by the absolute
            maximum value of the original weight matrix.
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
    ) -> 'AnalogConv1d':
        """Return an AnalogConv1d layer from a torch Conv1d layer.

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

        # pytorch just always uses NCHW order
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
        realistic_read_write: whether to enable realistic read/write
            for setting initial weights and read out of weights.
        weight_scaling_omega: If non-zero, the analog weights will be
            scaled by ``weight_scaling_omega`` divided by the absolute
            maximum value of the original weight matrix.
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
    ) -> 'AnalogConv2d':
        """Return an AnalogConv2d layer from a torch Conv2d layer.

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
            an AnalogConv2d layer based on the digital Conv2d ``module``.
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
        realistic_read_write: whether to enable realistic read/write
            for setting initial weights and read out of weights.
        weight_scaling_omega: If non-zero, the analog weights will be
            scaled by ``weight_scaling_omega`` divided by the absolute
            maximum value of the original weight matrix.
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
    ) -> 'AnalogConv3d':
        """Return an AnalogConv3d layer from a torch Conv3d layer.

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
        return (in_channels // groups) * (
                kernel_size[0] * kernel_size[1] * kernel_size[2])

    def _calculate_indexes(self, x_input: Tensor,
                           in_channels: int) -> Tuple[Tensor, List[int], int]:
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

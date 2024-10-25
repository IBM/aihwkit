# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=missing-function-docstring, too-few-public-methods, no-member

"""Layer helpers for aihwkit tests."""

from unittest import SkipTest

from torch.nn import LSTM as LSTM_nn
from torch.nn import GRU as GRU_nn
from torch.nn import RNN as RNN_nn

from aihwkit.nn import (
    AnalogConv1d,
    AnalogConv2d,
    AnalogConv3d,
    AnalogLinear,
    AnalogRNN,
    AnalogLSTMCell,
    AnalogLSTMCellCombinedWeight,
    AnalogGRUCell,
    AnalogVanillaRNNCell,
    AnalogLinearMapped,
    AnalogConv1dMapped,
    AnalogConv2dMapped,
    AnalogConv3dMapped,
)


class Linear:
    """AnalogLinear."""

    use_cuda = False

    def get_layer(self, in_features=3, out_features=4, rpu_config=None, **kwargs):
        kwargs.setdefault("bias", self.bias)

        if rpu_config is None:
            rpu_config = self.get_rpu_config()

        if not self.digital_bias and self.bias and hasattr(rpu_config, "simulator_tile_class"):
            raise SkipTest("Analog Bias is not supported.")

        rpu_config.mapping.max_input_size = 0
        rpu_config.mapping.max_output_size = 0
        rpu_config.mapping.digital_bias = self.digital_bias

        return AnalogLinear(in_features, out_features, rpu_config=rpu_config, **kwargs)


class LinearMapped:
    """AnalogLinearMapped."""

    use_cuda = False

    def get_layer(self, in_features=10, out_features=6, rpu_config=None, **kwargs):
        if not self.digital_bias and self.bias:
            raise SkipTest("Analog Bias is not supported.")

        kwargs.setdefault("bias", self.bias)

        if rpu_config is None:
            rpu_config = self.get_rpu_config()

        rpu_config.mapping.max_input_size = 4
        rpu_config.mapping.max_output_size = 3
        rpu_config.mapping.digital_bias = self.digital_bias

        return AnalogLinearMapped(in_features, out_features, rpu_config=rpu_config, **kwargs)


class Conv1d:
    """AnalogConv1d."""

    use_cuda = False

    def get_layer(self, in_channels=2, out_channels=3, rpu_config=None, **kwargs):
        kwargs.setdefault("bias", self.bias)
        kwargs.setdefault("kernel_size", [4])
        kwargs.setdefault("padding", 2)

        if rpu_config is None:
            rpu_config = self.get_rpu_config()

        if not self.digital_bias and self.bias and hasattr(rpu_config, "simulator_tile_class"):
            raise SkipTest("Analog Bias is not supported.")

        rpu_config.mapping.max_input_size = 0
        rpu_config.mapping.max_output_size = 0
        rpu_config.mapping.digital_bias = self.digital_bias

        return AnalogConv1d(in_channels, out_channels, rpu_config=rpu_config, **kwargs)


class Conv2d:
    """AnalogConv2d."""

    use_cuda = False

    def get_layer(self, in_channels=2, out_channels=3, rpu_config=None, **kwargs):
        kwargs.setdefault("bias", self.bias)
        kwargs.setdefault("kernel_size", [3, 3])
        kwargs.setdefault("padding", 2)

        if rpu_config is None:
            rpu_config = self.get_rpu_config()

        if not self.digital_bias and self.bias and hasattr(rpu_config, "simulator_tile_class"):
            raise SkipTest("Analog Bias is not supported.")

        rpu_config.mapping.max_input_size = 0
        rpu_config.mapping.max_output_size = 0
        rpu_config.mapping.digital_bias = self.digital_bias

        return AnalogConv2d(in_channels, out_channels, rpu_config=rpu_config, **kwargs)


class Conv3d:
    """AnalogConv3d."""

    use_cuda = False

    def get_layer(self, in_channels=2, out_channels=3, rpu_config=None, **kwargs):
        kwargs.setdefault("bias", self.bias)
        kwargs.setdefault("kernel_size", [2, 2, 2])
        kwargs.setdefault("padding", 2)

        if rpu_config is None:
            rpu_config = self.get_rpu_config()

        if not self.digital_bias and self.bias and hasattr(rpu_config, "simulator_tile_class"):
            raise SkipTest("Analog Bias is not supported.")

        rpu_config.mapping.max_input_size = 0
        rpu_config.mapping.max_output_size = 0
        rpu_config.mapping.digital_bias = self.digital_bias

        return AnalogConv3d(in_channels, out_channels, rpu_config=rpu_config, **kwargs)


class Conv1dMapped:
    """AnalogConv1dMapped."""

    use_cuda = False

    def get_layer(self, in_channels=2, out_channels=3, rpu_config=None, **kwargs):
        if not self.digital_bias and self.bias:
            raise SkipTest("Analog Bias is not supported.")

        kwargs.setdefault("bias", self.bias)
        kwargs.setdefault("kernel_size", [4])
        kwargs.setdefault("padding", 2)

        if rpu_config is None:
            rpu_config = self.get_rpu_config()

        rpu_config.mapping.max_input_size = 4
        rpu_config.mapping.max_output_size = 3
        rpu_config.mapping.digital_bias = self.digital_bias

        return AnalogConv1dMapped(in_channels, out_channels, rpu_config=rpu_config, **kwargs)


class Conv2dMapped:
    """AnalogConv2dMapped."""

    use_cuda = False

    def get_layer(self, in_channels=2, out_channels=3, rpu_config=None, **kwargs):
        if not self.digital_bias and self.bias:
            raise SkipTest("Analog Bias is not supported.")

        kwargs.setdefault("bias", self.bias)
        kwargs.setdefault("kernel_size", [3, 3])
        kwargs.setdefault("padding", 2)

        if rpu_config is None:
            rpu_config = self.get_rpu_config()

        rpu_config.mapping.max_input_size = 21
        rpu_config.mapping.max_output_size = 3
        rpu_config.mapping.digital_bias = self.digital_bias

        return AnalogConv2dMapped(in_channels, out_channels, rpu_config=rpu_config, **kwargs)


class Conv3dMapped:
    """AnalogConv3dMapped."""

    use_cuda = False

    def get_layer(self, in_channels=2, out_channels=3, rpu_config=None, **kwargs):
        if not self.digital_bias and self.bias:
            raise SkipTest("Analog Bias is not supported.")

        kwargs.setdefault("bias", self.bias)
        kwargs.setdefault("kernel_size", [2, 2, 2])
        kwargs.setdefault("padding", 2)

        if rpu_config is None:
            rpu_config = self.get_rpu_config()

        rpu_config.mapping.max_input_size = 21
        rpu_config.mapping.max_output_size = 3
        rpu_config.mapping.digital_bias = self.digital_bias

        return AnalogConv3dMapped(in_channels, out_channels, rpu_config=rpu_config, **kwargs)


class LSTM:
    """AnalogLSTM."""

    use_cuda = False

    def get_layer(self, input_size=2, hidden_size=3, rpu_config=None, **kwargs):
        kwargs.setdefault("bias", self.bias)

        if rpu_config is None:
            rpu_config = self.get_rpu_config()

        if not self.digital_bias and self.bias and hasattr(rpu_config, "simulator_tile_class"):
            raise SkipTest("Analog Bias is not supported.")

        rpu_config.mapping.max_input_size = 0
        rpu_config.mapping.max_output_size = 0
        rpu_config.mapping.digital_bias = self.digital_bias

        return AnalogRNN(AnalogLSTMCell, input_size, hidden_size, rpu_config=rpu_config, **kwargs)

    def get_native_layer_comparison(self, *args, **kwargs):
        return LSTM_nn(*args, **kwargs)


class LSTMCombinedWeight:
    """AnalogLSTM on a single RPU tile."""

    use_cuda = False

    def get_layer(self, input_size=2, hidden_size=3, rpu_config=None, **kwargs):
        kwargs.setdefault("bias", self.bias)

        if rpu_config is None:
            rpu_config = self.get_rpu_config()

        if not self.digital_bias and self.bias and hasattr(rpu_config, "simulator_tile_class"):
            raise SkipTest("Analog Bias is not supported.")

        rpu_config.mapping.max_input_size = 0
        rpu_config.mapping.max_output_size = 0
        rpu_config.mapping.digital_bias = self.digital_bias

        return AnalogRNN(
            AnalogLSTMCellCombinedWeight, input_size, hidden_size, rpu_config=rpu_config, **kwargs
        )

    def get_native_layer_comparison(self, *args, **kwargs):
        return LSTM_nn(*args, **kwargs)


class GRU:
    """AnalogGRU."""

    use_cuda = False

    def get_layer(self, input_size=2, hidden_size=3, rpu_config=None, **kwargs):
        kwargs.setdefault("bias", self.bias)

        if rpu_config is None:
            rpu_config = self.get_rpu_config()

        if not self.digital_bias and self.bias and hasattr(rpu_config, "simulator_tile_class"):
            raise SkipTest("Analog Bias is not supported.")

        rpu_config.mapping.max_input_size = 0
        rpu_config.mapping.max_output_size = 0
        rpu_config.mapping.digital_bias = self.digital_bias

        return AnalogRNN(AnalogGRUCell, input_size, hidden_size, rpu_config=rpu_config, **kwargs)

    def get_native_layer_comparison(self, *args, **kwargs):
        return GRU_nn(*args, **kwargs)


class VanillaRNN:
    """AnalogVanillaRNN."""

    use_cuda = False

    def get_layer(self, input_size=2, hidden_size=3, rpu_config=None, **kwargs):
        kwargs.setdefault("bias", self.bias)

        if rpu_config is None:
            rpu_config = self.get_rpu_config()

        if not self.digital_bias and self.bias and hasattr(rpu_config, "simulator_tile_class"):
            raise SkipTest("Analog Bias is not supported.")

        rpu_config.mapping.max_input_size = 0
        rpu_config.mapping.max_output_size = 0
        rpu_config.mapping.digital_bias = self.digital_bias

        return AnalogRNN(
            AnalogVanillaRNNCell, input_size, hidden_size, rpu_config=rpu_config, **kwargs
        )

    def get_native_layer_comparison(self, *args, **kwargs):
        return RNN_nn(*args, **kwargs)


class LinearCuda(Linear):
    """AnalogLinear."""

    use_cuda = True

    def get_layer(self, *args, **kwargs):
        return Linear.get_layer(self, *args, **kwargs).cuda()


class LinearMappedCuda(LinearMapped):
    """AnalogLinearMapped."""

    use_cuda = True

    def get_layer(self, *args, **kwargs):
        return LinearMapped.get_layer(self, *args, **kwargs).cuda()


class Conv1dCuda(Conv1d):
    """AnalogConv1dCuda."""

    use_cuda = True

    def get_layer(self, *args, **kwargs):
        return Conv1d.get_layer(self, *args, **kwargs).cuda()


class Conv2dCuda(Conv2d):
    """AnalogConv2dCuda."""

    use_cuda = True

    def get_layer(self, *args, **kwargs):
        return Conv2d.get_layer(self, *args, **kwargs).cuda()


class Conv3dCuda(Conv2d):
    """AnalogConv3dCuda."""

    use_cuda = True

    def get_layer(self, *args, **kwargs):
        return Conv3d.get_layer(self, *args, **kwargs).cuda()


class Conv1dMappedCuda(Conv1dMapped):
    """AnalogConv1dMappedCuda."""

    use_cuda = True

    def get_layer(self, *args, **kwargs):
        return Conv1dMapped.get_layer(self, *args, **kwargs).cuda()


class Conv2dMappedCuda(Conv2dMapped):
    """AnalogConv2dMappedCuda."""

    use_cuda = True

    def get_layer(self, *args, **kwargs):
        return Conv2dMapped.get_layer(self, *args, **kwargs).cuda()


class Conv3dMappedCuda(Conv3dMapped):
    """AnalogConv3dMappedCuda."""

    use_cuda = True

    def get_layer(self, *args, **kwargs):
        return Conv3dMapped.get_layer(self, *args, **kwargs).cuda()


class LSTMCuda(LSTM):
    """AnalogLSTMCuda."""

    use_cuda = True

    def get_layer(self, *args, **kwargs):
        return LSTM.get_layer(self, *args, **kwargs).cuda()

    def get_native_layer_comparison(self, *args, **kwargs):
        return LSTM.get_native_layer_comparison(self, *args, **kwargs).cuda()


class LSTMCombinedWeightCuda(LSTMCombinedWeight):
    """AnalogLSTMCuda."""

    use_cuda = True

    def get_layer(self, *args, **kwargs):
        return LSTMCombinedWeight.get_layer(self, *args, **kwargs).cuda()

    def get_native_layer_comparison(self, *args, **kwargs):
        return LSTMCombinedWeight.get_native_layer_comparison(self, *args, **kwargs).cuda()


class GRUCuda(GRU):
    """AnalogGRUCuda."""

    use_cuda = True

    def get_layer(self, *args, **kwargs):
        return GRU.get_layer(self, *args, **kwargs).cuda()

    def get_native_layer_comparison(self, *args, **kwargs):
        return GRU.get_native_layer_comparison(self, *args, **kwargs).cuda()


class VanillaRNNCuda(GRU):
    """AnalogVanillaRNNCuda."""

    use_cuda = True

    def get_layer(self, *args, **kwargs):
        return VanillaRNN.get_layer(self, *args, **kwargs).cuda()

    def get_native_layer_comparison(self, *args, **kwargs):
        return VanillaRNN.get_native_layer_comparison(self, *args, **kwargs).cuda()

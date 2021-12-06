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

# pylint: disable=missing-function-docstring, too-few-public-methods, no-member

"""Layer helpers for aihwkit tests."""

from unittest import SkipTest

from aihwkit.nn import (
    AnalogConv1d, AnalogConv2d, AnalogConv3d, AnalogLinear, AnalogLSTM, AnalogLinearMapped,
    AnalogConv1dMapped, AnalogConv2dMapped, AnalogConv3dMapped
)
from aihwkit.simulator.configs.utils import MappingParameter


class Linear:
    """AnalogLinear."""

    use_cuda = False

    def get_layer(self, in_features=3, out_features=4, **kwargs):
        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)
        kwargs['rpu_config'].mapping.digital_bias = self.digital_bias
        return AnalogLinear(in_features, out_features, **kwargs)


class LinearMapped:
    """AnalogLinearMapped."""

    use_cuda = False

    def get_rpu_config(self, **kwargs):
        kwargs.setdefault('mapping', MappingParameter(max_input_size=4, max_output_size=3))
        return super().get_rpu_config(**kwargs)

    def get_layer(self, in_features=10, out_features=6, **kwargs):
        if not self.digital_bias and self.bias:
            raise SkipTest('Analog Bias is not supported.')
        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)
        kwargs['rpu_config'].mapping.digital_bias = self.digital_bias
        return AnalogLinearMapped(in_features, out_features, **kwargs)


class Conv1d:
    """AnalogConv1d."""

    use_cuda = False

    def get_layer(self, in_channels=2, out_channels=3, **kwargs):
        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)
        kwargs.setdefault('kernel_size', [4])
        kwargs.setdefault('padding', 2)
        kwargs['rpu_config'].mapping.digital_bias = self.digital_bias
        return AnalogConv1d(in_channels, out_channels, **kwargs)


class Conv2d:
    """AnalogConv2d."""

    use_cuda = False

    def get_layer(self, in_channels=2, out_channels=3, **kwargs):
        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)
        kwargs.setdefault('kernel_size', [3, 3])
        kwargs.setdefault('padding', 2)
        kwargs['rpu_config'].mapping.digital_bias = self.digital_bias
        return AnalogConv2d(in_channels, out_channels, **kwargs)


class Conv3d:
    """AnalogConv3d."""

    use_cuda = False

    def get_layer(self, in_channels=2, out_channels=3, **kwargs):
        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)
        kwargs.setdefault('kernel_size', [2, 2, 2])
        kwargs.setdefault('padding', 2)
        kwargs['rpu_config'].mapping.digital_bias = self.digital_bias
        return AnalogConv3d(in_channels, out_channels, **kwargs)


class Conv1dMapped:
    """AnalogConv1dMapped."""

    use_cuda = False

    def get_rpu_config(self, **kwargs):
        kwargs.setdefault('mapping', MappingParameter(max_input_size=4, max_output_size=3))
        return super().get_rpu_config(**kwargs)

    def get_layer(self, in_channels=2, out_channels=3, **kwargs):

        if not self.digital_bias and self.bias:
            raise SkipTest('Analog Bias is not supported.')

        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)
        kwargs.setdefault('kernel_size', [4])
        kwargs.setdefault('padding', 2)
        kwargs['rpu_config'].mapping.digital_bias = self.digital_bias
        return AnalogConv1dMapped(in_channels, out_channels, **kwargs)


class Conv2dMapped:
    """AnalogConv2dMapped."""

    use_cuda = False

    def get_rpu_config(self, **kwargs):
        kwargs.setdefault('mapping', MappingParameter(max_input_size=21, max_output_size=3))
        return super().get_rpu_config(**kwargs)

    def get_layer(self, in_channels=2, out_channels=3, **kwargs):

        if not self.digital_bias and self.bias:
            raise SkipTest('Analog Bias is not supported.')

        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)
        kwargs.setdefault('kernel_size', [3, 3])
        kwargs.setdefault('padding', 2)
        kwargs['rpu_config'].mapping.digital_bias = self.digital_bias
        return AnalogConv2dMapped(in_channels, out_channels, **kwargs)


class Conv3dMapped:
    """AnalogConv3dMapped."""

    use_cuda = False

    def get_rpu_config(self, **kwargs):
        kwargs.setdefault('mapping', MappingParameter(max_input_size=21, max_output_size=3))
        return super().get_rpu_config(**kwargs)

    def get_layer(self, in_channels=2, out_channels=3, **kwargs):

        if not self.digital_bias and self.bias:
            raise SkipTest('Analog Bias is not supported.')

        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)
        kwargs.setdefault('kernel_size', [2, 2, 2])
        kwargs.setdefault('padding', 2)
        kwargs['rpu_config'].mapping.digital_bias = self.digital_bias
        return AnalogConv3dMapped(in_channels, out_channels, **kwargs)


class LSTM:
    """AnalogLSTM."""

    use_cuda = False

    def get_layer(self, input_size=2, hidden_size=3, **kwargs):
        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)
        kwargs['rpu_config'].mapping.digital_bias = self.digital_bias
        return AnalogLSTM(input_size, hidden_size, **kwargs)


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

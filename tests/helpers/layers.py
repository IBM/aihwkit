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

from aihwkit.nn import (
    AnalogConv1d, AnalogConv2d, AnalogConv3d, AnalogLinear, AnalogLSTM, AnalogLinearMapped
)
from aihwkit.simulator.configs.utils import MappingParameter


class Linear:
    """AnalogLinear."""

    use_cuda = False

    def get_layer(self, in_features=3, out_features=4, **kwargs):
        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)
        kwargs.setdefault('digital_bias', self.digital_bias)

        return AnalogLinear(in_features, out_features, **kwargs)


class LinearMapped:
    """AnalogLinearMapped."""

    use_cuda = False

    def get_rpu_config(self, **kwargs):
        kwargs.setdefault('mapping', MappingParameter(max_input_size=4, max_output_size=3))
        return super().get_rpu_config(**kwargs)

    def get_layer(self, in_features=10, out_features=6, **kwargs):

        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)

        return AnalogLinearMapped(in_features, out_features, **kwargs)


class Conv1d:
    """AnalogConv1d."""

    use_cuda = False

    def get_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2, **kwargs):
        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)
        kwargs.setdefault('digital_bias', self.digital_bias)

        return AnalogConv1d(in_channels, out_channels, kernel_size,
                            padding=padding,
                            **kwargs)


class Conv2d:
    """AnalogConv2d."""

    use_cuda = False

    def get_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2, **kwargs):
        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)
        kwargs.setdefault('digital_bias', self.digital_bias)

        return AnalogConv2d(in_channels, out_channels, kernel_size,
                            padding=padding,
                            **kwargs)


class Conv3d:
    """AnalogConv3d."""

    use_cuda = False

    def get_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2, **kwargs):
        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)
        kwargs.setdefault('digital_bias', self.digital_bias)

        return AnalogConv3d(in_channels, out_channels, kernel_size,
                            padding=padding,
                            **kwargs)


class LSTM:
    """AnalogLSTM."""

    use_cuda = False

    def get_layer(self, input_size=2, hidden_size=3, **kwargs):
        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)

        return AnalogLSTM(input_size, hidden_size, **kwargs)


class LinearCuda:
    """AnalogLinear."""

    use_cuda = True

    def get_layer(self, in_features=3, out_features=4, **kwargs):
        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)
        kwargs.setdefault('digital_bias', self.digital_bias)

        return AnalogLinear(in_features, out_features, **kwargs).cuda()


class LinearMappedCuda:
    """AnalogLinearMapped."""

    use_cuda = True

    def get_rpu_config(self, **kwargs):
        kwargs.setdefault('mapping', MappingParameter(max_input_size=4, max_output_size=3))
        return super().get_rpu_config(**kwargs)

    def get_layer(self, in_features=10, out_features=6, **kwargs):
        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)

        return AnalogLinearMapped(in_features, out_features, **kwargs).cuda()


class Conv1dCuda:
    """AnalogConv1d."""

    use_cuda = True

    def get_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2, **kwargs):
        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)
        kwargs.setdefault('digital_bias', self.digital_bias)

        return AnalogConv1d(in_channels, out_channels, kernel_size,
                            padding=padding,
                            **kwargs).cuda()


class Conv2dCuda:
    """AnalogConv2d."""

    use_cuda = True

    def get_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2, **kwargs):
        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)
        kwargs.setdefault('digital_bias', self.digital_bias)

        return AnalogConv2d(in_channels, out_channels, kernel_size,
                            padding=padding,
                            **kwargs).cuda()


class Conv3dCuda:
    """AnalogConv3d."""

    use_cuda = True

    def get_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2, **kwargs):
        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)
        kwargs.setdefault('digital_bias', self.digital_bias)

        return AnalogConv3d(in_channels, out_channels, kernel_size,
                            padding=padding,
                            **kwargs).cuda()


class LSTMCuda:
    """AnalogLSTM."""

    use_cuda = True

    def get_layer(self, input_size=2, hidden_size=3, **kwargs):
        kwargs.setdefault('rpu_config', self.get_rpu_config())
        kwargs.setdefault('bias', self.bias)

        return AnalogLSTM(input_size, hidden_size, **kwargs).cuda()

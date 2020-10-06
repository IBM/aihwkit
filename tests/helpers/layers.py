# -*- coding: utf-8 -*-

# (C) Copyright 2020 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-function-docstring,too-few-public-methods

"""Layer helpers for aihwkit tests."""

from aihwkit.nn import AnalogConv2d, AnalogLinear


class Linear:
    """AnalogLinear."""

    use_cuda = False

    def get_layer(self, in_features=3, out_features=4, **kwargs):
        kwargs.setdefault('resistive_device', self.get_resistive_device())
        kwargs.setdefault('bias', self.bias)

        return AnalogLinear(in_features, out_features, **kwargs)


class Conv2d:
    """AnalogConv2d."""

    use_cuda = False

    def get_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2, **kwargs):
        kwargs.setdefault('resistive_device', self.get_resistive_device())
        kwargs.setdefault('bias', self.bias)

        return AnalogConv2d(in_channels, out_channels, kernel_size,
                            padding=padding,
                            **kwargs)


class LinearCuda:
    """AnalogLinear."""

    use_cuda = True

    def get_layer(self, in_features=3, out_features=4, **kwargs):
        kwargs.setdefault('resistive_device', self.get_resistive_device())
        kwargs.setdefault('bias', self.bias)

        return AnalogLinear(in_features, out_features, **kwargs).cuda()


class Conv2dCuda:
    """AnalogConv2d."""

    use_cuda = True

    def get_layer(self, in_channels=2, out_channels=3, kernel_size=4, padding=2, **kwargs):
        kwargs.setdefault('resistive_device', self.get_resistive_device())
        kwargs.setdefault('bias', self.bias)

        return AnalogConv2d(in_channels, out_channels, kernel_size,
                            padding=padding,
                            **kwargs).cuda()

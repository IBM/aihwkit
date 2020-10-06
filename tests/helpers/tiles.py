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

"""Tile helpers for aihwkit tests."""

# pylint: disable=missing-function-docstring,too-few-public-methods

from aihwkit.simulator.devices import (FloatingPointResistiveDevice,
                                       IdealResistiveDevice,
                                       ConstantStepResistiveDevice,
                                       LinearStepResistiveDevice,
                                       ExpStepResistiveDevice,
                                       PulsedResistiveDevice,
                                       DifferenceUnitCell,
                                       VectorUnitCell,
                                       TransferUnitCell)
from aihwkit.simulator.tiles import AnalogTile, FloatingPointTile
from aihwkit.simulator.parameters import (ConstantStepResistiveDeviceParameters,
                                          LinearStepResistiveDeviceParameters,
                                          ExpStepResistiveDeviceParameters,
                                          SoftBoundsResistiveDeviceParameters,
                                          AnalogTileInputOutputParameters)
from aihwkit.simulator.rpu_base import tiles


class FloatingPoint:
    """FloatingPointTile."""

    simulator_tile_class = tiles.FloatingPointTile
    first_hidden_field = None
    use_cuda = False

    def get_resistive_device(self):
        return FloatingPointResistiveDevice()

    def get_tile(self, out_size, in_size, resistive_device=None, **kwargs):
        resistive_device = resistive_device or self.get_resistive_device()
        return FloatingPointTile(out_size, in_size, resistive_device, **kwargs)


class Ideal:
    """AnalogTile with IdealResistiveDevice."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = None
    use_cuda = False

    def get_resistive_device(self):
        return IdealResistiveDevice()

    def get_tile(self, out_size, in_size, resistive_device=None, **kwargs):
        resistive_device = resistive_device or self.get_resistive_device()
        return AnalogTile(out_size, in_size, resistive_device, **kwargs)


class ConstantStep:
    """AnalogTile with ConstantStepResistiveDevice."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound'
    use_cuda = False

    def get_resistive_device(self):
        return ConstantStepResistiveDevice(
            ConstantStepResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, resistive_device=None, **kwargs):
        resistive_device = resistive_device or self.get_resistive_device()
        return AnalogTile(out_size, in_size, resistive_device, **kwargs)


class LinearStep:
    """AnalogTile with LinearStepResistiveDevice."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound'
    use_cuda = False

    def get_resistive_device(self):
        return LinearStepResistiveDevice(
            LinearStepResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, resistive_device=None, **kwargs):
        resistive_device = resistive_device or self.get_resistive_device()
        return AnalogTile(out_size, in_size, resistive_device, **kwargs)


class Pulsed:
    """AnalogTile with PulsedResistiveDevice."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound'
    use_cuda = False

    def get_resistive_device(self):
        return PulsedResistiveDevice(
            LinearStepResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, resistive_device=None, **kwargs):
        resistive_device = resistive_device or self.get_resistive_device()
        return AnalogTile(out_size, in_size, resistive_device, **kwargs)


class ExpStep:
    """AnalogTile with ExpStepResistiveDevice."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound'
    use_cuda = False

    def get_resistive_device(self):
        return ExpStepResistiveDevice(
            ExpStepResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, resistive_device=None, **kwargs):
        resistive_device = resistive_device or self.get_resistive_device()
        return AnalogTile(out_size, in_size, resistive_device, **kwargs)


class Vector:
    """AnalogTile with VectorUnitCell."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound_0'
    use_cuda = False

    def get_resistive_device(self):
        return VectorUnitCell(
            [ConstantStepResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0),
             ConstantStepResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0)])

    def get_tile(self, out_size, in_size, resistive_device=None, **kwargs):
        resistive_device = resistive_device or self.get_resistive_device()
        return AnalogTile(out_size, in_size, resistive_device, **kwargs)


class Difference:
    """AnalogTile with DifferenceUnitCell."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound_0'
    use_cuda = False

    def get_resistive_device(self):
        return DifferenceUnitCell(
            ConstantStepResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0)
        )

    def get_tile(self, out_size, in_size, resistive_device=None, **kwargs):
        resistive_device = resistive_device or self.get_resistive_device()
        return AnalogTile(out_size, in_size, resistive_device, **kwargs)


class Transfer:
    """AnalogTile with TransferUnitCell."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound_0'
    use_cuda = False

    def get_resistive_device(self):
        return TransferUnitCell(
            [SoftBoundsResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0),
             SoftBoundsResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0)],
            params_transfer_forward=AnalogTileInputOutputParameters(is_perfect=True)
        )

    def get_tile(self, out_size, in_size, resistive_device=None, **kwargs):
        resistive_device = resistive_device or self.get_resistive_device()
        return AnalogTile(out_size, in_size, resistive_device, **kwargs)


class FloatingPointCuda:
    """FloatingPointTile."""

    simulator_tile_class = getattr(tiles, 'CudaFloatingPointTile', None)
    first_hidden_field = None
    use_cuda = True

    def get_resistive_device(self):
        return None

    def get_tile(self, out_size, in_size, resistive_device=None, **kwargs):
        return FloatingPointTile(out_size, in_size, resistive_device, **kwargs).cuda()


class IdealCuda:
    """AnalogTile with IdealResistiveDevice."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = None
    use_cuda = True

    def get_resistive_device(self):
        return IdealResistiveDevice()

    def get_tile(self, out_size, in_size, resistive_device=None, **kwargs):
        resistive_device = resistive_device or self.get_resistive_device()
        return AnalogTile(out_size, in_size, resistive_device, **kwargs).cuda()


class ConstantStepCuda:
    """AnalogTile with ConstantStepResistiveDevice."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound'
    use_cuda = True

    def get_resistive_device(self):
        return ConstantStepResistiveDevice(
            ConstantStepResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, resistive_device=None, **kwargs):
        resistive_device = resistive_device or self.get_resistive_device()
        return AnalogTile(out_size, in_size, resistive_device, **kwargs).cuda()


class LinearStepCuda:
    """AnalogTile with LinearStepResistiveDevice."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound'
    use_cuda = True

    def get_resistive_device(self):
        return LinearStepResistiveDevice(
            LinearStepResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, resistive_device=None, **kwargs):
        resistive_device = resistive_device or self.get_resistive_device()
        return AnalogTile(out_size, in_size, resistive_device, **kwargs).cuda()


class PulsedCuda:
    """AnalogTile with PulsedResistiveDevice."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound'
    use_cuda = True

    def get_resistive_device(self):
        return PulsedResistiveDevice(
            LinearStepResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, resistive_device=None, **kwargs):
        resistive_device = resistive_device or self.get_resistive_device()
        return AnalogTile(out_size, in_size, resistive_device, **kwargs).cuda()


class ExpStepCuda:
    """AnalogTile with ExpStepResistiveDevice."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound'
    use_cuda = True

    def get_resistive_device(self):
        return ExpStepResistiveDevice(
            ExpStepResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, resistive_device=None, **kwargs):
        resistive_device = resistive_device or self.get_resistive_device()
        return AnalogTile(out_size, in_size, resistive_device, **kwargs).cuda()


class VectorCuda:
    """AnalogTile with VectorUnitCell."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound_0'
    use_cuda = True

    def get_resistive_device(self):
        return VectorUnitCell(
            [ConstantStepResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0),
             ConstantStepResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0)])

    def get_tile(self, out_size, in_size, resistive_device=None, **kwargs):
        resistive_device = resistive_device or self.get_resistive_device()
        return AnalogTile(out_size, in_size, resistive_device, **kwargs).cuda()


class DifferenceCuda:
    """AnalogTile with DifferenceUnitCell."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound_0'
    use_cuda = True

    def get_resistive_device(self):
        return DifferenceUnitCell(
            ConstantStepResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0)
        )

    def get_tile(self, out_size, in_size, resistive_device=None, **kwargs):
        resistive_device = resistive_device or self.get_resistive_device()
        return AnalogTile(out_size, in_size, resistive_device, **kwargs).cuda()


class TransferCuda:
    """AnalogTile with TransferUnitCell."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound_0'
    use_cuda = True

    def get_resistive_device(self):
        return TransferUnitCell(
            [SoftBoundsResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0),
             SoftBoundsResistiveDeviceParameters(w_max_dtod=0, w_min_dtod=0)],
            params_transfer_forward=AnalogTileInputOutputParameters(is_perfect=True)
        )

    def get_tile(self, out_size, in_size, resistive_device=None, **kwargs):
        resistive_device = resistive_device or self.get_resistive_device()
        return AnalogTile(out_size, in_size, resistive_device, **kwargs).cuda()

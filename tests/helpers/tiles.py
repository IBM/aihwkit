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

from aihwkit.simulator.tiles import AnalogTile, FloatingPointTile
from aihwkit.simulator.inference_tiles import InferenceTile
from aihwkit.simulator.configs.devices import (
    IdealDevice,
    ConstantStepDevice,
    LinearStepDevice,
    ExpStepDevice,
    SoftBoundsDevice,
    IOParameters,
    DifferenceUnitCellDevice,
    VectorUnitCellDevice,
    TransferUnitCellDevice
)
from aihwkit.simulator.configs import (
    FloatingPointRPUConfig,
    InferenceRPUConfig,
    SingleRPUConfig,
    UnitCellRPUConfig,
)

from aihwkit.simulator.rpu_base import tiles


class FloatingPoint:
    """FloatingPointTile."""

    simulator_tile_class = tiles.FloatingPointTile
    first_hidden_field = None
    use_cuda = False

    def get_rpu_config(self):
        return FloatingPointRPUConfig()

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return FloatingPointTile(out_size, in_size, rpu_config, **kwargs)


class Ideal:
    """AnalogTile with IdealResistiveDevice."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = None
    use_cuda = False

    def get_rpu_config(self):
        return SingleRPUConfig(device=IdealDevice())

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class ConstantStep:
    """AnalogTile with ConstantStepResistiveDevice."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound'
    use_cuda = False

    def get_rpu_config(self):
        return SingleRPUConfig(device=ConstantStepDevice(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class LinearStep:
    """AnalogTile with LinearStepResistiveDevice."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound'
    use_cuda = False

    def get_rpu_config(self):
        return SingleRPUConfig(device=LinearStepDevice(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class ExpStep:
    """AnalogTile with ExpStepResistiveDevice."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound'
    use_cuda = False

    def get_rpu_config(self):
        return SingleRPUConfig(device=ExpStepDevice(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class Vector:
    """AnalogTile with VectorUnitCell."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound_0'
    use_cuda = False

    def get_rpu_config(self):
        return UnitCellRPUConfig(device=VectorUnitCellDevice(
            unit_cell_devices=[
                ConstantStepDevice(w_max_dtod=0, w_min_dtod=0),
                ConstantStepDevice(w_max_dtod=0, w_min_dtod=0)
            ]))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class Difference:
    """AnalogTile with DifferenceUnitCell."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound_0'
    use_cuda = False

    def get_rpu_config(self):
        return UnitCellRPUConfig(device=DifferenceUnitCellDevice(
            unit_cell_devices=[
                ConstantStepDevice(w_max_dtod=0, w_min_dtod=0)
            ]))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class Transfer:
    """AnalogTile with TransferUnitCell."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound_0'
    use_cuda = False

    def get_rpu_config(self):
        return UnitCellRPUConfig(device=TransferUnitCellDevice(
            unit_cell_devices=[
                SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0),
                SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0)
            ],
            params_transfer_forward=IOParameters(is_perfect=True)
        ))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class Inference:
    """Inference tile."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = None
    use_cuda = False

    def get_rpu_config(self):
        return InferenceRPUConfig()

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return InferenceTile(out_size, in_size, rpu_config, **kwargs)


class FloatingPointCuda:
    """FloatingPointTile."""

    simulator_tile_class = getattr(tiles, 'CudaFloatingPointTile', None)
    first_hidden_field = None
    use_cuda = True

    def get_rpu_config(self):
        return FloatingPointRPUConfig()

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return FloatingPointTile(out_size, in_size, rpu_config, **kwargs).cuda()


class IdealCuda:
    """AnalogTile with IdealResistiveDevice."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = None
    use_cuda = True

    def get_rpu_config(self):
        return SingleRPUConfig(device=IdealDevice())

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class ConstantStepCuda:
    """AnalogTile with ConstantStepResistiveDevice."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound'
    use_cuda = True

    def get_rpu_config(self):
        return SingleRPUConfig(device=ConstantStepDevice(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class LinearStepCuda:
    """AnalogTile with LinearStepResistiveDevice."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound'
    use_cuda = True

    def get_rpu_config(self):
        return SingleRPUConfig(device=LinearStepDevice(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class ExpStepCuda:
    """AnalogTile with ExpStepResistiveDevice."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound'
    use_cuda = True

    def get_rpu_config(self):
        return SingleRPUConfig(device=ExpStepDevice(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class VectorCuda:
    """AnalogTile with VectorUnitCell."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound_0'
    use_cuda = True

    def get_rpu_config(self):
        return UnitCellRPUConfig(device=VectorUnitCellDevice(
            unit_cell_devices=[
                ConstantStepDevice(w_max_dtod=0, w_min_dtod=0),
                ConstantStepDevice(w_max_dtod=0, w_min_dtod=0)
            ]))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class DifferenceCuda:
    """AnalogTile with DifferenceUnitCell."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound_0'
    use_cuda = True

    def get_rpu_config(self):
        return UnitCellRPUConfig(device=DifferenceUnitCellDevice(
            unit_cell_devices=[
                ConstantStepDevice(w_max_dtod=0, w_min_dtod=0)
            ]))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class TransferCuda:
    """AnalogTile with TransferUnitCell."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound_0'
    use_cuda = True

    def get_rpu_config(self):
        return UnitCellRPUConfig(device=TransferUnitCellDevice(
            unit_cell_devices=[
                SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0),
                SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0)
            ],
            params_transfer_forward=IOParameters(is_perfect=True)
        ))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class InferenceCuda:
    """Inference tile."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = None
    use_cuda = True

    def get_rpu_config(self):
        return InferenceRPUConfig()

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return InferenceTile(out_size, in_size, rpu_config, **kwargs).cuda()

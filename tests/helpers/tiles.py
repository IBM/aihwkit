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

"""Tile helpers for aihwkit tests."""

# pylint: disable=missing-function-docstring,too-few-public-methods

from aihwkit.simulator.tiles import AnalogTile, FloatingPointTile, InferenceTile
from aihwkit.simulator.configs.devices import (
    IdealDevice,
    ConstantStepDevice,
    LinearStepDevice,
    ExpStepDevice,
    SoftBoundsDevice,
    SoftBoundsPmaxDevice,
    PowStepDevice,
    IOParameters,
    OneSidedUnitCell,
    VectorUnitCell,
    TransferCompound,
    BufferedTransferCompound,
    ReferenceUnitCell,
    MixedPrecisionCompound,
)
from aihwkit.simulator.configs import (
    FloatingPointRPUConfig,
    InferenceRPUConfig,
    SingleRPUConfig,
    UnitCellRPUConfig,
    DigitalRankUpdateRPUConfig,
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
    """AnalogTile with IdealDevice."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = None
    use_cuda = False

    def get_rpu_config(self):
        rpu_config = SingleRPUConfig(device=IdealDevice())
        rpu_config.forward.is_perfect = True
        rpu_config.backward.is_perfect = True
        return rpu_config

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class ConstantStep:
    """AnalogTile with ConstantStepDevice."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound'
    use_cuda = False

    def get_rpu_config(self):
        return SingleRPUConfig(device=ConstantStepDevice(w_max_dtod=0, w_min_dtod=0,
                                                         up_down_dtod=0.0))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class IdealizedConstantStep:
    """AnalogTile with ConstantStepDevice."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound'
    use_cuda = False

    def get_rpu_config(self):
        rpu_config = SingleRPUConfig(device=ConstantStepDevice(
            w_max_dtod=0, w_min_dtod=0, dw_min_std=0.0, dw_min=0.0001, dw_min_dtod=0.0,
            up_down_dtod=0.0, w_max=1.0, w_min=-1.0)
        )
        rpu_config.forward.is_perfect = True
        rpu_config.backward.is_perfect = True
        return rpu_config

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class LinearStep:
    """AnalogTile with LinearStepDevice."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound'
    use_cuda = False

    def get_rpu_config(self):
        return SingleRPUConfig(device=LinearStepDevice(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class SoftBounds:
    """AnalogTile with SoftBoundsDevice."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound'
    use_cuda = False

    def get_rpu_config(self):
        return SingleRPUConfig(device=SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class SoftBoundsPmax:
    """AnalogTile with SoftBoundsPmaxDevice."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound'
    use_cuda = False

    def get_rpu_config(self):
        return SingleRPUConfig(device=SoftBoundsPmaxDevice(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class ExpStep:
    """AnalogTile with ExpStepDevice."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound'
    use_cuda = False

    def get_rpu_config(self):
        return SingleRPUConfig(device=ExpStepDevice(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class PowStep:
    """AnalogTile with PowStepDevice."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound'
    use_cuda = False

    def get_rpu_config(self):
        return SingleRPUConfig(device=PowStepDevice(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class Vector:
    """AnalogTile with VectorUnitCell."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound_0'
    use_cuda = False

    def get_rpu_config(self):
        return UnitCellRPUConfig(device=VectorUnitCell(
            unit_cell_devices=[
                ConstantStepDevice(w_max_dtod=0, w_min_dtod=0),
                ConstantStepDevice(w_max_dtod=0, w_min_dtod=0)
            ]))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class Reference:
    """AnalogTile with ReferenceUnitCell."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound_0'
    use_cuda = False

    def get_rpu_config(self):
        return UnitCellRPUConfig(device=ReferenceUnitCell(
            unit_cell_devices=[
                SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0),
                SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0)
            ]))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class OneSided:
    """AnalogTile with OneSidedUnitCell."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound_0'
    use_cuda = False

    def get_rpu_config(self):
        return UnitCellRPUConfig(device=OneSidedUnitCell(
            unit_cell_devices=[
                ConstantStepDevice(w_max_dtod=0, w_min_dtod=0)
            ]))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class Transfer:
    """AnalogTile with TransferCompound."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound_0'
    use_cuda = False

    def get_rpu_config(self):
        return UnitCellRPUConfig(device=TransferCompound(
            unit_cell_devices=[
                SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0),
                SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0)
            ],
            transfer_forward=IOParameters(is_perfect=True),
            transfer_every=1,
            gamma=0.1

        ))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class BufferedTransfer:
    """AnalogTile with BufferedTransferCompound."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound_0'
    use_cuda = False

    def get_rpu_config(self):
        return UnitCellRPUConfig(device=BufferedTransferCompound(
            unit_cell_devices=[
                SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0),
                SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0)
            ],
            transfer_forward=IOParameters(is_perfect=True),
            transfer_every=1,
            gamma=0.1

        ))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class MixedPrecision:
    """AnalogTile with MixedPrecisionCompound."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = 'max_bound'
    use_cuda = False

    def get_rpu_config(self):
        return DigitalRankUpdateRPUConfig(
            device=MixedPrecisionCompound(
                device=SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0),
                transfer_every=1),
        )

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs)


class Inference:
    """Inference tile (perfect forward)."""

    simulator_tile_class = tiles.AnalogTile
    first_hidden_field = None
    use_cuda = False

    def get_rpu_config(self):
        rpu_config = InferenceRPUConfig()
        rpu_config.forward.is_perfect = True
        return rpu_config

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
    """AnalogTile with IdealDevice."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = None
    use_cuda = True

    def get_rpu_config(self):
        rpu_config = SingleRPUConfig(device=IdealDevice())
        rpu_config.forward.is_perfect = True
        rpu_config.backward.is_perfect = True
        return rpu_config

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class ConstantStepCuda:
    """AnalogTile with ConstantStepDevice."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound'
    use_cuda = True

    def get_rpu_config(self):
        return SingleRPUConfig(device=ConstantStepDevice(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class IdealizedConstantStepCuda:
    """AnalogTile with ConstantStepDevice."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound'
    use_cuda = True

    def get_rpu_config(self):
        rpu_config = SingleRPUConfig(device=ConstantStepDevice(
            w_max_dtod=0, w_min_dtod=0, dw_min_std=0.0, dw_min=0.0001, dw_min_dtod=0.0,
            up_down_dtod=0.0, w_max=1.0, w_min=-1.0)
        )
        rpu_config.forward.is_perfect = True
        rpu_config.backward.is_perfect = True
        return rpu_config

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class LinearStepCuda:
    """AnalogTile with LinearStepDevice."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound'
    use_cuda = True

    def get_rpu_config(self):
        return SingleRPUConfig(device=LinearStepDevice(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class SoftBoundsCuda:
    """AnalogTile with SoftBoundsDevice."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound'
    use_cuda = True

    def get_rpu_config(self):
        return SingleRPUConfig(device=SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class SoftBoundsPmaxCuda:
    """AnalogTile with SoftBoundsPmaxDevice."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound'
    use_cuda = True

    def get_rpu_config(self):
        return SingleRPUConfig(device=SoftBoundsPmaxDevice(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class ExpStepCuda:
    """AnalogTile with ExpStepDevice."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound'
    use_cuda = True

    def get_rpu_config(self):
        return SingleRPUConfig(device=ExpStepDevice(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class PowStepCuda:
    """AnalogTile with PowStepDevice."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound'
    use_cuda = True

    def get_rpu_config(self):
        return SingleRPUConfig(device=PowStepDevice(w_max_dtod=0, w_min_dtod=0))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class VectorCuda:
    """AnalogTile with VectorUnitCell."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound_0'
    use_cuda = True

    def get_rpu_config(self):
        return UnitCellRPUConfig(device=VectorUnitCell(
            unit_cell_devices=[
                ConstantStepDevice(w_max_dtod=0, w_min_dtod=0),
                ConstantStepDevice(w_max_dtod=0, w_min_dtod=0)
            ]))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class ReferenceCuda:
    """AnalogTile with ReferenceUnitCell."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound_0'
    use_cuda = True

    def get_rpu_config(self):
        return UnitCellRPUConfig(device=ReferenceUnitCell(
            unit_cell_devices=[
                SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0),
                SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0)
            ]))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class OneSidedCuda:
    """AnalogTile with OneSidedUnitCell."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound_0'
    use_cuda = True

    def get_rpu_config(self):
        return UnitCellRPUConfig(device=OneSidedUnitCell(
            unit_cell_devices=[
                ConstantStepDevice(w_max_dtod=0, w_min_dtod=0)
            ]))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class TransferCuda:
    """AnalogTile with TransferCompound."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound_0'
    use_cuda = True

    def get_rpu_config(self):
        return UnitCellRPUConfig(device=TransferCompound(
            unit_cell_devices=[
                SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0),
                SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0)
            ],
            transfer_forward=IOParameters(is_perfect=True),
            transfer_every=1,
            gamma=0.1
        ))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class BufferedTransferCuda:
    """AnalogTile with BufferedTransferCompound."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound_0'
    use_cuda = True

    def get_rpu_config(self):
        return UnitCellRPUConfig(device=BufferedTransferCompound(
            unit_cell_devices=[
                SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0),
                SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0)
            ],
            transfer_forward=IOParameters(is_perfect=True),
            transfer_every=1,
            gamma=0.1
        ))

    def get_tile(self, out_size, in_size, rpu_config=None, **kwargs):
        rpu_config = rpu_config or self.get_rpu_config()
        return AnalogTile(out_size, in_size, rpu_config, **kwargs).cuda()


class MixedPrecisionCuda:
    """AnalogTile with MixedPrecisionCompound."""

    simulator_tile_class = getattr(tiles, 'CudaAnalogTile', None)
    first_hidden_field = 'max_bound'
    use_cuda = True

    def get_rpu_config(self):
        return DigitalRankUpdateRPUConfig(
            device=MixedPrecisionCompound(
                device=SoftBoundsDevice(w_max_dtod=0, w_min_dtod=0),
                transfer_every=1),
        )

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

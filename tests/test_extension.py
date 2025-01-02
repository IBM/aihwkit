# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=invalid-name, no-name-in-module, import-error

"""Tests for the AIHWKIT extension module ."""

from unittest import skipIf
from torch import Tensor, float32, randn
from aihwkit.extension import EXTENSION_COMPILED
from aihwkit.simulator.tiles.analog_mvm_irdrop_t import AnalogMVMIRDropT
from .helpers.testcases import AihwkitTestCase, SKIP_CUDA_TESTS

if EXTENSION_COMPILED:
    from aihwkit.extension.aihwkit_extension.ops import float_precision_cast


class FloatPrecisionCastTest(AihwkitTestCase):
    """Tests float precision cast."""

    @skipIf(not EXTENSION_COMPILED, "extension not compiled")
    def test_float_prec_cast(self) -> None:
        """Test float precision."""
        x = 1 + Tensor([1.0, 1.0 / 2.0, 1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0])

        y = float_precision_cast(x, 3, 4, False)
        y_ref = Tensor([2.0000, 1.5000, 1.2500, 1.1250, 1.0625, 1.0625, 1.0000])

        self.assertTensorAlmostEqual(y, y_ref)

    @skipIf(SKIP_CUDA_TESTS or not EXTENSION_COMPILED, "not compiled with CUDA support")
    def test_float_prec_cast_cuda(self) -> None:
        """Test float precision."""
        x = 1 + Tensor([1.0, 1.0 / 2.0, 1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0])

        y = float_precision_cast(x.cuda(), 3, 4, False)
        y_ref = Tensor([2.0000, 1.5000, 1.2500, 1.1250, 1.0625, 1.0625, 1.0000])

        self.assertTensorAlmostEqual(y.cpu(), y_ref)


class TheveninEquivTest(AihwkitTestCase):
    """Tests float precision cast."""

    # pylint: disable=protected-access

    @skipIf(not EXTENSION_COMPILED, "extension not compiled")
    def test_thevenin_equiv(self) -> None:
        """Test float precision."""

        weight = randn(20, 10, dtype=float32)
        x_input = randn(2, 20, dtype=float32)
        size = 128

        vth_3d, rth_3d = AnalogMVMIRDropT._thev_equiv(
            x_input, weight, use_extension=False, segments=size, phys_input_size=size
        )
        vth_3d_ext, rth_3d_ext = AnalogMVMIRDropT._thev_equiv(
            x_input, weight, use_extension=True, segments=size, phys_input_size=size
        )

        self.assertTensorAlmostEqual(vth_3d, vth_3d_ext)
        self.assertTensorAlmostEqual(rth_3d, rth_3d_ext)

    @skipIf(SKIP_CUDA_TESTS or not EXTENSION_COMPILED, "not compiled with CUDA support")
    def test_thevenin_equiv_cuda(self) -> None:
        """Test float precision."""

        weight = randn(20, 10, dtype=float32).cuda()
        x_input = randn(2, 20, dtype=float32).cuda()
        size = 128

        vth_3d, rth_3d = AnalogMVMIRDropT._thev_equiv(
            x_input, weight, use_extension=False, segments=size, phys_input_size=size
        )
        vth_3d_ext, rth_3d_ext = AnalogMVMIRDropT._thev_equiv(
            x_input, weight, use_extension=True, segments=size, phys_input_size=size
        )

        self.assertTensorAlmostEqual(vth_3d.cpu(), vth_3d_ext.cpu())
        self.assertTensorAlmostEqual(rth_3d.cpu(), rth_3d_ext.cpu())

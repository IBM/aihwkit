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

"""Tests for the high level simulator devices functionality."""

from unittest import TestCase

from aihwkit.simulator.devices import ConstantStepResistiveDevice, FloatingPointResistiveDevice
from aihwkit.simulator.parameters import (
    ConstantStepResistiveDeviceParameters,
    AnalogTileBackwardInputOutputParameters,
    AnalogTileInputOutputParameters,
    AnalogTileUpdateParameters
)

from aihwkit.simulator.rpu_base.tiles import AnalogTile, FloatingPointTile


class ResistiveDevicesTest(TestCase):
    """Test the different resistive devices."""

    def test_simple_device_array(self):
        """Test creating an array from a `FloatingPointResistiveDevice`."""
        resistive_device = FloatingPointResistiveDevice()
        analog_tile = resistive_device.create_tile(2, 4)

        self.assertIsInstance(analog_tile, FloatingPointTile)

    def test_constant_step_device_array(self):
        """Test creating an array from a `ConstantStepResistiveDevice`."""
        resistive_device = ConstantStepResistiveDevice()
        analog_tile = resistive_device.create_tile(2, 4)

        self.assertIsInstance(analog_tile, AnalogTile)

    def test_device_parameter_constructor(self):
        """Test creating a device using parameters in the constructor."""
        params_devices = ConstantStepResistiveDeviceParameters(w_max=0.987)
        params_forward = AnalogTileInputOutputParameters(inp_noise=0.321)
        params_backward = AnalogTileBackwardInputOutputParameters(inp_noise=0.456)
        params_update = AnalogTileUpdateParameters(desired_bl=78)

        # Create the device and the array.
        resistive_device = ConstantStepResistiveDevice(
            params_devices, params_forward, params_backward, params_update)
        analog_tile = resistive_device.create_tile(10, 20)

        # Assert over the parameters in the binding objects.
        parameters = analog_tile.get_parameters()
        self.assertAlmostEqual(parameters.forward_io.inp_noise, 0.321)
        self.assertAlmostEqual(parameters.backward_io.inp_noise, 0.456)
        self.assertAlmostEqual(parameters.update.desired_bl, 78)

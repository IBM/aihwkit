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

"""Some more tests for specific tiles."""

from torch import ones

from aihwkit.simulator.configs.devices import (
    TransferCompound,
    ReferenceUnitCell,
    SoftBoundsDevice
)
from aihwkit.simulator.configs.configs import UnitCellRPUConfig

from .helpers.decorators import parametrize_over_layers
from .helpers.layers import Linear, LinearCuda
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import FloatingPoint


@parametrize_over_layers(
    layers=[Linear, LinearCuda],
    tiles=[FloatingPoint],
    biases=[True, False],
    digital_biases=[True, False]
)
class TransferCompoundTest(ParametrizedTestCase):
    """Tests for transfer compound."""

    @staticmethod
    def get_transfer_compound(gamma):
        """Get a Tiki-taka compound with reference cell """
        def custom_device():
            """Custom device """
            return SoftBoundsDevice(w_max_dtod=0.0, w_min_dtod=0.0, w_max=1.0, w_min=-1.0)

        rpu_config = UnitCellRPUConfig(
            device=TransferCompound(

                # Devices that compose the Tiki-taka compound.
                unit_cell_devices=[
                    ReferenceUnitCell([custom_device(), custom_device()]),  # fast "A" matrix
                    ReferenceUnitCell([custom_device(), custom_device()])   # slow "C" matrix
                ],
                gamma=gamma,
            )
        )
        return rpu_config

    def test_hidden_parameter_setting(self):
        """Test hidden parameter set."""
        # pylint: disable=invalid-name

        for gamma in [0.0, 0.1]:
            model = self.get_layer(rpu_config=self.get_transfer_compound(gamma))

            weight, bias = model.get_weights()
            model.set_weights(weight * 0.0, bias * 0.0 if bias is not None else None)

            params = model.analog_tile.get_hidden_parameters()
            shape = params['hidden_weights_0_0'].shape

            # just dummy settings
            a, b, c, d = 0.47, 0.21, 0.64, 0.12
            params['hidden_weights_0_0'] = a*ones(*shape)  # A
            params['hidden_weights_1_0'] = b*ones(*shape)  # A ref
            params['hidden_weights_0_1'] = c*ones(*shape)  # C
            params['hidden_weights_1_1'] = d*ones(*shape)  # C_ref

            model.analog_tile.set_hidden_parameters(params)

            print(model.analog_tile.tile)
            weight, bias = model.get_weights()

            # should be
            if self.digital_bias:
                self.assertEqual(bias[0], 0.0)
            if self.bias and not self.digital_bias:
                self.assertEqual(bias[0], gamma*(a - b) + c - d)

            self.assertEqual(weight[0][0], gamma*(a - b) + c - d)

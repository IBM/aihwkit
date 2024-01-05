# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Some more tests for specific tiles."""

from torch import ones, Tensor
from torch.nn.functional import mse_loss

from aihwkit.simulator.configs.devices import SoftBoundsDevice
from aihwkit.simulator.configs.compounds import TransferCompound, ReferenceUnitCell

from aihwkit.simulator.configs.configs import UnitCellRPUConfig
from aihwkit.optim import AnalogSGD

from .helpers.decorators import parametrize_over_layers
from .helpers.layers import Linear, LinearCuda
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import FloatingPoint


@parametrize_over_layers(
    layers=[Linear, LinearCuda], tiles=[FloatingPoint], biases=["analog", "digital", None]
)
class TransferCompoundTest(ParametrizedTestCase):
    """Tests for transfer compound."""

    @staticmethod
    def get_transfer_compound(gamma, **kwargs):
        """Get a Tiki-taka compound with reference cell"""

        def custom_device(**kwargs):
            """Custom device"""
            return SoftBoundsDevice(w_max_dtod=0.0, w_min_dtod=0.0, w_max=1.0, w_min=-1.0, **kwargs)

        rpu_config = UnitCellRPUConfig(
            device=TransferCompound(
                # Devices that compose the Tiki-taka compound.
                unit_cell_devices=[
                    # fast "A" matrix
                    ReferenceUnitCell([custom_device(**kwargs), custom_device(**kwargs)]),
                    # slow "C" matrix
                    ReferenceUnitCell([custom_device(**kwargs), custom_device(**kwargs)]),
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

            analog_tile = next(model.analog_tiles())
            params = analog_tile.get_hidden_parameters()
            shape = params["hidden_weights_0_0"].shape

            # just dummy settings
            a, b, c, d = 0.47, 0.21, 0.64, 0.12
            params["hidden_weights_0_0"] = a * ones(*shape)  # A
            params["hidden_weights_1_0"] = b * ones(*shape)  # A ref
            params["hidden_weights_0_1"] = c * ones(*shape)  # C
            params["hidden_weights_1_1"] = d * ones(*shape)  # C_ref

            analog_tile.set_hidden_parameters(params)

            weight, bias = model.get_weights()

            # should be
            if self.digital_bias:
                self.assertEqual(bias[0], 0.0)
            if self.bias and not self.digital_bias:
                self.assertEqual(bias[0], gamma * (a - b) + c - d)

            self.assertEqual(weight[0][0], gamma * (a - b) + c - d)

    def test_decay(self):
        """Test hidden parameter set."""
        # pylint: disable=invalid-name, too-many-locals

        lifetime = 100.0  # initial setting (needs to be larger 1)
        gamma = 0.1
        reset_bias = 0.3  # decay shift
        rpu_config = self.get_transfer_compound(
            gamma=gamma, lifetime=lifetime, lifetime_dtod=0.0, reset=reset_bias, reset_std=0.0
        )
        model = self.get_layer(in_features=2, out_features=1, rpu_config=rpu_config)

        weight, bias = model.get_weights()
        model.set_weights(weight * 0.0, bias * 0.0 if bias is not None else None)

        analog_tile = next(model.analog_tiles())
        params = analog_tile.get_hidden_parameters()
        shape = params["hidden_weights_0_0"].shape

        # just dummy settings
        a, b, c, d = 0.47, 0.21, 0.64, 0.12
        params["hidden_weights_0_0"] = a * ones(*shape)  # A
        params["hidden_weights_1_0"] = b * ones(*shape)  # A ref
        params["hidden_weights_0_1"] = c * ones(*shape)  # C
        params["hidden_weights_1_1"] = d * ones(*shape)  # C_ref

        # explicitly set the decay scales (which is 1-1/lifetime)
        a_dcy, b_dcy, c_dcy, d_dcy = 0.95, 0.28, 0.33, 0.12
        params["decay_scales_0_0"] = a_dcy * ones(*shape)  # A
        params["decay_scales_1_0"] = b_dcy * ones(*shape)  # A ref
        params["decay_scales_0_1"] = c_dcy * ones(*shape)  # C
        params["decay_scales_1_1"] = d_dcy * ones(*shape)  # C_ref

        analog_tile.set_hidden_parameters(params)
        weight, bias = model.get_weights()
        x_b = Tensor([[0.1, 0.2], [0.2, 0.4]])
        y_b = Tensor([[0.3], [0.6]])

        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()

        # LR set to zero. Only lifetime will be applied
        opt = AnalogSGD(model.parameters(), lr=0.0)

        epochs = 2
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(x_b)
            loss = mse_loss(pred, y_b)

            loss.backward()
            opt.step()

        weight, bias = model.get_weights()

        # reference values
        a = (a - reset_bias) * pow(a_dcy, epochs) + reset_bias
        b = (b - reset_bias) * pow(b_dcy, epochs) + reset_bias
        c = (c - reset_bias) * pow(c_dcy, epochs) + reset_bias
        d = (d - reset_bias) * pow(d_dcy, epochs) + reset_bias

        if self.digital_bias:
            self.assertAlmostEqual(bias[0].item(), 0.0)
        if self.bias and not self.digital_bias:
            self.assertAlmostEqual(bias[0].item(), gamma * (a - b) + c - d, 5)

        self.assertAlmostEqual(weight[0][0].item(), gamma * (a - b) + c - d, 5)

# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Some more tests for specific tiles."""

from torch import ones, Tensor
from torch.nn.functional import mse_loss

from aihwkit.simulator.configs.devices import SoftBoundsDevice
from aihwkit.simulator.configs.compounds import (
    TransferCompound,
    ChoppedTransferCompound,
    ReferenceUnitCell,
)

from aihwkit.simulator.configs.configs import UnitCellRPUConfig
from aihwkit.simulator.parameters import IOParameters
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

    def test_decay_with_negative_reset_bias(self):
        """Test that decay keeps a negative reset bias."""
        # pylint: disable=invalid-name, too-many-locals

        lifetime = 100.0
        gamma = 0.1
        reset_bias = -0.3
        rpu_config = self.get_transfer_compound(
            gamma=gamma, lifetime=lifetime, lifetime_dtod=0.0, reset=reset_bias, reset_std=0.0
        )
        model = self.get_layer(in_features=2, out_features=1, rpu_config=rpu_config)

        weight, bias = model.get_weights()
        model.set_weights(weight * 0.0, bias * 0.0 if bias is not None else None)

        analog_tile = next(model.analog_tiles())
        params = analog_tile.get_hidden_parameters()
        shape = params["hidden_weights_0_0"].shape

        a, b, c, d = 0.47, 0.21, 0.64, 0.12
        params["hidden_weights_0_0"] = a * ones(*shape)
        params["hidden_weights_1_0"] = b * ones(*shape)
        params["hidden_weights_0_1"] = c * ones(*shape)
        params["hidden_weights_1_1"] = d * ones(*shape)

        a_dcy, b_dcy, c_dcy, d_dcy = 0.95, 0.28, 0.33, 0.12
        params["decay_scales_0_0"] = a_dcy * ones(*shape)
        params["decay_scales_1_0"] = b_dcy * ones(*shape)
        params["decay_scales_0_1"] = c_dcy * ones(*shape)
        params["decay_scales_1_1"] = d_dcy * ones(*shape)

        analog_tile.set_hidden_parameters(params)
        x_b = Tensor([[0.1, 0.2], [0.2, 0.4]])
        y_b = Tensor([[0.3], [0.6]])

        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()

        opt = AnalogSGD(model.parameters(), lr=0.0)

        epochs = 2
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(x_b)
            loss = mse_loss(pred, y_b)

            loss.backward()
            opt.step()

        weight, bias = model.get_weights()

        a = (a - reset_bias) * pow(a_dcy, epochs) + reset_bias
        b = (b - reset_bias) * pow(b_dcy, epochs) + reset_bias
        c = (c - reset_bias) * pow(c_dcy, epochs) + reset_bias
        d = (d - reset_bias) * pow(d_dcy, epochs) + reset_bias

        if self.digital_bias:
            self.assertAlmostEqual(bias[0].item(), 0.0)
        if self.bias and not self.digital_bias:
            self.assertAlmostEqual(bias[0].item(), gamma * (a - b) + c - d, 5)

        self.assertAlmostEqual(weight[0][0].item(), gamma * (a - b) + c - d, 5)


@parametrize_over_layers(
    layers=[Linear, LinearCuda], tiles=[FloatingPoint], biases=["analog", "digital", None]
)
class ChoppedTransferCompoundTest(ParametrizedTestCase):
    """Tests for non-zero ``gamma`` in the chopped transfer compound (PR #764).

    The fast array ``A`` is updated with per-element chopper sign flips and is
    therefore stored in *chopped* form
    ``A_stored[i, j] = c_d[i] * c_x[j] * A_true[i, j]`` where the de-chopping
    sign is ``+1`` if the input and output choppers agree and ``-1`` if they
    differ (see ``ChoppedTransferRPUDevice::readAndUpdate``).  With
    ``gamma != 0`` the fast array contributes directly to the visible weight
    (residual learning), so the weight reduction must apply the de-chopping:

        ``W = gamma * (c_d (x) c_x) * A_stored + C``

    These tests pin down that reduction, including when the chopper state
    changes.
    """

    @staticmethod
    def get_chopped_transfer_compound(gamma, fast_lr=1.0, transfer_every=1.0, **device_kwargs):
        """Get a chopped-transfer compound (fast ``A`` + slow ``C``)."""

        def custom_device(**kwargs):
            """Bound-free soft-bounds device."""
            return SoftBoundsDevice(w_max_dtod=0.0, w_min_dtod=0.0, w_max=1.0, w_min=-1.0, **kwargs)

        rpu_config = UnitCellRPUConfig(
            device=ChoppedTransferCompound(
                unit_cell_devices=[
                    custom_device(**device_kwargs),  # fast "A" matrix
                    custom_device(**device_kwargs),  # slow "C" matrix
                ],
                gamma=gamma,
                fast_lr=fast_lr,
                transfer_every=transfer_every,
                # Keep the choppers static so the tests can set them explicitly.
                in_chop_prob=0.0,
                out_chop_prob=0.0,
                transfer_forward=IOParameters(is_perfect=True),
            )
        )
        return rpu_config

    @staticmethod
    def chopper_sign(in_chop, out_chop):
        """De-chopping sign: ``+1`` if choppers agree, ``-1`` if they differ."""
        return -1.0 if in_chop != out_chop else 1.0

    @staticmethod
    def set_choppers(analog_tile, in_chop, out_chop):
        """Force the (static) input/output choppers to a known uniform state.

        Choppers live in the device ``extra`` state rather than in the hidden
        parameters, so they are round-tripped through ``dump_extra`` /
        ``load_extra`` (the same mechanism as ``test_dump_extra``).
        """
        state = analog_tile.tile.dump_extra()
        for key in state:
            if key.endswith("rpu_device.in_chopper"):
                state[key] = [float(in_chop)] * len(state[key])
            elif key.endswith("rpu_device.out_chopper"):
                state[key] = [float(out_chop)] * len(state[key])
        analog_tile.tile.load_extra(state, True)

    def test_chopper_correction(self):
        """Visible weight de-chops the fast array: ``W = gamma*sign*A + C``."""
        # pylint: disable=invalid-name
        gamma = 0.1
        a, c = 0.4, 0.2
        model = self.get_layer(
            in_features=2, out_features=1, rpu_config=self.get_chopped_transfer_compound(gamma)
        )

        weight, bias = model.get_weights()
        model.set_weights(weight * 0.0, bias * 0.0 if bias is not None else None)

        analog_tile = next(model.analog_tiles())
        params = analog_tile.get_hidden_parameters()
        shape = params["hidden_weights_0"].shape
        params["hidden_weights_0"] = a * ones(*shape)  # fast A
        params["hidden_weights_1"] = c * ones(*shape)  # slow C

        # Sweep every chopper combination, including the default identity state
        # ``(0, 0)`` which must already yield the full ``gamma*A + C``.
        for in_chop, out_chop in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]:
            self.set_choppers(analog_tile, in_chop, out_chop)
            # Re-apply the hidden weights to trigger a ``reduceToWeights`` with
            # the current chopper state.
            analog_tile.set_hidden_parameters(params)

            weight, bias = model.get_weights()
            expected = gamma * self.chopper_sign(in_chop, out_chop) * a + c
            msg = f"in_chop={in_chop}, out_chop={out_chop}"

            if self.digital_bias:
                self.assertAlmostEqual(bias[0].item(), 0.0)
            if self.bias and not self.digital_bias:
                self.assertAlmostEqual(bias[0].item(), expected, 5, msg=msg)
            self.assertAlmostEqual(weight[0][0].item(), expected, 5, msg=msg)

    def test_decay(self):
        """Decay applied to A and C is reflected in ``W = gamma*A + C``."""
        # pylint: disable=invalid-name, too-many-locals

        lifetime = 100.0  # initial setting (needs to be larger 1)
        gamma = 0.1
        reset_bias = 0.3  # decay shift
        rpu_config = self.get_chopped_transfer_compound(
            gamma=gamma,
            fast_lr=0.0,  # no fast-array update (LR is zero anyway)
            transfer_every=1.0e6,  # no transfer during the test, only decay
            lifetime=lifetime,
            lifetime_dtod=0.0,
            reset=reset_bias,
            reset_std=0.0,
        )
        model = self.get_layer(in_features=2, out_features=1, rpu_config=rpu_config)

        weight, bias = model.get_weights()
        model.set_weights(weight * 0.0, bias * 0.0 if bias is not None else None)

        analog_tile = next(model.analog_tiles())
        params = analog_tile.get_hidden_parameters()
        shape = params["hidden_weights_0"].shape

        # just dummy settings
        a, c = 0.47, 0.64
        params["hidden_weights_0"] = a * ones(*shape)  # A
        params["hidden_weights_1"] = c * ones(*shape)  # C

        # explicitly set the decay scales (which is 1-1/lifetime)
        a_dcy, c_dcy = 0.95, 0.33
        params["decay_scales_0"] = a_dcy * ones(*shape)  # A
        params["decay_scales_1"] = c_dcy * ones(*shape)  # C

        analog_tile.set_hidden_parameters(params)

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

        # reference values (choppers are identity -> de-chopping sign is +1)
        a = (a - reset_bias) * pow(a_dcy, epochs) + reset_bias
        c = (c - reset_bias) * pow(c_dcy, epochs) + reset_bias

        if self.digital_bias:
            self.assertAlmostEqual(bias[0].item(), 0.0)
        if self.bias and not self.digital_bias:
            self.assertAlmostEqual(bias[0].item(), gamma * a + c, 5)

        self.assertAlmostEqual(weight[0][0].item(), gamma * a + c, 5)

    def test_decay_with_negative_reset_bias(self):
        """Negative reset bias survives decay; a chopper flip then flips A."""
        # pylint: disable=invalid-name, too-many-locals

        lifetime = 100.0
        gamma = 0.1
        reset_bias = -0.3
        rpu_config = self.get_chopped_transfer_compound(
            gamma=gamma,
            fast_lr=0.0,
            transfer_every=1.0e6,
            lifetime=lifetime,
            lifetime_dtod=0.0,
            reset=reset_bias,
            reset_std=0.0,
        )
        model = self.get_layer(in_features=2, out_features=1, rpu_config=rpu_config)

        weight, bias = model.get_weights()
        model.set_weights(weight * 0.0, bias * 0.0 if bias is not None else None)

        analog_tile = next(model.analog_tiles())
        params = analog_tile.get_hidden_parameters()
        shape = params["hidden_weights_0"].shape

        a, c = 0.47, 0.64
        params["hidden_weights_0"] = a * ones(*shape)
        params["hidden_weights_1"] = c * ones(*shape)

        a_dcy, c_dcy = 0.95, 0.33
        params["decay_scales_0"] = a_dcy * ones(*shape)
        params["decay_scales_1"] = c_dcy * ones(*shape)

        analog_tile.set_hidden_parameters(params)

        x_b = Tensor([[0.1, 0.2], [0.2, 0.4]])
        y_b = Tensor([[0.3], [0.6]])
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()

        opt = AnalogSGD(model.parameters(), lr=0.0)

        epochs = 2
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(x_b)
            loss = mse_loss(pred, y_b)

            loss.backward()
            opt.step()

        weight, bias = model.get_weights()

        a = (a - reset_bias) * pow(a_dcy, epochs) + reset_bias
        c = (c - reset_bias) * pow(c_dcy, epochs) + reset_bias

        if self.digital_bias:
            self.assertAlmostEqual(bias[0].item(), 0.0)
        if self.bias and not self.digital_bias:
            self.assertAlmostEqual(bias[0].item(), gamma * a + c, 5)

        self.assertAlmostEqual(weight[0][0].item(), gamma * a + c, 5)

        # Dynamically change the chopper: make the input and output choppers
        # differ so the de-chopping sign flips to -1. The decayed A and C are
        # unchanged, so only the sign of the gamma*A term flips.
        params = analog_tile.get_hidden_parameters()  # current (decayed) A and C
        self.set_choppers(analog_tile, 1.0, 0.0)
        analog_tile.set_hidden_parameters(params)  # force reduce with new choppers

        weight, bias = model.get_weights()
        if self.bias and not self.digital_bias:
            self.assertAlmostEqual(bias[0].item(), gamma * (-1.0) * a + c, 5)

        self.assertAlmostEqual(weight[0][0].item(), gamma * (-1.0) * a + c, 5)

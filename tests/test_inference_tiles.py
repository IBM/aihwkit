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

"""Tests for inference tiles."""

from torch import ones
from torch import Tensor
from torch.nn.functional import mse_loss

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.utils import (
    WeightNoiseType,
    WeightClipType,
    WeightModifierType,
    WeightModifierParameter
)
from aihwkit.simulator.noise_models import PCMLikeNoiseModel

from .helpers.decorators import parametrize_over_tiles
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import Inference, InferenceCuda


@parametrize_over_tiles([
    Inference,
    InferenceCuda
])
class InferenceTileTest(ParametrizedTestCase):
    """Inference model tests."""

    def get_model_and_x(self):
        """Trains a simple model."""
        # Prepare the datasets (input and expected output).
        x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
        y = Tensor([[1.0, 0.5], [0.7, 0.3]])

        # Define a single-layer network, using a constant step device type.
        rpu_config = self.get_rpu_config()
        rpu_config.forward.out_res = -1.  # Turn off (output) ADC discretization.
        rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
        rpu_config.forward.w_noise = 0.02
        rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)

        model = AnalogLinear(4, 2, bias=True, rpu_config=rpu_config)

        # Move the model and tensors to cuda if it is available.
        if self.use_cuda:
            x = x.cuda()
            y = y.cuda()
            model.cuda()

        # Define an analog-aware optimizer, preparing it for using the layers.
        opt = AnalogSGD(model.parameters(), lr=0.1)
        opt.regroup_param_groups(model)

        for _ in range(100):
            # Add the training Tensor to the model (input).
            pred = model(x)
            # Add the expected output Tensor.
            loss = mse_loss(pred, y)
            # Run training (backward propagation).
            loss.backward()

            opt.step()

        return model, x

    def test_drift(self):
        """Test using realistic weights (bias)."""
        model, x = self.get_model_and_x()

        # do inference with drift
        pred_before = model(x)

        pred_last = pred_before
        model.eval()
        for t_inference in [0., 1., 20., 1000., 1e5]:
            model.drift_analog_weights(t_inference)
            pred_drift = model(x)
            self.assertNotAlmostEqualTensor(pred_last, pred_drift)
            pred_last = pred_drift

        self.assertNotAlmostEqualTensor(model.analog_tile.alpha, ones((1,)))

    def test_post_update_step_clip(self):
        """Tests whether post update diffusion is performed"""
        rpu_config = self.get_rpu_config()
        rpu_config.clip.type = WeightClipType.FIXED_VALUE
        rpu_config.clip.fixed_value = 0.3
        analog_tile = self.get_tile(2, 3, rpu_config=rpu_config, bias=True)

        weights = Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        biases = Tensor([-0.1, -0.6])

        analog_tile.set_learning_rate(0.123)
        analog_tile.set_weights(weights, biases)

        analog_tile.post_update_step()

        tile_weights, tile_biases = analog_tile.get_weights()

        self.assertNotAlmostEqualTensor(tile_weights, weights)
        self.assertNotAlmostEqualTensor(tile_biases, biases)

    def do_post_forward_modifier_test(self, modifier):
        """Tests whether post update diffusion is performed"""
        rpu_config = self.get_rpu_config()
        rpu_config.drift_compensation = None
        rpu_config.forward.w_noise = 0.0
        rpu_config.forward.out_noise = 0.0
        rpu_config.forward.inp_noise = 0.0

        if modifier is not None:
            rpu_config.modifier = modifier

        analog_tile = self.get_tile(2, 3, rpu_config=rpu_config, bias=True)
        x_input = Tensor([[0.1, 0.2, 0.3]])
        weights = Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        biases = Tensor([-0.1, -0.2])

        if self.use_cuda:
            x_input = x_input.cuda()

        analog_tile.set_learning_rate(0.123)
        analog_tile.set_weights(weights, biases)

        x_output = analog_tile.forward(x_input, is_test=True)
        x_output_post = analog_tile.forward(x_input, is_test=False)
        x_output_post_true = analog_tile.forward(x_input, is_test=True)
        tile_weights, tile_biases = analog_tile.get_weights()

        self.assertTensorAlmostEqual(tile_weights, weights)
        self.assertTensorAlmostEqual(tile_biases, biases)

        if modifier is None:
            self.assertTensorAlmostEqual(x_output, x_output_post)
        else:
            self.assertNotAlmostEqualTensor(x_output, x_output_post)

        self.assertTensorAlmostEqual(x_output, x_output_post_true)

    def get_modifier(self, mod_type):
        """Returns the modifier parameter """
        modifier = WeightModifierParameter()
        modifier.std_dev = 1.0
        modifier.enable_during_test = False
        modifier.res = 0.132
        modifier.type = mod_type
        modifier.coeff0 = 1.0
        modifier.coeff1 = 0.1
        modifier.coeff2 = 0.2
        modifier.rel_to_actual_wmax = False
        modifier.assumed_wmax = 1.0
        return modifier

    def test_post_forward_modifier_types(self):
        """Tests whether post update diffusion is performed"""

        self.do_post_forward_modifier_test(None)

        modifier = self.get_modifier(WeightModifierType.POLY)
        self.do_post_forward_modifier_test(modifier)

        modifier = self.get_modifier(WeightModifierType.DOREFA)
        self.do_post_forward_modifier_test(modifier)

        modifier = self.get_modifier(WeightModifierType.MULT_NORMAL)
        self.do_post_forward_modifier_test(modifier)

        modifier = self.get_modifier(WeightModifierType.DISCRETIZE)
        self.do_post_forward_modifier_test(modifier)

        modifier = self.get_modifier(WeightModifierType.DISCRETIZE_ADD_NORMAL)
        self.do_post_forward_modifier_test(modifier)

        modifier = self.get_modifier(WeightModifierType.COPY)
        modifier.pdrop = 0.9999
        self.do_post_forward_modifier_test(modifier)

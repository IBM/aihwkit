# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for inference tiles."""

from typing import Optional, List

from parameterized import parameterized
from torch import ones
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.optim import SGD
from torch.nn import Linear

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.configs.utils import (
    WeightNoiseType,
    WeightClipType,
    WeightModifierType,
    WeightModifierParameter,
    WeightRemapType,
)
from aihwkit.inference import PCMLikeNoiseModel

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
            opt.zero_grad()

            # Add the training Tensor to the model (input).
            pred = model(x)
            # Add the expected output Tensor.
            loss = mse_loss(pred, y)
            # Run training (backward propagation).
            loss.backward()

            opt.step()

        return model, x

    def test_against_fp(self):
        """Test whether FP is same as is_perfect inference tile."""
        # pylint: disable-msg=too-many-locals
        # Prepare the datasets (input and expected output).
        x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
        y = Tensor([[1.0, 0.5], [0.7, 0.3]])

        # Define a single-layer network, using a constant step device type.
        rpu_config = self.get_rpu_config()
        rpu_config.forward.is_perfect = True
        model_torch = Linear(4, 2, bias=True)
        model = AnalogLinear(4, 2, bias=True, rpu_config=rpu_config)
        model.set_weights(model_torch.weight, model_torch.bias)
        model_fp = AnalogLinear(4, 2, bias=True, rpu_config=FloatingPointRPUConfig())
        model_fp.set_weights(model_torch.weight, model_torch.bias)

        self.assertTensorAlmostEqual(model.get_weights()[0], model_torch.weight)
        self.assertTensorAlmostEqual(model.get_weights()[0], model_fp.get_weights()[0])

        # Move the model and tensors to cuda if it is available.
        if self.use_cuda:
            x = x.cuda()
            y = y.cuda()
            model.cuda()
            model_fp.cuda()
            model_torch.cuda()

        # Define an analog-aware optimizer, preparing it for using the layers.
        opt = AnalogSGD(model.parameters(), lr=0.1)
        opt_fp = AnalogSGD(model_fp.parameters(), lr=0.1)
        opt_torch = SGD(model_torch.parameters(), lr=0.1)

        for _ in range(100):

            # inference
            opt.zero_grad()
            pred = model(x)
            loss = mse_loss(pred, y)
            loss.backward()
            opt.step()

            # same for fp
            opt_fp.zero_grad()
            pred_fp = model_fp(x)
            loss_fp = mse_loss(pred_fp, y)
            loss_fp.backward()
            opt_fp.step()

            # same for torch
            opt_torch.zero_grad()
            pred_torch = model_torch(x)
            loss_torch = mse_loss(pred_torch, y)
            loss_torch.backward()
            opt_torch.step()

            self.assertTensorAlmostEqual(pred_torch, pred)
            self.assertTensorAlmostEqual(loss_torch, loss)
            self.assertTensorAlmostEqual(model.get_weights()[0], model_torch.weight)

            self.assertTensorAlmostEqual(pred_fp, pred)
            self.assertTensorAlmostEqual(loss_fp, loss)
            self.assertTensorAlmostEqual(model.get_weights()[0], model_fp.get_weights()[0])

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
        """Tests whether post update diffusion is performed."""
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

    def test_post_update_step_remap_layer(self):
        """Tests whether post update remap is performed."""
        rpu_config = self.get_rpu_config()
        rpu_config.mapping.out_scaling_columnwise = False
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_omega = 1.0
        rpu_config.mapping.weight_scaling_columnwise = False

        rpu_config.remap.type = WeightRemapType.LAYERWISE_SYMMETRIC
        analog_tile = self.get_tile(2, 3, rpu_config=rpu_config, bias=True)

        weights = Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        biases = Tensor([-0.1, -0.3])

        analog_tile.set_learning_rate(0.123)
        analog_tile.set_weights(weights, biases)

        analog_tile.post_update_step()

        tile_weights, tile_biases = analog_tile.get_weights()

        self.assertTensorAlmostEqual(tile_weights, weights / 0.6)
        self.assertTensorAlmostEqual(tile_biases, biases / 0.6)

        scales = analog_tile.get_mapping_scales()
        self.assertTensorAlmostEqual(scales, Tensor([0.6, 0.6]))

    def test_post_update_step_remap_column(self):
        """Tests whether post update remap is performed."""
        rpu_config = self.get_rpu_config()
        rpu_config.mapping.out_scaling_columnwise = True
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_omega = 1.0
        rpu_config.mapping.weight_scaling_columnwise = True

        rpu_config.remap.type = WeightRemapType.CHANNELWISE_SYMMETRIC
        analog_tile = self.get_tile(2, 3, rpu_config=rpu_config, bias=True)

        weights = Tensor([[-0.7, 0.2, 0.3], [0.4, 0.5, 0.6]])
        biases = Tensor([-0.1, -0.3])

        analog_tile.set_learning_rate(0.123)
        analog_tile.set_weights(weights, biases)

        analog_tile.post_update_step()

        tile_weights, tile_biases = analog_tile.get_weights()

        expected_scales = Tensor([0.7, 0.6])
        self.assertTensorAlmostEqual(tile_weights, weights / expected_scales.view(-1, 1))
        self.assertTensorAlmostEqual(tile_biases, biases / expected_scales)

        scales = analog_tile.get_mapping_scales()
        self.assertTensorAlmostEqual(scales, expected_scales)

    @parameterized.expand([
        ('none', None,),
        ('dorefa', WeightModifierType.DOREFA,),
        ('mult_normal', WeightModifierType.MULT_NORMAL,),
        ('poly', WeightModifierType.POLY, [1., 3.]),
        ('polyN', WeightModifierType.POLY, [0.1, 0.2, 0.2, 0.3]),
        ('discretize', WeightModifierType.DISCRETIZE,),
        ('add_normal', WeightModifierType.DISCRETIZE_ADD_NORMAL,),
    ])
    def test_post_forward_modifier_types(self, _,
                                         modifier_type: 'WeightModifierType',
                                         coeffs: Optional[List] = None):
        """Tests whether modifier is performed."""
        rpu_config = self.get_rpu_config()
        rpu_config.drift_compensation = None
        rpu_config.forward.is_perfect = True
        rpu_config.forward.out_noise = 0.0
        rpu_config.forward.inp_noise = 0.0

        modifier = self.get_modifier(modifier_type, coeffs)
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
        analog_tile.post_update_step()
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

    @staticmethod
    def get_modifier(
            modifier_type: Optional[WeightModifierType],
            coeffs: Optional[List] = None,
    ) -> Optional[WeightModifierParameter]:
        """Returns the modifier parameter."""
        if modifier_type is None:
            return None
        if coeffs is None:
            coeffs = [1.0, 0.1]

        modifier = WeightModifierParameter(
            type=modifier_type,
            std_dev=1.0,
            enable_during_test=False,
            res=0.132,
            coeffs=coeffs,
            rel_to_actual_wmax=False,
            assumed_wmax=1.0
        )

        if modifier_type == WeightModifierType.COPY:
            modifier.pdrop = 0.5

        return modifier

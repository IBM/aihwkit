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

"""Tests for linear layer."""

from numpy.testing import assert_array_almost_equal

from torch import Tensor, manual_seed
from torch.nn import Sequential, Linear as torchLinear
from torch.nn.functional import mse_loss
from torch.optim import SGD

from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.configs import InferenceRPUConfig, FloatingPointRPUConfig
from aihwkit.simulator.configs.utils import MappingParameter
from aihwkit.nn import AnalogSequential, AnalogLinear

from .helpers.decorators import parametrize_over_layers
from .helpers.layers import Linear, LinearCuda
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import FloatingPoint, IdealizedConstantStep, Inference


@parametrize_over_layers(
    layers=[Linear, LinearCuda],
    tiles=[FloatingPoint, IdealizedConstantStep, Inference],
    biases=['analog', 'digital', None]
)
class LinearLayerTest(ParametrizedTestCase):
    """Linear layer abstractions tests."""

    @staticmethod
    def train_model(model, loss_func, x_b, y_b):
        """Train the model."""
        opt = AnalogSGD(model.parameters(), lr=0.5)
        opt.regroup_param_groups(model)

        epochs = 100
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(x_b)
            loss = loss_func(pred, y_b)

            loss.backward()
            opt.step()

    @staticmethod
    def train_model_torch(model, loss_func, x_b, y_b):
        """Train the model with torch SGD."""
        opt = SGD(model.parameters(), lr=0.5)
        epochs = 100
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(x_b)
            loss = loss_func(pred, y_b)

            loss.backward()
            opt.step()

    def test_single_analog_layer(self):
        """Check using a single layer."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2], [0.2, 0.4]])
        y_b = Tensor([[0.3], [0.6]])

        manual_seed(4321)
        model = self.get_layer(2, 1)
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            model = model.cuda()

        initial_loss = loss_func(model(x_b), y_b)
        self.train_model(model, loss_func, x_b, y_b)
        self.assertLess(loss_func(model(x_b), y_b), initial_loss)

    def test_single_analog_layer_sequential(self):
        """Check using a single layer as a Sequential."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2], [0.2, 0.4]])
        y_b = Tensor([[0.3], [0.6]])

        manual_seed(4321)
        model = Sequential(self.get_layer(2, 1))
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            model = model.cuda()

        initial_loss = loss_func(model(x_b), y_b)
        self.train_model(model, loss_func, x_b, y_b)
        self.assertLess(loss_func(model(x_b), y_b), initial_loss)

    def test_seed(self):
        """Check layer seed."""

        manual_seed(4321)
        layer1 = self.get_layer(4, 2)

        manual_seed(4321)
        layer2 = self.get_layer(4, 2)

        weight1, bias1 = layer1.get_weights()
        weight2, bias2 = layer2.get_weights()

        assert_array_almost_equal(weight1, weight2)
        if bias1 is not None:
            assert_array_almost_equal(bias1, bias2)

    def test_several_analog_layers(self):
        """Check using a several analog layers."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.3, 0.1]])
        y_b = Tensor([[0.3], [0.6]])

        manual_seed(4321)
        model = Sequential(
            self.get_layer(4, 2),
            self.get_layer(2, 1)
        )
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            model = model.cuda()

        initial_loss = loss_func(model(x_b), y_b)
        self.train_model(model, loss_func, x_b, y_b)
        self.assertLess(loss_func(model(x_b), y_b), initial_loss)

    def test_analog_and_digital(self):
        """Check mixing analog and digital layers."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.3, 0.1]])
        y_b = Tensor([[0.3], [0.6]])

        manual_seed(4321)
        model = Sequential(
            self.get_layer(4, 3),
            torchLinear(3, 3),
            self.get_layer(3, 1)
        )
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            model = model.cuda()

        initial_loss = loss_func(model(x_b), y_b)
        self.train_model(model, loss_func, x_b, y_b)
        self.assertLess(loss_func(model(x_b), y_b), initial_loss)

    def test_analog_torch_optimizer(self):
        """Check analog layers with torch SGD for inference."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.3, 0.1]])
        y_b = Tensor([[0.3], [0.6]])

        manual_seed(4321)
        model = Sequential(
            self.get_layer(4, 3),
            self.get_layer(3, 1),
        )
        if not isinstance(model[0].analog_tile.rpu_config, InferenceRPUConfig):
            return

        manual_seed(4321)
        rpu_config = FloatingPointRPUConfig(
            mapping=MappingParameter(digital_bias=self.digital_bias)
        )
        model2 = AnalogSequential(
            AnalogLinear(4, 3, rpu_config=rpu_config, bias=self.bias),
            AnalogLinear(3, 1, rpu_config=rpu_config, bias=self.bias)
        )

        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            model = model.cuda()
            model2 = model2.cuda()

        initial_loss = loss_func(model(x_b), y_b)

        # train with SGD
        self.train_model_torch(model, loss_func, x_b, y_b)
        self.assertLess(loss_func(model(x_b), y_b), initial_loss)

        # train with AnalogSGD
        self.train_model(model2, loss_func, x_b, y_b)
        self.assertLess(loss_func(model2(x_b), y_b), initial_loss)
        final_loss = loss_func(model(x_b), y_b).detach().cpu().numpy()
        final_loss2 = loss_func(model2(x_b), y_b).detach().cpu().numpy()

        assert_array_almost_equal(final_loss, final_loss2)

    def test_learning_rate_update(self):
        """Check the learning rate update is applied to tile."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2], [0.2, 0.4]])
        y_b = Tensor([[0.3], [0.6]])

        layer1 = self.get_layer(2, 3)
        layer2 = self.get_layer(3, 1)

        model = Sequential(layer1, layer2)
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            model = model.cuda()
        opt = AnalogSGD(model.parameters(), lr=0.5)
        opt.regroup_param_groups(model)
        opt.zero_grad()

        new_lr = 0.07
        for param_group in opt.param_groups:
            param_group['lr'] = new_lr

        pred = model(x_b)
        loss = loss_func(pred, y_b)
        loss.backward()
        opt.step()

        if not layer1.analog_tile.get_analog_ctx().use_torch_update:
            self.assertAlmostEqual(layer1.analog_tile.get_learning_rate(), new_lr)

    def test_learning_rate_update_fn(self):
        """Check the learning rate update is applied to tile."""
        layer1 = self.get_layer(2, 3)
        layer2 = self.get_layer(3, 1)

        model = Sequential(layer1, layer2)
        if self.use_cuda:
            model = model.cuda()
        opt = AnalogSGD(model.parameters(), lr=0.5)
        opt.regroup_param_groups(model)
        opt.zero_grad()

        new_lr = 0.07

        opt.set_learning_rate(new_lr)

        self.assertAlmostEqual(layer1.analog_tile.get_learning_rate(), new_lr)
        self.assertAlmostEqual(layer2.analog_tile.get_learning_rate(), new_lr)

    def test_out_scaling_alpha_learning(self):
        """Check if out scaling alpha are learning."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.3, 0.1]])
        y_b = Tensor([[0.3], [0.6]])

        manual_seed(4321)

        rpu_config = InferenceRPUConfig(mapping=MappingParameter(
            weight_scaling_omega=0.6,
            learn_out_scaling_alpha=True))

        model = Sequential(
            self.get_layer(4, 2, rpu_config=rpu_config),
            self.get_layer(2, 1, rpu_config=rpu_config)
        )
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            model = model.cuda()

        initial_out_scaling_alpha_0 = model[0].analog_tile.get_out_scaling_alpha().clone()
        initial_out_scaling_alpha_1 = model[1].analog_tile.get_out_scaling_alpha().clone()

        self.train_model(model, loss_func, x_b, y_b)

        learned_out_scaling_alpha_0 = model[0].analog_tile.get_out_scaling_alpha().data.clone()
        learned_out_scaling_alpha_1 = model[1].analog_tile.get_out_scaling_alpha().data.clone()

        self.assertEqual(initial_out_scaling_alpha_0.numel(), 1)
        self.assertIsNotNone(model[0].analog_tile.get_out_scaling_alpha().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_alpha_0, learned_out_scaling_alpha_0)

        self.assertEqual(initial_out_scaling_alpha_0.numel(), 1)
        self.assertIsNotNone(model[1].analog_tile.get_out_scaling_alpha().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_alpha_1, learned_out_scaling_alpha_1)
        self.assertEqual(initial_out_scaling_alpha_0.numel(), 1)

    def test_out_scaling_alpha_learning_columnwise(self):
        """Check if out scaling alpha are learning when columnwise is True."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.3, 0.1]])
        y_b = Tensor([[0.3], [0.6]])

        manual_seed(4321)

        rpu_config = InferenceRPUConfig(mapping=MappingParameter(
            weight_scaling_omega=0.6,
            learn_out_scaling_alpha=True,
            weight_scaling_omega_columnwise=True))

        model = Sequential(
            self.get_layer(4, 2, rpu_config=rpu_config),
            self.get_layer(2, 1, rpu_config=rpu_config)
        )
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            model = model.cuda()

        initial_out_scaling_alpha_0 = model[0].analog_tile.get_out_scaling_alpha().clone()
        initial_out_scaling_alpha_1 = model[1].analog_tile.get_out_scaling_alpha().clone()

        self.train_model(model, loss_func, x_b, y_b)

        learned_out_scaling_alpha_0 = model[0].analog_tile.get_out_scaling_alpha().clone()
        learned_out_scaling_alpha_1 = model[1].analog_tile.get_out_scaling_alpha().clone()

        self.assertGreaterEqual(initial_out_scaling_alpha_0.numel(), 1)
        self.assertIsNotNone(model[0].analog_tile.get_out_scaling_alpha().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_alpha_0, learned_out_scaling_alpha_0)

        self.assertGreaterEqual(initial_out_scaling_alpha_1.numel(), 1)
        self.assertIsNotNone(model[1].analog_tile.get_out_scaling_alpha().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_alpha_1, learned_out_scaling_alpha_1)

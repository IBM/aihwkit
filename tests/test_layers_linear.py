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

"""Tests for linear layer."""

from unittest import SkipTest

from numpy.testing import assert_array_almost_equal
from torch import Tensor, manual_seed
from torch.nn import Sequential, Linear as torchLinear
from torch.nn.functional import mse_loss
from torch.optim import SGD

from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.configs import InferenceRPUConfig, FloatingPointRPUConfig
from aihwkit.simulator.parameters import (
    MappingParameter,
    WeightModifierType,
    WeightModifierParameter,
    IOParameters,
    WeightRemapType,
)
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.inference.noise.custom import StateIndependentNoiseModel


from aihwkit.nn import AnalogSequential, AnalogLinear

from .helpers.decorators import parametrize_over_layers
from .helpers.layers import Linear, LinearCuda
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import FloatingPoint, IdealizedConstantStep, Inference, TorchInference, Custom


@parametrize_over_layers(
    layers=[Linear, LinearCuda],
    tiles=[FloatingPoint, IdealizedConstantStep, Inference, TorchInference, Custom],
    biases=["analog", "digital", None],
)
class LinearLayerTest(ParametrizedTestCase):
    """Linear layer abstractions tests."""

    @staticmethod
    def train_model(model, loss_func, x_b, y_b, **kwargs):
        """Train the model."""
        opt = AnalogSGD(model.parameters(), lr=0.5, **kwargs)
        opt.regroup_param_groups(model)

        epochs = 100
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(x_b)
            loss = loss_func(pred, y_b)

            loss.backward()
            opt.step()

    @staticmethod
    def train_model_torch(model, loss_func, x_b, y_b, **kwargs):
        """Train the model with torch SGD."""
        opt = SGD(model.parameters(), lr=0.5, **kwargs)
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
        model = Sequential(self.get_layer(4, 2), self.get_layer(2, 1))
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            model.cuda()

        initial_loss = loss_func(model(x_b), y_b)
        self.train_model(model, loss_func, x_b, y_b)
        self.assertLess(loss_func(model(x_b), y_b), initial_loss)

    def test_analog_and_digital(self):
        """Check mixing analog and digital layers."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.3, 0.1]])
        y_b = Tensor([[0.3], [0.6]])

        manual_seed(4321)
        model = Sequential(self.get_layer(4, 3), torchLinear(3, 3), self.get_layer(3, 1))
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            model.cuda()

        initial_loss = loss_func(model(x_b), y_b)
        self.train_model(model, loss_func, x_b, y_b)
        self.assertLess(loss_func(model(x_b), y_b), initial_loss)

    def test_analog_torch_optimizer(self):
        """Check analog layers with torch SGD for inference."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.3, 0.1]])
        y_b = Tensor([[0.3], [0.6]])

        manual_seed(4321)
        model = Sequential(self.get_layer(4, 3), self.get_layer(3, 1))
        if not isinstance(model[0].analog_module.rpu_config, InferenceRPUConfig):
            return

        rpu_config = FloatingPointRPUConfig(
            mapping=MappingParameter(digital_bias=self.digital_bias)
        )

        manual_seed(4321)
        model2 = AnalogSequential(
            AnalogLinear(4, 3, rpu_config=rpu_config, bias=self.bias),
            AnalogLinear(3, 1, rpu_config=rpu_config, bias=self.bias),
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
            param_group["lr"] = new_lr

        pred = model(x_b)
        loss = loss_func(pred, y_b)
        loss.backward()
        opt.step()

        if not layer1.analog_module.get_analog_ctx().use_torch_update:
            self.assertAlmostEqual(layer1.analog_module.get_learning_rate(), new_lr)

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

        if layer1.analog_module.analog_ctx.use_torch_update:
            raise SkipTest("Not supported")

        self.assertAlmostEqual(layer1.analog_module.get_learning_rate(), new_lr)
        self.assertAlmostEqual(layer2.analog_module.get_learning_rate(), new_lr)

    def test_out_scaling_learning(self):
        """Check if out scales are learning."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.3, 0.1]])
        y_b = Tensor([[0.3], [0.6]])

        manual_seed(4321)

        rpu_config = InferenceRPUConfig(
            mapping=MappingParameter(
                weight_scaling_omega=0.6, learn_out_scaling=True, out_scaling_columnwise=False
            )
        )

        model = Sequential(
            self.get_layer(4, 2, rpu_config=rpu_config), self.get_layer(2, 1, rpu_config=rpu_config)
        )
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            model = model.cuda()

        initial_out_scaling_0 = model[0].analog_module.get_learned_out_scales().clone()
        initial_out_scaling_1 = model[1].analog_module.get_learned_out_scales().clone()

        self.train_model(model, loss_func, x_b, y_b)

        learned_out_scaling_0 = model[0].analog_module.get_learned_out_scales().data.clone()
        learned_out_scaling_1 = model[1].analog_module.get_learned_out_scales().data.clone()

        self.assertEqual(initial_out_scaling_0.numel(), 1)
        self.assertIsNotNone(model[0].analog_module.get_learned_out_scales().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_0, learned_out_scaling_0)

        self.assertEqual(initial_out_scaling_0.numel(), 1)
        self.assertIsNotNone(model[1].analog_module.get_learned_out_scales().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_1, learned_out_scaling_1)
        self.assertEqual(initial_out_scaling_0.numel(), 1)

    def test_out_scaling_learning_columnwise(self):
        """Check if out scaling alpha are learning when columnwise is True."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.3, 0.1]])
        y_b = Tensor([[0.3], [0.6]])

        manual_seed(4321)

        rpu_config = InferenceRPUConfig(
            mapping=MappingParameter(
                weight_scaling_omega=0.6, learn_out_scaling=True, weight_scaling_columnwise=True
            )
        )

        model = Sequential(
            self.get_layer(4, 2, rpu_config=rpu_config), self.get_layer(2, 1, rpu_config=rpu_config)
        )
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            model = model.cuda()

        initial_out_scaling_0 = model[0].analog_module.get_learned_out_scales().clone()
        initial_out_scaling_1 = model[1].analog_module.get_learned_out_scales().clone()

        self.train_model(model, loss_func, x_b, y_b)

        learned_out_scaling_0 = model[0].analog_module.get_learned_out_scales().clone()
        learned_out_scaling_1 = model[1].analog_module.get_learned_out_scales().clone()

        self.assertGreaterEqual(initial_out_scaling_0.numel(), 1)
        self.assertIsNotNone(model[0].analog_module.get_learned_out_scales().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_0, learned_out_scaling_0)

        self.assertGreaterEqual(initial_out_scaling_1.numel(), 1)
        self.assertIsNotNone(model[1].analog_module.get_learned_out_scales().grad)
        self.assertNotAlmostEqualTensor(initial_out_scaling_1, learned_out_scaling_1)


@parametrize_over_layers(
    layers=[Linear, LinearCuda], tiles=[Inference, TorchInference], biases=["digital"]
)
class LinearLayerInferenceTest(ParametrizedTestCase):
    """Linear layer abstractions tests for inference."""

    def test_remapping_learning(self):
        """Check analog layers with torch SGD for inference."""
        loss_func = mse_loss

        x_b = Tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.3, 0.1]])
        y_b = Tensor([[0.3], [0.6]])

        manual_seed(4321)
        rpu_config = self.get_rpu_config()
        rpu_config.remap.type = WeightRemapType.LAYERWISE_SYMMETRIC
        rpu_config.forward.is_perfect = True

        rpu_config.mapping.learn_out_scaling = False
        rpu_config.mapping.weight_scaling_omega = 1.0
        rpu_config.mapping.weight_scaling_columnwise = False

        analog_model = Sequential(
            self.get_layer(4, 3, rpu_config=rpu_config), self.get_layer(3, 1, rpu_config=rpu_config)
        )

        manual_seed(4321)
        torch_model = Sequential(
            torchLinear(4, 3, bias=self.bias), torchLinear(3, 1, bias=self.bias)
        )

        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            analog_model = analog_model.cuda()
            torch_model = torch_model.cuda()

        initial_loss = loss_func(analog_model(x_b), y_b).detach().cpu().numpy()

        # train analog model with AnalogSGD
        LinearLayerTest.train_model(analog_model, loss_func, x_b, y_b)
        self.assertLess(loss_func(analog_model(x_b), y_b).detach().cpu().numpy(), initial_loss)

        # check remapping
        tile_weights = analog_model[0].get_weights(apply_weight_scaling=False)[0]
        self.assertAlmostEqual(tile_weights.abs().flatten().max().item(), 1.0)

        # train torch model with SGD
        initial_loss2 = loss_func(torch_model(x_b), y_b).detach().cpu().numpy()
        LinearLayerTest.train_model_torch(torch_model, loss_func, x_b, y_b)
        self.assertLess(loss_func(torch_model(x_b), y_b).detach().cpu().numpy(), initial_loss)

        # should be same
        final_loss = loss_func(analog_model(x_b), y_b).detach().cpu().numpy()
        final_loss2 = loss_func(torch_model(x_b), y_b).detach().cpu().numpy()

        assert_array_almost_equal(initial_loss, initial_loss2)
        assert_array_almost_equal(final_loss, final_loss2)

    def test_inference_modifier(self):
        """tests whether modifier are used"""

        x_b = Tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.3, 0.1]])

        rpu_config = self.get_rpu_config()

        rpu_config.mapping.weight_scaling_omega = 0.0
        rpu_config.mapping.learn_out_scaling = False
        rpu_config.mapping.weight_scaling_columnwise = False
        rpu_config.modifier = WeightModifierParameter(
            type=WeightModifierType.ADD_NORMAL, std_dev=1.0
        )
        rpu_config.forward = IOParameters(is_perfect=True)

        model = AnalogSequential(
            self.get_layer(4, 2, rpu_config=rpu_config), self.get_layer(2, 1, rpu_config=rpu_config)
        )
        if self.use_cuda:
            x_b = x_b.cuda()
            model = model.cuda()

        opt = AnalogSGD(model.parameters(), lr=0.0)
        opt.step()

        model.eval()
        y_eval1 = model(x_b)
        model.train()
        opt.step()
        model.eval()
        y_eval2 = model(x_b)

        model.train()
        y_train1 = model(x_b)
        opt.step()
        y_train2 = model(x_b)

        self.assertNotAlmostEqualTensor(y_train1, y_train2)
        self.assertTensorAlmostEqual(y_eval2, y_eval1)

    def test_drift_compensation(self):
        """tests whether drift compensation is performed"""

        x_b = Tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.3, 0.1]])

        rpu_config = self.get_rpu_config()

        rpu_config.mapping.weight_scaling_omega = 0.0
        rpu_config.mapping.learn_out_scaling = False
        rpu_config.mapping.weight_scaling_columnwise = False
        rpu_config.forward = IOParameters(is_perfect=True)
        rpu_config.drift_compensation = GlobalDriftCompensation()
        rpu_config.noise_model = StateIndependentNoiseModel(
            prog_noise_scale=0.0, read_noise_scale=0.0, drift_nu_std=0.0, drift_nu_mean=0.1
        )

        model = AnalogSequential(
            self.get_layer(4, 2, rpu_config=rpu_config), self.get_layer(2, 1, rpu_config=rpu_config)
        )

        rpu_config.drift_compensation = None
        model_without = AnalogSequential(
            self.get_layer(4, 2, rpu_config=rpu_config), self.get_layer(2, 1, rpu_config=rpu_config)
        )

        model_without.load_state_dict(model.state_dict(), load_rpu_config=False)

        if self.use_cuda:
            x_b = x_b.cuda()
            model_without.cuda()
            model.cuda()

        model.eval()
        y_before = model(x_b)
        model.drift_analog_weights(1000.0)
        y_after = model(x_b)

        model_without.eval()
        y_without_before = model_without(x_b)
        model_without.drift_analog_weights(1000.0)
        y_without_after = model_without(x_b)

        self.assertTensorAlmostEqual(y_before, y_without_before)
        self.assertTensorAlmostEqual(y_before, y_after)
        self.assertNotAlmostEqualTensor(y_after, y_without_after)
        self.assertNotAlmostEqualTensor(y_without_before, y_without_after)

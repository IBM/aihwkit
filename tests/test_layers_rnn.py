# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for RNN layers."""
from torch import randn, ones, no_grad
from torch.nn import MSELoss
from numpy.testing import assert_array_almost_equal, assert_raises

from aihwkit.optim import AnalogSGD
from aihwkit.optim.context import AnalogContext

from .helpers.decorators import parametrize_over_layers
from .helpers.layers import (
    LSTM,
    LSTMCuda,
    GRU,
    GRUCuda,
    VanillaRNN,
    VanillaRNNCuda,
    LSTMCombinedWeight,
    LSTMCombinedWeightCuda,
)
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import FloatingPoint, Inference, TorchInference, TorchInferenceIRDropT, Custom


@parametrize_over_layers(
    layers=[LSTM, VanillaRNN, GRU, LSTMCuda, GRUCuda, VanillaRNNCuda],
    tiles=[FloatingPoint, Inference, TorchInference, TorchInferenceIRDropT, Custom],
    biases=["analog", "digital", None],
)
class RNNLayerTest(ParametrizedTestCase):
    """Base test for RNNs"""

    @staticmethod
    def train_once(model, y_in, y_out, analog_if, lr=0.5, digital_bias_lr_scale=1.0):
        """Train once."""

        criterion = MSELoss()
        optimizer = AnalogSGD(model.parameters(), lr=lr, momentum=0.0, nesterov=0.0)
        optimizer.regroup_param_groups()
        batch_size = y_in.size()[1]

        if analog_if:
            for param_group in optimizer.param_groups:
                if isinstance(param_group["params"][0], AnalogContext):
                    param_group["lr"] = lr
                else:
                    param_group["lr"] = lr * digital_bias_lr_scale
            states = model.get_zero_state(batch_size)

        else:
            states = None

        for _ in range(2):
            optimizer.zero_grad()
            pred, _ = model(y_in, states)
            loss = criterion(pred.mean(axis=2, keepdim=True), y_out)
            loss.backward()
            optimizer.step()

        return pred.detach().cpu().numpy()

    @staticmethod
    def train_once_bidir(model, y_in, y_out, analog_if):
        """Train once."""
        criterion = MSELoss()
        optimizer = AnalogSGD(model.parameters(), lr=0.5, momentum=0.0, nesterov=0.0)
        batch_size = y_in.size()[1]

        if analog_if:
            states = model.get_zero_state(batch_size)
        else:
            states = None

        for _ in range(2):
            optimizer.zero_grad()
            pred, _ = model(y_in, states)
            loss = criterion(pred.mean(axis=2, keepdim=True), y_out)
            loss.backward()
            optimizer.step()

        return pred.detach().cpu().numpy()

    def test_layer_instantiation(self):
        """Test AnalogLSTM layer instantiation."""
        input_size = 2
        hidden_size = 3
        num_layers = 4

        model = self.get_layer(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
        )

        # Assert over the stacked layers.
        self.assertEqual(len(model.rnn.layers), num_layers)
        for i, layer in enumerate(model.rnn.layers):
            # Assert over the size of weight_ih.
            if i == 0:
                self.assertEqual(layer.cell.weight_ih.in_features, input_size)
            else:
                self.assertEqual(layer.cell.weight_ih.in_features, hidden_size)
            self.assertEqual(layer.cell.weight_hh.in_features, hidden_size)
            # Assert over the rpu_config.
            analog_tile_ih = next(layer.cell.weight_ih.analog_tiles())
            analog_tile_hh = next(layer.cell.weight_hh.analog_tiles())
            self.assertEqual(analog_tile_ih.rpu_config.__class__, self.get_rpu_config().__class__)
            self.assertEqual(analog_tile_hh.rpu_config.__class__, self.get_rpu_config().__class__)

    def get_native_layer_comparison(self, *args, **kwargs):
        """Returns the torch native model"""
        raise NotImplementedError

    def test_layer_training(self):
        """Test AnalogLSTM layer training."""

        # pylint: disable=too-many-locals, too-many-statements
        def get_parameters(model, analog_if) -> dict:
            """Returns the parameter in an dict."""

            dic = {}
            for name, param in model.named_parameters():
                if isinstance(param, AnalogContext):
                    weight, bias = param.analog_tile.get_weights()
                    splits = name.split(".")
                    add_on = "_" + splits[-3].split("_")[-1] + "_l" + splits[2]
                    dic["weight" + add_on] = weight
                    if bias is not None:
                        dic["bias" + add_on] = bias
                elif analog_if and name.endswith("bias"):  # digital bias
                    splits = name.split(".")
                    add_on = "_" + splits[-3].split("_")[-1] + "_l" + splits[2]
                    dic["bias" + add_on] = param
                else:
                    dic[name] = param

            return dic

        input_size = 2
        hidden_size = 2
        num_layers = 2
        seq_length = 3
        batch_size = 3
        test_for_update = False  # For debugging. Does test whether all weights are updated.

        # Make dataset (just random).
        y_in = randn(seq_length, batch_size, input_size)
        y_out = ones(seq_length, batch_size, 1)

        rnn_analog = self.get_layer(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.0
        )

        rnn = self.get_native_layer_comparison(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0,
            bias=self.bias,
        )

        weights_org = rnn_analog.get_weights()

        rnn_pars0 = get_parameters(rnn, False)
        rnn_analog_pars0 = get_parameters(rnn_analog, True)

        for par_name, par_item in rnn_pars0.items():
            par_item.data = rnn_analog_pars0[par_name].detach().clone()

        # Make independent for comparison below.
        rnn_pars0 = {key: value.detach().clone() for key, value in rnn_pars0.items()}

        if self.use_cuda:
            y_in = y_in.cuda()
            y_out = y_out.cuda()
            rnn_analog.cuda()
            rnn.cuda()

        with no_grad():
            self.assertTensorAlmostEqual(rnn(y_in)[0], rnn_analog(y_in)[0])

        # First train analog and make sure weights differ.
        pred_analog = self.train_once(rnn_analog, y_in, y_out, True)

        analog_weights = rnn_analog.get_weights()

        if test_for_update:
            for weight, weight_org in zip(analog_weights.values(), weights_org.values()):
                assert_raises(AssertionError, assert_array_almost_equal, weight[0], weight_org[0])

        # Compare with RNN.
        pred = self.train_once(rnn, y_in, y_out, False)
        assert_array_almost_equal(pred, pred_analog)

        rnn_pars = get_parameters(rnn, False)
        rnn_analog_pars = get_parameters(rnn_analog, True)

        if test_for_update:
            for par_name, par_item in rnn_pars.items():
                par0 = rnn_pars0[par_name].detach().cpu().numpy()
                par = par_item.detach().cpu().numpy()
                assert_raises(AssertionError, assert_array_almost_equal, par, par0)

        for par_name, par_item in rnn_pars.items():
            assert_array_almost_equal(
                par_item.detach().cpu().numpy(), rnn_analog_pars[par_name].detach().cpu().numpy()
            )

    def test_bidir_layer_training(self):
        """Test AnalogLSTM bidirectional layer training."""

        # pylint: disable=too-many-locals, too-many-statements
        def get_parameters(model, analog_if) -> dict:
            """Returns the parameter in an dict."""
            dic = {}
            for name, param in model.named_parameters():
                if isinstance(param, AnalogContext):
                    weight, bias = param.analog_tile.get_weights()
                    splits = name.split(".")
                    add_on = "_" + splits[-3].split("_")[-1] + "_l" + splits[2]
                    if splits[4] == "1":
                        add_on += "_reverse"

                    dic["weight" + add_on] = weight
                    if bias is not None:
                        dic["bias" + add_on] = bias
                elif analog_if and name.endswith("bias"):  # digital bias
                    splits = name.split(".")
                    add_on = "_" + splits[-3].split("_")[-1] + "_l" + splits[2]
                    if splits[4] == "1":
                        add_on += "_reverse"
                    dic["bias" + add_on] = param
                else:
                    dic[name] = param

            return dic

        input_size = 4
        hidden_size = 5
        num_layers = 2
        seq_length = 10
        batch_size = 3
        test_for_update = False  # For debugging. Does test whether all weights are updated.

        # Make dataset (just random).
        y_in = randn(seq_length, batch_size, input_size)
        y_out = ones(seq_length, batch_size, 1)

        rnn_analog = self.get_layer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0,
            bidir=True,
        )

        rnn = self.get_native_layer_comparison(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0,
            bias=self.bias,
            bidirectional=True,
        )

        weights_org = rnn_analog.get_weights()

        rnn_pars0 = get_parameters(rnn, False)
        rnn_analog_pars0 = get_parameters(rnn_analog, True)

        for par_name, par_item in rnn_pars0.items():
            par_item.data = rnn_analog_pars0[par_name].detach().clone()

        # Make independent for comparison below.
        rnn_pars0 = {key: value.detach().clone() for key, value in rnn_pars0.items()}

        if self.use_cuda:
            y_in = y_in.cuda()
            y_out = y_out.cuda()
            rnn_analog.cuda()
            rnn.cuda()

        with no_grad():
            self.assertTensorAlmostEqual(rnn(y_in)[0], rnn_analog(y_in)[0])

        # First train analog and make sure weights differ.
        pred_analog = self.train_once_bidir(rnn_analog, y_in, y_out, True)

        analog_weights = rnn_analog.get_weights()

        if test_for_update:
            for weight, weight_org in zip(analog_weights.values(), weights_org.values()):
                self.assertNotAlmostEqualTensor(weight[0], weight_org[0])

        # Compare with RNN.
        pred = self.train_once_bidir(rnn, y_in, y_out, False)
        assert_array_almost_equal(pred, pred_analog)

        rnn_pars = get_parameters(rnn, False)
        rnn_analog_pars = get_parameters(rnn_analog, True)

        if test_for_update:
            for par_name, par_item in rnn_pars.items():
                par0 = rnn_pars0[par_name].detach().cpu().numpy()
                par = par_item.detach().cpu().numpy()
                assert_raises(AssertionError, assert_array_almost_equal, par, par0)

        for par_name, par_item in rnn_pars.items():
            assert_array_almost_equal(
                par_item.detach().cpu().numpy(), rnn_analog_pars[par_name].detach().cpu().numpy()
            )


@parametrize_over_layers(
    layers=[LSTMCombinedWeight, LSTMCombinedWeightCuda],
    tiles=[FloatingPoint],
    biases=["digital", None],
)
class LSTMCombinedWeightTest(RNNLayerTest):
    """Base test for AnalogLSTMCombinedWeight"""

    def test_layer_instantiation(self):
        """Test AnalogLSTM layer instantiation."""
        input_size = 2
        hidden_size = 3
        num_layers = 4

        model = self.get_layer(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
        )

        # Assert over the stacked layers.
        self.assertEqual(len(model.rnn.layers), num_layers)
        for i, layer in enumerate(model.rnn.layers):
            # Assert over the size of weight_ih.
            if i == 0:
                self.assertEqual(layer.cell.weight.in_features, input_size + hidden_size)
            else:
                self.assertEqual(layer.cell.weight.in_features, 2 * hidden_size)
            self.assertEqual(layer.cell.weight.out_features, 4 * hidden_size)
            # Assert over the rpu_config.
            analog_tile = list(layer.cell.weight.analog_tiles())[0]
            self.assertEqual(analog_tile.rpu_config.__class__, self.get_rpu_config().__class__)

    def get_native_layer_comparison(self, *args, **kwargs):
        """Returns the torch native model"""
        raise NotImplementedError

    def test_layer_training(self):
        """Test AnalogLSTM layer training."""

        # pylint: disable=too-many-locals, too-many-statements
        def get_parameters(model, analog_if) -> dict:
            """Returns the parameter in an dict."""

            dic = {}
            for name, param in model.named_parameters():
                if isinstance(param, AnalogContext):
                    weight, bias = param.analog_tile.get_weights()
                    lay = int(name.split(".")[2])
                    add_on = "_l" + str(lay)
                    if lay == 0:
                        in_dim = input_size
                    else:
                        in_dim = hidden_size
                    dic["weight_ih" + add_on] = weight[..., :in_dim]
                    dic["weight_hh" + add_on] = weight[..., in_dim:]

                    if bias is not None:
                        dic["bias_ih" + add_on] = bias
                        dic["bias_hh" + add_on] = 0.0 * bias

                elif analog_if and name.endswith("bias"):  # digital bias
                    add_on = "_l" + name.split(".")[2]
                    dic["bias_ih" + add_on] = param
                    dic["bias_hh" + add_on] = 0.0 * param
                else:
                    dic[name] = param

            return dic

        input_size = 5
        hidden_size = 3
        num_layers = 2
        seq_length = 10
        batch_size = 4
        test_for_update = True  # For debugging. Does test whether all weights are updated.

        # Make dataset (just random).
        y_in = randn(seq_length, batch_size, input_size)
        y_out = ones(seq_length, batch_size, 1)

        rnn_analog = self.get_layer(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.0
        )

        rnn = self.get_native_layer_comparison(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0,
            bias=self.bias,
        )

        rnn_pars0 = get_parameters(rnn, False)
        rnn_analog_pars0 = get_parameters(rnn_analog, True)

        for par_name, par_item in rnn_pars0.items():
            par_item.data = rnn_analog_pars0[par_name].detach().clone()

        if test_for_update:
            weights_org = rnn_analog.get_weights()

        if self.use_cuda:
            y_in = y_in.cuda()
            y_out = y_out.cuda()
            rnn_analog.cuda()
            rnn.cuda()

        with no_grad():
            assert_array_almost_equal(
                rnn(y_in)[0].detach().clone().cpu(), rnn_analog(y_in)[0].detach().clone().cpu()
            )

        # First train analog and make sure weights differ.
        # since there is only one bias the LR of the bias is changed
        pred_analog = self.train_once(rnn_analog, y_in, y_out, True, digital_bias_lr_scale=2.0)

        if test_for_update:
            analog_weights = rnn_analog.get_weights()

            for weight, weight_org in zip(analog_weights.values(), weights_org.values()):
                self.assertNotAlmostEqualTensor(weight[0], weight_org[0])

        # Compare with RNN.
        pred = self.train_once(rnn, y_in, y_out, False)

        assert_array_almost_equal(pred, pred_analog)

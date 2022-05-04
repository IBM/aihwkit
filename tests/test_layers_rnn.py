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

"""Tests for RNN layers."""

from torch import randn, ones
from torch.nn import MSELoss
from numpy.testing import assert_array_almost_equal, assert_raises

from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.configs import InferenceRPUConfig
from aihwkit.optim.context import AnalogContext

from .helpers.decorators import parametrize_over_layers
from .helpers.layers import LSTM, LSTMCuda, GRU, GRUCuda, VanillaRNN, VanillaRNNCuda
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import FloatingPoint, Inference


@parametrize_over_layers(
    layers=[LSTM, VanillaRNN, GRU, LSTMCuda, GRUCuda, VanillaRNNCuda],
    tiles=[FloatingPoint, Inference],
    biases=['analog', 'digital', None]
)
class RNNLayerTest(ParametrizedTestCase):
    """ Base test for RNNs"""

    @staticmethod
    def train_once(model, y_in, y_out, analog_if):
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

        model = self.get_layer(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers)

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
            if not isinstance(layer.cell.weight_ih.analog_tile.rpu_config, InferenceRPUConfig):
                self.assertEqual(layer.cell.weight_ih.analog_tile.rpu_config, self.get_rpu_config())
                self.assertEqual(layer.cell.weight_hh.analog_tile.rpu_config, self.get_rpu_config())
            else:
                self.assertEqual(layer.cell.weight_ih.analog_tile.rpu_config.__class__,
                                 self.get_rpu_config().__class__)
                self.assertEqual(layer.cell.weight_hh.analog_tile.rpu_config.__class__,
                                 self.get_rpu_config().__class__)

    def get_native_layer_comparison(self, *args, **kwargs):
        """ Returns the torch native model """
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
                    splits = name.split('.')
                    add_on = '_' + splits[-2].split('_')[-1] + '_l' + splits[2]

                    dic['weight' + add_on] = weight
                    if bias is not None:
                        dic['bias' + add_on] = bias
                elif analog_if and name.endswith('bias'):  # digital bias
                    splits = name.split('.')
                    add_on = '_' + splits[-2].split('_')[-1] + '_l' + splits[2]
                    dic['bias' + add_on] = param
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

        rnn_analog = self.get_layer(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    realistic_read_write=False,
                                    dropout=0.0)

        rnn = self.get_native_layer_comparison(input_size=input_size,
                                               hidden_size=hidden_size,
                                               num_layers=num_layers,
                                               dropout=0.0,
                                               bias=self.bias)

        weights_org = []

        # pylint: disable=protected-access
        rnn_analog._apply_to_analog(lambda lay: weights_org.append(
            lay.analog_tile.tile.get_weights()))

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

        # First train analog and make sure weights differ.
        pred_analog = self.train_once(rnn_analog, y_in, y_out, True)

        analog_weights = []
        rnn_analog._apply_to_analog(lambda lay: analog_weights.append(
            lay.analog_tile.tile.get_weights()))

        if test_for_update:
            for weight, weight_org in zip(analog_weights, weights_org):
                assert_raises(AssertionError, assert_array_almost_equal, weight, weight_org)

        # Compare with RNN.
        pred = self.train_once(rnn, y_in, y_out, False)
        assert_array_almost_equal(pred, pred_analog)

        rnn_analog._apply_to_analog(lambda lay: lay._sync_weights_from_tile())

        rnn_pars = get_parameters(rnn, False)
        rnn_analog_pars = get_parameters(rnn_analog, True)

        if test_for_update:
            for par_name, par_item in rnn_pars.items():
                par0 = rnn_pars0[par_name].detach().cpu().numpy()
                par = par_item.detach().cpu().numpy()
                assert_raises(AssertionError, assert_array_almost_equal, par, par0)

        for par_name, par_item in rnn_pars.items():
            assert_array_almost_equal(par_item.detach().cpu().numpy(),
                                      rnn_analog_pars[par_name].detach().cpu().numpy())

    def test_bidir_layer_training(self):
        """Test AnalogLSTM bidirectional layer training."""
        # pylint: disable=too-many-locals, too-many-statements
        def get_parameters(model, analog_if) -> dict:
            """Returns the parameter in an dict."""
            dic = {}
            for name, param in model.named_parameters():
                if isinstance(param, AnalogContext):
                    weight, bias = param.analog_tile.get_weights()
                    splits = name.split('.')
                    add_on = '_' + splits[-2].split('_')[-1] + '_l' + splits[2]
                    if splits[4] == '1':
                        add_on += '_reverse'

                    dic['weight' + add_on] = weight
                    if bias is not None:
                        dic['bias' + add_on] = bias
                elif analog_if and name.endswith('bias'):  # digital bias
                    splits = name.split('.')
                    add_on = '_' + splits[-2].split('_')[-1] + '_l' + splits[2]
                    if splits[4] == '1':
                        add_on += '_reverse'
                    dic['bias' + add_on] = param
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

        rnn_analog = self.get_layer(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    realistic_read_write=False,
                                    dropout=0.0,
                                    bidir=True)

        rnn = self.get_native_layer_comparison(input_size=input_size,
                                               hidden_size=hidden_size,
                                               num_layers=num_layers,
                                               dropout=0.0,
                                               bias=self.bias,
                                               bidirectional=True)

        weights_org = []

        # pylint: disable=protected-access
        rnn_analog._apply_to_analog(lambda lay: weights_org.append(
            lay.analog_tile.tile.get_weights()))

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

        # First train analog and make sure weights differ.
        pred_analog = self.train_once_bidir(rnn_analog, y_in, y_out, True)

        analog_weights = []
        rnn_analog._apply_to_analog(lambda lay: analog_weights.append(
            lay.analog_tile.tile.get_weights()))

        if test_for_update:
            for weight, weight_org in zip(analog_weights, weights_org):
                assert_raises(AssertionError, assert_array_almost_equal, weight, weight_org)

        # Compare with RNN.
        pred = self.train_once_bidir(rnn, y_in, y_out, False)
        assert_array_almost_equal(pred, pred_analog)

        rnn_analog._apply_to_analog(lambda lay: lay._sync_weights_from_tile())

        rnn_pars = get_parameters(rnn, False)
        rnn_analog_pars = get_parameters(rnn_analog, True)

        if test_for_update:
            for par_name, par_item in rnn_pars.items():
                par0 = rnn_pars0[par_name].detach().cpu().numpy()
                par = par_item.detach().cpu().numpy()
                assert_raises(AssertionError, assert_array_almost_equal, par, par0)

        for par_name, par_item in rnn_pars.items():
            assert_array_almost_equal(par_item.detach().cpu().numpy(),
                                      rnn_analog_pars[par_name].detach().cpu().numpy())

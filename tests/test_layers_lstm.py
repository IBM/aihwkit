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

"""Tests for LSTM layers."""

from torch import randn, ones, zeros
from torch.nn import MSELoss
from torch.nn import LSTM as LSTM_nn
from torch.nn import GRU as GRU_nn
from torch.nn import RNN as RNN_nn
from numpy.testing import assert_array_almost_equal, assert_raises

from aihwkit.nn.modules.rnn.cells import LSTMState
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.configs import InferenceRPUConfig
from aihwkit.optim.context import AnalogContext

from .helpers.decorators import parametrize_over_layers
from .helpers.layers import LSTM, LSTMCuda, GRU, GRUCuda, VanillaRNN, VanillaRNNCuda
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import FloatingPoint, Inference


@parametrize_over_layers(
    layers=[LSTM, LSTMCuda],
    tiles=[FloatingPoint, Inference],
    biases=['analog', 'digital', None]
)
class LSTMLayerTest(ParametrizedTestCase):
    """Tests for AnalogLSTM layer."""

    @staticmethod
    def train_once(model, y_in, y_out, analog_if, use_cuda=False):
        """Train once."""
        criterion = MSELoss()
        optimizer = AnalogSGD(model.parameters(), lr=0.5, momentum=0.0, nesterov=0.0)
        optimizer.regroup_param_groups(model)
        if analog_if:
            # why is this format so difference?
            # TODO: better use same state format as for native Pytorch's LSTM?
            if use_cuda:
                states = [LSTMState(zeros(y_in.size()[1], model.hidden_size).cuda(),
                                    zeros(y_in.size()[1], model.hidden_size).cuda())
                          for _ in range(model.num_layers)]
            else:
                states = [LSTMState(zeros(y_in.size()[1], model.hidden_size),
                                    zeros(y_in.size()[1], model.hidden_size))
                          for _ in range(model.num_layers)]

        else:
            if use_cuda:
                states = (zeros(model.num_layers, y_in.size()[1], model.hidden_size).cuda(),
                          zeros(model.num_layers, y_in.size()[1], model.hidden_size).cuda())
            else:
                states = (zeros(model.num_layers, y_in.size()[1], model.hidden_size),
                          zeros(model.num_layers, y_in.size()[1], model.hidden_size))

        for _ in range(2):
            optimizer.zero_grad()
            pred, _ = model(y_in, states)
            loss = criterion(pred.mean(axis=2, keepdim=True), y_out)
            loss.backward()
            optimizer.step()

        return pred.detach().cpu().numpy()

    @staticmethod
    def train_once_bidir(model, y_in, y_out, analog_if, use_cuda=False):
        """Train once."""
        criterion = MSELoss()
        optimizer = AnalogSGD(model.parameters(), lr=0.5, momentum=0.0, nesterov=0.0)
        optimizer.regroup_param_groups(model)
        if analog_if:
            # why is this format so difference?
            # TODO: better use same state format as for native Pytorch's LSTM?
            if use_cuda:
                states = [[LSTMState(zeros(y_in.size()[1], model.hidden_size).cuda(), zeros(y_in.size()[1], model.hidden_size).cuda()),
                           LSTMState(zeros(y_in.size()[1], model.hidden_size).cuda(), zeros(y_in.size()[1], model.hidden_size).cuda())]
                          for _ in range(model.num_layers)]
            else:
                states = [[LSTMState(zeros(y_in.size()[1], model.hidden_size), zeros(y_in.size()[1], model.hidden_size)),
                           LSTMState(zeros(y_in.size()[1], model.hidden_size), zeros(y_in.size()[1], model.hidden_size))]
                          for _ in range(model.num_layers)]

        else:
            if use_cuda:
                states = (zeros(model.num_layers*2, y_in.size()[1], model.hidden_size).cuda(),
                          zeros(model.num_layers*2, y_in.size()[1], model.hidden_size).cuda())
            else:
                states = (zeros(model.num_layers*2, y_in.size()[1], model.hidden_size),
                          zeros(model.num_layers*2, y_in.size()[1], model.hidden_size))
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
                # TODO: comparison of rpu_config not possible..
                self.assertEqual(layer.cell.weight_ih.analog_tile.rpu_config, self.get_rpu_config())
                self.assertEqual(layer.cell.weight_hh.analog_tile.rpu_config, self.get_rpu_config())
            else:
                self.assertEqual(layer.cell.weight_ih.analog_tile.rpu_config.__class__,
                                 self.get_rpu_config().__class__)
                self.assertEqual(layer.cell.weight_hh.analog_tile.rpu_config.__class__,
                                 self.get_rpu_config().__class__)

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
        hidden_size = 3
        num_layers = 2
        seq_length = 10
        batch_size = 3
        test_for_update = False  # For debugging. Does test whether all weights are updated.

        # Make dataset (just random).
        y_in = randn(seq_length, batch_size, input_size)
        y_out = ones(seq_length, batch_size, 1)

        lstm_analog = self.get_layer(input_size=input_size,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     realistic_read_write=False,
                                     dropout=0.0)

        lstm = LSTM_nn(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       dropout=0.0,
                       bias=self.bias)

        weights_org = []

        # pylint: disable=protected-access
        lstm_analog._apply_to_analog(lambda lay: weights_org.append(
            lay.analog_tile.tile.get_weights()))

        lstm_pars0 = get_parameters(lstm, False)
        lstm_analog_pars0 = get_parameters(lstm_analog, True)

        for par_name, par_item in lstm_pars0.items():
            par_item.data = lstm_analog_pars0[par_name].detach().clone()

        # Make independent for comparison below.
        lstm_pars0 = {key: value.detach().clone() for key, value in lstm_pars0.items()}

        if self.use_cuda:
            y_in = y_in.cuda()
            y_out = y_out.cuda()
            lstm_analog.cuda()
            lstm.cuda()

        # First train analog and make sure weights differ.
        pred_analog = self.train_once(lstm_analog, y_in, y_out, True, use_cuda=self.use_cuda)

        analog_weights = []
        lstm_analog._apply_to_analog(lambda lay: analog_weights.append(
            lay.analog_tile.tile.get_weights()))

        if test_for_update:
            for weight, weight_org in zip(analog_weights, weights_org):
                assert_raises(AssertionError, assert_array_almost_equal, weight, weight_org)

        # Compare with LSTM.
        pred = self.train_once(lstm, y_in, y_out, False, use_cuda=self.use_cuda)
        assert_array_almost_equal(pred, pred_analog)

        lstm_analog._apply_to_analog(lambda lay: lay._sync_weights_from_tile())

        lstm_pars = get_parameters(lstm, False)
        lstm_analog_pars = get_parameters(lstm_analog, True)

        if test_for_update:
            for par_name, par_item in lstm_pars.items():
                par0 = lstm_pars0[par_name].detach().cpu().numpy()
                par = par_item.detach().cpu().numpy()
                assert_raises(AssertionError, assert_array_almost_equal, par, par0)

        for par_name, par_item in lstm_pars.items():
            assert_array_almost_equal(par_item.detach().cpu().numpy(),
                                      lstm_analog_pars[par_name].detach().cpu().numpy())

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
        hidden_size = 3
        num_layers = 2
        seq_length = 10
        batch_size = 3
        test_for_update = False  # For debugging. Does test whether all weights are updated.

        # Make dataset (just random).
        y_in = randn(seq_length, batch_size, input_size)
        y_out = ones(seq_length, batch_size, 1)

        lstm_analog = self.get_layer(input_size=input_size,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     realistic_read_write=False,
                                     dropout=0.0,
                                     bidir=True)

        lstm = LSTM_nn(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       dropout=0.0,
                       bias=self.bias,
                       bidirectional=True)

        weights_org = []

        # pylint: disable=protected-access
        lstm_analog._apply_to_analog(lambda lay: weights_org.append(
            lay.analog_tile.tile.get_weights()))

        lstm_pars0 = get_parameters(lstm, False)
        lstm_analog_pars0 = get_parameters(lstm_analog, True)

        print(lstm_pars0.keys())
        print('\n\n')
        print(lstm_analog_pars0.keys())
        # exit()

        for par_name, par_item in lstm_pars0.items():
            par_item.data = lstm_analog_pars0[par_name].detach().clone()

        # Make independent for comparison below.
        lstm_pars0 = {key: value.detach().clone() for key, value in lstm_pars0.items()}

        if self.use_cuda:
            y_in = y_in.cuda()
            y_out = y_out.cuda()
            lstm_analog.cuda()
            lstm.cuda()

        # First train analog and make sure weights differ.
        pred_analog = self.train_once_bidir(lstm_analog, y_in, y_out, True, use_cuda=self.use_cuda)

        analog_weights = []
        lstm_analog._apply_to_analog(lambda lay: analog_weights.append(
            lay.analog_tile.tile.get_weights()))

        if test_for_update:
            for weight, weight_org in zip(analog_weights, weights_org):
                assert_raises(AssertionError, assert_array_almost_equal, weight, weight_org)

        # Compare with LSTM.
        pred = self.train_once_bidir(lstm, y_in, y_out, False, use_cuda=self.use_cuda)
        assert_array_almost_equal(pred, pred_analog)

        lstm_analog._apply_to_analog(lambda lay: lay._sync_weights_from_tile())

        lstm_pars = get_parameters(lstm, False)
        lstm_analog_pars = get_parameters(lstm_analog, True)

        if test_for_update:
            for par_name, par_item in lstm_pars.items():
                par0 = lstm_pars0[par_name].detach().cpu().numpy()
                par = par_item.detach().cpu().numpy()
                assert_raises(AssertionError, assert_array_almost_equal, par, par0)

        for par_name, par_item in lstm_pars.items():
            assert_array_almost_equal(par_item.detach().cpu().numpy(),
                                      lstm_analog_pars[par_name].detach().cpu().numpy())


@parametrize_over_layers(
    layers=[GRU, GRUCuda],
    tiles=[FloatingPoint, Inference],
    biases=['analog', 'digital', None]
)
class GRULayerTest(ParametrizedTestCase):
    """Tests for AnalogGRU layer."""

    @staticmethod
    def train_once(model, y_in, y_out, analog_if, use_cuda=False):
        """Train once."""
        criterion = MSELoss()
        optimizer = AnalogSGD(model.parameters(), lr=0.5, momentum=0.0, nesterov=0.0)
        optimizer.regroup_param_groups(model)
        if analog_if:
            # why is this format so difference?
            # TODO: better use same state format as for native Pytorch's LSTM?
            if use_cuda:
                states = [zeros(y_in.size()[1], model.hidden_size).cuda()
                          for _ in range(model.num_layers)]
            else:
                states = [zeros(y_in.size()[1], model.hidden_size)
                          for _ in range(model.num_layers)] 

        else:
            if use_cuda:
                states = zeros(model.num_layers, y_in.size()[1], model.hidden_size).cuda()
            else:
                states = zeros(model.num_layers, y_in.size()[1], model.hidden_size)

        for _ in range(2):
            optimizer.zero_grad()
            pred, _ = model(y_in, states)
            loss = criterion(pred.mean(axis=2, keepdim=True), y_out)
            loss.backward()
            optimizer.step()

        return pred.detach().cpu().numpy()

    @staticmethod
    def train_once_bidir(model, y_in, y_out, analog_if, use_cuda=False):
        """Train once."""
        criterion = MSELoss()
        optimizer = AnalogSGD(model.parameters(), lr=0.5, momentum=0.0, nesterov=0.0)
        optimizer.regroup_param_groups(model)
        if analog_if:
            # why is this format so difference?
            # TODO: better use same state format as for native Pytorch's LSTM?
            if use_cuda:
                states = [[zeros(y_in.size()[1], model.hidden_size).cuda(), zeros(y_in.size()[1], model.hidden_size).cuda()]
                          for _ in range(model.num_layers)]
            else:
                states = [[zeros(y_in.size()[1], model.hidden_size), zeros(y_in.size()[1], model.hidden_size)]
                          for _ in range(model.num_layers)] 

        else:
            if use_cuda:
                states = zeros(model.num_layers*2, y_in.size()[1], model.hidden_size).cuda()
            else:
                states = zeros(model.num_layers*2, y_in.size()[1], model.hidden_size)
        for _ in range(2):
            optimizer.zero_grad()
            pred, _ = model(y_in, states)
            loss = criterion(pred.mean(axis=2, keepdim=True), y_out)
            loss.backward()
            optimizer.step()

        return pred.detach().cpu().numpy()

    def test_layer_instantiation(self):
        """Test AnalogGRU layer instantiation."""
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
                # TODO: comparison of rpu_config not possible..
                self.assertEqual(layer.cell.weight_ih.analog_tile.rpu_config, self.get_rpu_config())
                self.assertEqual(layer.cell.weight_hh.analog_tile.rpu_config, self.get_rpu_config())
            else:
                self.assertEqual(layer.cell.weight_ih.analog_tile.rpu_config.__class__,
                                 self.get_rpu_config().__class__)
                self.assertEqual(layer.cell.weight_hh.analog_tile.rpu_config.__class__,
                                 self.get_rpu_config().__class__)

    def test_layer_training(self):
        """Test AnalogGRU layer training."""
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
        hidden_size = 3
        num_layers = 1
        seq_length = 10
        batch_size = 3
        test_for_update = False  # For debugging. Does test whether all weights are updated.

        # Make dataset (just random).
        y_in = randn(seq_length, batch_size, input_size)
        y_out = ones(seq_length, batch_size, 1)

        gru_analog = self.get_layer(input_size=input_size,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     realistic_read_write=False,
                                     dropout=0.0)

        gru = GRU_nn(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       dropout=0.0,
                       bias=self.bias)

        weights_org = []

        # pylint: disable=protected-access
        gru_analog._apply_to_analog(lambda lay: weights_org.append(
            lay.analog_tile.tile.get_weights()))

        gru_pars0 = get_parameters(gru, False)
        gru_analog_pars0 = get_parameters(gru_analog, True)

        for par_name, par_item in gru_pars0.items():
            par_item.data = gru_analog_pars0[par_name].detach().clone()

        # Make independent for comparison below.
        gru_pars0 = {key: value.detach().clone() for key, value in gru_pars0.items()}

        if self.use_cuda:
            y_in = y_in.cuda()
            y_out = y_out.cuda()
            gru_analog.cuda()
            gru.cuda()

        # First train analog and make sure weights differ.
        pred_analog = self.train_once(gru_analog, y_in, y_out, True, use_cuda=self.use_cuda)

        analog_weights = []
        gru_analog._apply_to_analog(lambda lay: analog_weights.append(
            lay.analog_tile.tile.get_weights()))

        if test_for_update:
            for weight, weight_org in zip(analog_weights, weights_org):
                assert_raises(AssertionError, assert_array_almost_equal, weight, weight_org)

        # Compare with GRU.
        pred = self.train_once(gru, y_in, y_out, False, use_cuda=self.use_cuda)
        assert_array_almost_equal(pred, pred_analog)

        gru_analog._apply_to_analog(lambda lay: lay._sync_weights_from_tile())

        gru_pars = get_parameters(gru, False)
        gru_analog_pars = get_parameters(gru_analog, True)

        if test_for_update:
            for par_name, par_item in gru_pars.items():
                par0 = gru_pars0[par_name].detach().cpu().numpy()
                par = par_item.detach().cpu().numpy()
                assert_raises(AssertionError, assert_array_almost_equal, par, par0)

        for par_name, par_item in gru_pars.items():
            assert_array_almost_equal(par_item.detach().cpu().numpy(),
                                      gru_analog_pars[par_name].detach().cpu().numpy())

    def test_bidir_layer_training(self):
        """Test AnalogGRU bidirectional layer training."""
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
        hidden_size = 3
        num_layers = 1
        seq_length = 10
        batch_size = 3
        test_for_update = False  # For debugging. Does test whether all weights are updated.

        # Make dataset (just random).
        y_in = randn(seq_length, batch_size, input_size)
        y_out = ones(seq_length, batch_size, 1)

        gru_analog = self.get_layer(input_size=input_size,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     realistic_read_write=False,
                                     dropout=0.0,
                                     bidir=True)

        gru = GRU_nn(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       dropout=0.0,
                       bias=self.bias,
                       bidirectional=True)

        weights_org = []

        # pylint: disable=protected-access
        gru_analog._apply_to_analog(lambda lay: weights_org.append(
            lay.analog_tile.tile.get_weights()))

        gru_pars0 = get_parameters(gru, False)
        gru_analog_pars0 = get_parameters(gru_analog, True)

        for par_name, par_item in gru_pars0.items():
            par_item.data = gru_analog_pars0[par_name].detach().clone()

        # Make independent for comparison below.
        gru_pars0 = {key: value.detach().clone() for key, value in gru_pars0.items()}

        if self.use_cuda:
            y_in = y_in.cuda()
            y_out = y_out.cuda()
            gru_analog.cuda()
            gru.cuda()

        # First train analog and make sure weights differ.
        pred_analog = self.train_once_bidir(gru_analog, y_in, y_out, True, use_cuda=self.use_cuda)

        analog_weights = []
        gru_analog._apply_to_analog(lambda lay: analog_weights.append(
            lay.analog_tile.tile.get_weights()))

        if test_for_update:
            for weight, weight_org in zip(analog_weights, weights_org):
                assert_raises(AssertionError, assert_array_almost_equal, weight, weight_org)

        # Compare with GRU.
        pred = self.train_once_bidir(gru, y_in, y_out, False, use_cuda=self.use_cuda)
        assert_array_almost_equal(pred, pred_analog)

        gru_analog._apply_to_analog(lambda lay: lay._sync_weights_from_tile())

        gru_pars = get_parameters(gru, False)
        gru_analog_pars = get_parameters(gru_analog, True)

        if test_for_update:
            for par_name, par_item in gru_pars.items():
                par0 = gru_pars0[par_name].detach().cpu().numpy()
                par = par_item.detach().cpu().numpy()
                assert_raises(AssertionError, assert_array_almost_equal, par, par0)

        for par_name, par_item in gru_pars.items():
            assert_array_almost_equal(par_item.detach().cpu().numpy(),
                                      gru_analog_pars[par_name].detach().cpu().numpy())

@parametrize_over_layers(
    layers=[VanillaRNN, VanillaRNNCuda],
    tiles=[FloatingPoint, Inference],
    biases=['analog', 'digital', None]
)
class VanillaRNNLayerTest(ParametrizedTestCase):
    """Tests for AnalogVanillaRNN layer."""

    @staticmethod
    def train_once(model, y_in, y_out, analog_if, use_cuda=False):
        """Train once."""
        criterion = MSELoss()
        optimizer = AnalogSGD(model.parameters(), lr=0.5, momentum=0.0, nesterov=0.0)
        optimizer.regroup_param_groups(model)
        if analog_if:
            # why is this format so difference?
            # TODO: better use same state format as for native Pytorch's LSTM?
            if use_cuda:
                states = [zeros(y_in.size()[1], model.hidden_size).cuda()
                          for _ in range(model.num_layers)]
            else:
                states = [zeros(y_in.size()[1], model.hidden_size)
                          for _ in range(model.num_layers)]

        else:
            if use_cuda:
                states = zeros(model.num_layers, y_in.size()[1], model.hidden_size).cuda()
            else:
                states = zeros(model.num_layers, y_in.size()[1], model.hidden_size)

        for _ in range(2):
            optimizer.zero_grad()
            pred, _ = model(y_in, states)
            loss = criterion(pred.mean(axis=2, keepdim=True), y_out)
            loss.backward()
            optimizer.step()

        return pred.detach().cpu().numpy()

    @staticmethod
    def train_once_bidir(model, y_in, y_out, analog_if, use_cuda=False):
        """Train once."""
        criterion = MSELoss()
        optimizer = AnalogSGD(model.parameters(), lr=0.5, momentum=0.0, nesterov=0.0)
        optimizer.regroup_param_groups(model)
        if analog_if:
            # why is this format so difference?
            # TODO: better use same state format as for native Pytorch's LSTM?
            if use_cuda:
                states = [[zeros(y_in.size()[1], model.hidden_size).cuda(), zeros(y_in.size()[1], model.hidden_size).cuda()]
                          for _ in range(model.num_layers)]
            else:
                states = [[zeros(y_in.size()[1], model.hidden_size), zeros(y_in.size()[1], model.hidden_size)]
                          for _ in range(model.num_layers)] 

        else:
            if use_cuda:
                states = zeros(model.num_layers*2, y_in.size()[1], model.hidden_size).cuda()
            else:
                states = zeros(model.num_layers*2, y_in.size()[1], model.hidden_size)
        for _ in range(2):
            optimizer.zero_grad()
            pred, _ = model(y_in, states)
            loss = criterion(pred.mean(axis=2, keepdim=True), y_out)
            loss.backward()
            optimizer.step()

        return pred.detach().cpu().numpy()

    def test_layer_instantiation(self):
        """Test AnalogVanillaRNN layer instantiation."""
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
                # TODO: comparison of rpu_config not possible..
                self.assertEqual(layer.cell.weight_ih.analog_tile.rpu_config, self.get_rpu_config())
                self.assertEqual(layer.cell.weight_hh.analog_tile.rpu_config, self.get_rpu_config())
            else:
                self.assertEqual(layer.cell.weight_ih.analog_tile.rpu_config.__class__,
                                 self.get_rpu_config().__class__)
                self.assertEqual(layer.cell.weight_hh.analog_tile.rpu_config.__class__,
                                 self.get_rpu_config().__class__)

    def test_layer_training(self):
        """Test AnalogVanillaRNN layer training."""
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
        hidden_size = 3
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

        rnn = RNN_nn(input_size=input_size,
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
        pred_analog = self.train_once(rnn_analog, y_in, y_out, True, use_cuda=self.use_cuda)

        analog_weights = []
        rnn_analog._apply_to_analog(lambda lay: analog_weights.append(
            lay.analog_tile.tile.get_weights()))

        if test_for_update:
            for weight, weight_org in zip(analog_weights, weights_org):
                assert_raises(AssertionError, assert_array_almost_equal, weight, weight_org)

        # Compare with RNN.
        pred = self.train_once(rnn, y_in, y_out, False, use_cuda=self.use_cuda)
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
        """Test AnalogVanillaRNN bidirectional layer training."""
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
        hidden_size = 3
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

        rnn = RNN_nn(input_size=input_size,
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
        pred_analog = self.train_once_bidir(rnn_analog, y_in, y_out, True, use_cuda=self.use_cuda)

        analog_weights = []
        rnn_analog._apply_to_analog(lambda lay: analog_weights.append(
            lay.analog_tile.tile.get_weights()))

        if test_for_update:
            for weight, weight_org in zip(analog_weights, weights_org):
                assert_raises(AssertionError, assert_array_almost_equal, weight, weight_org)

        # Compare with RNN.
        pred = self.train_once_bidir(rnn, y_in, y_out, False, use_cuda=self.use_cuda)
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


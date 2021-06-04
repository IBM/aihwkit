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
from numpy.testing import assert_array_almost_equal, assert_raises

from aihwkit.nn.modules.lstm import LSTMState
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.configs import InferenceRPUConfig
from aihwkit.optim.context import AnalogContext

from .helpers.decorators import parametrize_over_layers
from .helpers.layers import LSTM, LSTMCuda
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import FloatingPoint, Inference


@parametrize_over_layers(
    layers=[LSTM, LSTMCuda],
    tiles=[FloatingPoint, Inference],
    biases=[False, True]
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

    def test_layer_instantiation(self):
        """Test AnalogLSTM layer instantiation."""
        input_size = 2
        hidden_size = 3
        num_layers = 4

        model = self.get_layer(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers)

        # Assert over the stacked layers.
        self.assertEqual(len(model.lstm.layers), num_layers)
        for i, layer in enumerate(model.lstm.layers):
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
        # pylint: disable=too-many-locals
        def get_parameters(model, analog_if) -> dict:
            """Returns the parameter in an dict."""
            if not analog_if:
                return dict(model.named_parameters())

            dic = {}
            for name, param in model.named_parameters():
                if isinstance(param, AnalogContext):
                    weight, bias = param.analog_tile.get_weights()
                    splits = name.split('.')
                    add_on = '_' + splits[-2].split('_')[-1] + '_l' + splits[2]

                    dic['weight' + add_on] = weight
                    if bias is not None:
                        dic['bias' + add_on] = bias

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

        for par_name in lstm_pars0.keys():
            lstm_pars0[par_name].data = lstm_analog_pars0[par_name].detach().clone()

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
            for par_name in lstm_pars.keys():
                par0 = lstm_pars0[par_name].detach().cpu().numpy()
                par = lstm_pars[par_name].detach().cpu().numpy()
                assert_raises(AssertionError, assert_array_almost_equal, par, par0)

        for par_name in lstm_pars.keys():
            assert_array_almost_equal(lstm_pars[par_name].detach().cpu().numpy(),
                                      lstm_analog_pars[par_name].detach().cpu().numpy())

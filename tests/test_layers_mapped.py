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

"""aihwkit example 1: simple network with one layer.
Simple network that consist of one analog layer. The network aims to learn
to sum all the elements from one array.
"""
# pylint: disable=invalid-name, too-many-locals

from tempfile import TemporaryFile

# Imports from PyTorch.
from torch import randn, load, save, manual_seed
from torch.nn.functional import mse_loss

# Imports from aihwkit.
from aihwkit.nn import AnalogLinearMapped
from aihwkit.optim import AnalogSGD

from .helpers.decorators import parametrize_over_layers
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import FloatingPoint, Inference, Ideal
from .helpers.layers import Linear, LinearCuda

DECIMAL = 5


@parametrize_over_layers(
    layers=[Linear, LinearCuda],
    tiles=[FloatingPoint, Inference, Ideal],
    biases=[True, False],
    digital_biases=[True])
class MappedLayerLinearTest(ParametrizedTestCase):
    """Tests for the AnalogMappedLayer functionality """

    def train_model(self, model, in_vectors, out_vectors):
        """Trains a model """

        opt = AnalogSGD(model.parameters(), lr=0.1)

        for _ in range(10):
            opt.zero_grad()

            # Add the training Tensor to the model (input).
            pred_value = model(in_vectors)
            # Add the expected output Tensor.
            loss_value = mse_loss(pred_value, out_vectors)
            # Run training (backward propagation).
            loss_value.backward()
            opt.step()

        return loss_value

    def test_construction(self):
        """ Test construction of a mapped layer"""
        manual_seed(123)
        in_features = 12
        out_features = 56

        rpu_config = self.get_rpu_config()
        rpu_config.mapping.max_input_size = 6
        rpu_config.mapping.max_output_size = 12

        al_model = self.get_layer(in_features, out_features, rpu_config=rpu_config)

        weight = randn(out_features, in_features)
        bias = randn(out_features) if self.bias else None
        al_model.set_weights(weight, bias)
        al_weights, al_bias = al_model.get_weights()

        asl_model = AnalogLinearMapped(in_features, out_features,
                                       bias=self.bias,
                                       rpu_config=rpu_config)
        asl_model.set_weights(weight, bias)

        if self.use_cuda:
            asl_model = asl_model.cuda()

        asl_weights, asl_bias = asl_model.get_weights()

        self.assertTensorAlmostEqual(al_weights, asl_weights, decimal=DECIMAL)
        if self.bias:
            self.assertTensorAlmostEqual(al_bias, asl_bias, decimal=DECIMAL)

    def test_training(self):
        """ Test of training of a mapped linear layer"""
        manual_seed(123)

        in_features = 16
        out_features = 15
        batch_size = 10

        rpu_config = self.get_rpu_config()
        rpu_config.mapping.max_input_size = 4
        rpu_config.mapping.max_output_size = 3

        in_vectors = randn(batch_size, in_features)
        out_vectors = randn(batch_size, out_features)

        al_model = self.get_layer(in_features, out_features, rpu_config=rpu_config)

        weight = randn(out_features, in_features)
        bias = randn(out_features) if self.bias else None

        al_model.set_weights(weight, bias)

        asl_model = AnalogLinearMapped(in_features, out_features,
                                       bias=self.bias,
                                       rpu_config=rpu_config)
        asl_model.set_weights(weight, bias)

        if self.use_cuda:
            in_vectors = in_vectors.cuda()
            out_vectors = out_vectors.cuda()
            asl_model = asl_model.cuda()

        # compare predictions for analog linear and analog spit linear layers
        self.assertTensorAlmostEqual(al_model(in_vectors), asl_model(in_vectors), decimal=DECIMAL)

        # Define an analog-aware optimizer, preparing it for using the layers.
        al_loss = self.train_model(al_model, in_vectors, out_vectors)
        asl_loss = self.train_model(asl_model, in_vectors, out_vectors)

        self.assertTensorAlmostEqual(al_loss, asl_loss, decimal=DECIMAL)

        # Make sure that the train model produces the same forward pass
        self.assertTensorAlmostEqual(al_model(in_vectors), asl_model(in_vectors), decimal=DECIMAL)

    def test_training_after_save(self):
        """ Test training after it was saved """
        manual_seed(123)

        in_features = 7
        out_features = 8
        batch_size = 10

        rpu_config = self.get_rpu_config()
        rpu_config.mapping.max_input_size = 3
        rpu_config.mapping.max_output_size = 4

        in_vectors = randn(batch_size, in_features)
        out_vectors = randn(batch_size, out_features)

        al_model = self.get_layer(in_features, out_features, rpu_config=rpu_config)

        weight = randn(out_features, in_features)
        bias = randn(out_features) if self.bias else None

        al_model.set_weights(weight, bias)

        asl_model = AnalogLinearMapped(in_features, out_features,
                                       bias=self.bias,
                                       rpu_config=rpu_config)
        asl_model.set_weights(weight, bias)

        if self.use_cuda:
            in_vectors = in_vectors.cuda()
            out_vectors = out_vectors.cuda()
            asl_model = asl_model.cuda()
            al_model = al_model.cuda()

        # Save the model to a file.
        with TemporaryFile() as file:
            save(asl_model.state_dict(), file)
            # Create a new model and load its state dict.
            file.seek(0)
            new_model = AnalogLinearMapped(in_features, out_features,
                                           bias=self.bias,
                                           rpu_config=rpu_config)
            if self.use_cuda:
                new_model = new_model.cuda()
            new_model.load_state_dict(load(file))

        new_weight, new_bias = new_model.get_weights()

        self.assertTensorAlmostEqual(new_weight, weight, decimal=DECIMAL)
        if self.bias:
            self.assertTensorAlmostEqual(new_bias, bias, decimal=DECIMAL)
            self.assertTensorAlmostEqual(al_model.bias, bias, decimal=DECIMAL)
            self.assertTensorAlmostEqual(asl_model.bias, bias, decimal=DECIMAL)

        for new_tile, tile in zip(list(new_model.analog_tiles()), list(asl_model.analog_tiles())):
            new_tile_weight, _ = new_tile.get_weights()
            tile_weight, _ = tile.get_weights()
            self.assertTensorAlmostEqual(tile_weight, new_tile_weight, decimal=DECIMAL)

        # compare predictions for analog linear and analog spit linear layers
        self.assertTensorAlmostEqual(al_model(in_vectors), new_model(in_vectors), decimal=DECIMAL)

        # Define an analog-aware optimizer, preparing it for using the layers.
        al_loss = self.train_model(al_model, in_vectors, out_vectors)
        new_loss = self.train_model(new_model, in_vectors, out_vectors)

        # compare predictions for analog linear and analog spit linear layers
        self.assertTensorAlmostEqual(al_model(in_vectors), new_model(in_vectors), decimal=DECIMAL)

        self.assertTensorAlmostEqual(al_loss, new_loss, decimal=DECIMAL)

        # Make sure that the train model produces the same forward pass
        self.assertTensorAlmostEqual(al_model(in_vectors), new_model(in_vectors), decimal=DECIMAL)

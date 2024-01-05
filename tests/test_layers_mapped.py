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

"""aihwkit example 1: simple network with one layer.
Simple network that consist of one analog layer. The network aims to learn
to sum all the elements from one array.
"""
# pylint: disable=invalid-name, too-many-locals

from tempfile import TemporaryFile
from numpy import array

# Imports from PyTorch.
from torch import randn, load, save, manual_seed
from torch.nn.functional import mse_loss

# Imports from aihwkit.
from aihwkit.nn import (
    AnalogLinearMapped,
    AnalogConv1dMapped,
    AnalogConv2dMapped,
    AnalogConv3dMapped,
)
from aihwkit.optim import AnalogSGD

from .helpers.decorators import parametrize_over_layers
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import FloatingPoint, Inference, Ideal
from .helpers.layers import Linear, LinearCuda, Conv1d, Conv1dCuda, Conv2d, Conv2dCuda, Conv3dCuda

DECIMAL = 4


@parametrize_over_layers(
    layers=[Linear, LinearCuda, Conv1d, Conv2d, Conv1dCuda, Conv2dCuda, Conv3dCuda],
    tiles=[FloatingPoint, Inference, Ideal],
    biases=["digital", None],
)
class MappedLayerLinearTest(ParametrizedTestCase):
    """Tests for the AnalogMappedLayer functionality"""

    def get_mapped_class(self, model):
        """Returns the mapped class"""
        name = model.__class__.__name__
        if name == "AnalogLinear":
            return AnalogLinearMapped
        if name == "AnalogConv1d":
            return AnalogConv1dMapped
        if name == "AnalogConv2d":
            return AnalogConv2dMapped
        if name == "AnalogConv3d":
            return AnalogConv3dMapped

        raise RuntimeError("Cannot find mapped module.")

    def get_mapped_model(self, model, rpu_config):
        """Returns the mapped model"""
        weight, bias = model.weight, model.bias
        model.weight, model.bias = model.get_weights()
        mapped_model = self.get_mapped_class(model).from_digital(model, rpu_config=rpu_config)
        mapped_model.reset_parameters()  # should set it explicitly below
        model.weight, model.bias = weight, bias
        return mapped_model

    def get_image_size(self, model):
        """Returns the image size"""
        if not hasattr(model, "kernel_size"):
            return []
        return (array(model.kernel_size) * 2).tolist()

    def train_model(self, model, in_vectors, out_vectors):
        """Trains a model"""

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
        """Test construction of a mapped layer"""
        manual_seed(123)
        in_features = 14
        out_features = 5

        rpu_config = self.get_rpu_config()
        rpu_config.mapping.max_input_size = 10
        rpu_config.mapping.max_output_size = 4

        model = self.get_layer(in_features, out_features, rpu_config=rpu_config)
        weight, bias = model.get_weights()

        weight = randn(*weight.shape)
        if self.bias:
            bias = randn(*bias.shape)
        model.set_weights(weight, bias)

        mapped_model = self.get_mapped_model(model, rpu_config)
        mapped_model.set_weights(weight, bias)

        if self.use_cuda:
            mapped_model = mapped_model.cuda()

        mapped_weights, mapped_bias = mapped_model.get_weights()

        self.assertTensorAlmostEqual(weight, mapped_weights, decimal=DECIMAL)
        if self.bias:
            self.assertTensorAlmostEqual(bias, mapped_bias, decimal=DECIMAL)

    def test_training(self):
        """Test of training of a mapped linear layer"""
        manual_seed(123)

        in_features = 12
        out_features = 13
        batch_size = 10

        rpu_config = self.get_rpu_config()

        model = self.get_layer(in_features, out_features, rpu_config=rpu_config)
        weight, bias = model.get_weights()

        weight = randn(*weight.shape)
        if self.bias:
            bias = randn(*bias.shape)
        model.set_weights(weight, bias)

        rpu_config.mapping.max_input_size = 10
        rpu_config.mapping.max_output_size = 6

        mapped_model = self.get_mapped_model(model, rpu_config)
        mapped_model.set_weights(weight, bias)

        in_vectors = randn(*([batch_size, in_features] + self.get_image_size(model)))

        if self.use_cuda:
            in_vectors = in_vectors.cuda()
            mapped_model = mapped_model.cuda()

        out_vectors = randn(*model(in_vectors).shape)
        if self.use_cuda:
            out_vectors = out_vectors.cuda()

        # compare predictions for analog linear and analog split linear layers
        self.assertTensorAlmostEqual(model(in_vectors), mapped_model(in_vectors), decimal=DECIMAL)

        # Define an analog-aware optimizer, preparing it for using the layers.
        loss = self.train_model(model, in_vectors, out_vectors)
        mapped_loss = self.train_model(mapped_model, in_vectors, out_vectors)

        self.assertTensorAlmostEqual(loss, mapped_loss, decimal=DECIMAL)

        # Make sure that the train model produces the same forward pass
        self.assertTensorAlmostEqual(model(in_vectors), mapped_model(in_vectors), decimal=DECIMAL)

    def test_training_after_save(self):
        """Test training after it was saved"""
        manual_seed(123)

        in_features = 11
        out_features = 11
        batch_size = 10

        rpu_config = self.get_rpu_config()

        model = self.get_layer(in_features, out_features, rpu_config=rpu_config)
        weight, bias = model.get_weights()

        weight = randn(*weight.shape)
        if self.bias:
            bias = randn(*bias.shape)
        model.set_weights(weight, bias)

        rpu_config.mapping.max_input_size = 10
        rpu_config.mapping.max_output_size = 4

        mapped_model = self.get_mapped_model(model, rpu_config)
        mapped_model.set_weights(weight, bias)

        in_vectors = randn(*([batch_size, in_features] + self.get_image_size(model)))
        if self.use_cuda:
            in_vectors = in_vectors.cuda()
            mapped_model = mapped_model.cuda()

        out_vectors = randn(*model(in_vectors).shape)
        if self.use_cuda:
            out_vectors = out_vectors.cuda()

        # Save the model to a file.
        with TemporaryFile() as file:
            save(mapped_model.state_dict(), file)
            # Create a new model and load its state dict.
            file.seek(0)
            new_model = self.get_mapped_model(model, rpu_config)
            if self.use_cuda:
                new_model = new_model.cuda()
            new_model.load_state_dict(load(file))

        new_weight, new_bias = new_model.get_weights()

        self.assertTensorAlmostEqual(new_weight, weight, decimal=DECIMAL)
        if self.bias:
            self.assertTensorAlmostEqual(new_bias, bias, decimal=DECIMAL)

        for new_tile, tile in zip(
            list(new_model.analog_tiles()), list(mapped_model.analog_tiles())
        ):
            new_tile_weight, _ = new_tile.get_weights()
            tile_weight, _ = tile.get_weights()
            self.assertTensorAlmostEqual(tile_weight, new_tile_weight, decimal=DECIMAL)

        # compare predictions for analog linear and analog spit linear layers
        self.assertTensorAlmostEqual(model(in_vectors), new_model(in_vectors), decimal=DECIMAL)

        # Define an analog-aware optimizer, preparing it for using the layers.
        loss = self.train_model(model, in_vectors, out_vectors)
        new_loss = self.train_model(new_model, in_vectors, out_vectors)

        # compare predictions for analog linear and analog spit linear layers
        self.assertTensorAlmostEqual(model(in_vectors), new_model(in_vectors), decimal=DECIMAL)

        self.assertTensorAlmostEqual(loss, new_loss, decimal=DECIMAL)

        # Make sure that the train model produces the same forward pass
        self.assertTensorAlmostEqual(model(in_vectors), new_model(in_vectors), decimal=DECIMAL)

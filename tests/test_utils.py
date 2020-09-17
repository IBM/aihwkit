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

"""Test for different utility functionality."""

from tempfile import TemporaryFile
from unittest import TestCase, skipIf

from numpy.random import rand
from numpy.testing import assert_array_almost_equal, assert_raises
from torch import Tensor, save, load
from torch.nn.functional import mse_loss

from aihwkit.nn.modules.conv import AnalogConv2d
from aihwkit.nn.modules.linear import AnalogLinear
from aihwkit.optim.analog_sgd import AnalogSGD
from aihwkit.simulator.devices import (
    ConstantStepResistiveDevice,
    FloatingPointResistiveDevice
)
from aihwkit.simulator.parameters import (
    AnalogTileBackwardInputOutputParameters,
    AnalogTileInputOutputParameters,
    AnalogTileUpdateParameters,
    ConstantStepResistiveDeviceParameters
)
from aihwkit.simulator.rpu_base import cuda


class SerializationTestMixin:
    """Helper for tests for serialization."""

    USE_CUDA = False

    @staticmethod
    def train_model(model, loss_func, x_b, y_b):
        """Train the model."""
        opt = AnalogSGD(model.parameters(), lr=0.5)
        opt.regroup_param_groups(model)

        epochs = 1
        for _ in range(epochs):
            pred = model(x_b)
            loss = loss_func(pred, y_b)

            loss.backward()
            opt.step()
            opt.zero_grad()

    @staticmethod
    def get_model_and_tile_weights(model):
        """Return the weights and biases of the model and the tile."""
        weight = model.weight.data.detach().cpu().numpy()
        bias = model.bias.data.detach().cpu().numpy()
        analog_weight, analog_bias = model.analog_tile.get_weights()
        analog_weight = analog_weight.detach().cpu().numpy().reshape(weight.shape)
        analog_bias = analog_bias.detach().cpu().numpy()
        return weight, bias, analog_weight, analog_bias

    def get_model(self, **kwargs):
        """Return a layer."""
        raise NotImplementedError

    def test_save_load_state_dict_train(self):
        """Test saving and loading using a state dict after training."""
        model = self.get_model()

        # Perform an update in order to modify tile weights and biases.
        loss_func = mse_loss
        if isinstance(model, AnalogConv2d):
            input_x = Tensor(rand(2, 2, 3, 3))*0.2
            input_y = Tensor(rand(2, 3, 4, 4))*0.2
        else:
            input_x = Tensor(rand(2, model.in_features))*0.2
            input_y = Tensor(rand(2, model.out_features))*0.2

        if self.USE_CUDA:
            input_x = input_x.cuda()
            input_y = input_y.cuda()

        self.train_model(model, loss_func, input_x, input_y)

        # Keep track of the current weights and biases for comparing.
        (model_weights, model_biases,
         tile_weights, tile_biases) = self.get_model_and_tile_weights(model)

        # now the tile weights should be out of sync
        assert_raises(AssertionError, assert_array_almost_equal, model_weights, tile_weights)
        assert_raises(AssertionError, assert_array_almost_equal, model_biases, tile_biases)

        # Save the model to a file.
        file = TemporaryFile()
        save(model.state_dict(), file)

        # Create a new model and load its state dict.
        file.seek(0)
        new_model = self.get_model()
        new_model.load_state_dict(load(file))
        file.close()

        # Compare the new model weights and biases. they should now be in sync
        (new_model_weights, new_model_biases,
         new_tile_weights, new_tile_biases) = self.get_model_and_tile_weights(new_model)

        assert_array_almost_equal(tile_weights, new_model_weights)
        assert_array_almost_equal(tile_biases, new_model_biases)
        assert_array_almost_equal(tile_weights, new_tile_weights)
        assert_array_almost_equal(tile_biases, new_tile_biases)

    def test_save_load_model(self):
        """Test saving and loading a model directly."""
        model = self.get_model()

        # Keep track of the current weights and biases for comparing.
        (model_weights, model_biases,
         tile_weights, tile_biases) = self.get_model_and_tile_weights(model)
        assert_array_almost_equal(model_weights, tile_weights)
        assert_array_almost_equal(model_biases, tile_biases)

        # Save the model to a file.
        file = TemporaryFile()
        save(model, file)

        # Load the model.
        file.seek(0)
        new_model = load(file)
        file.close()

        # Compare the new model weights and biases.
        (new_model_weights, new_model_biases,
         new_tile_weights, new_tile_biases) = self.get_model_and_tile_weights(new_model)

        assert_array_almost_equal(model_weights, new_model_weights)
        assert_array_almost_equal(model_biases, new_model_biases)
        assert_array_almost_equal(tile_weights, new_tile_weights)
        assert_array_almost_equal(tile_biases, new_tile_biases)

    def test_save_load_meta_parameter(self):
        """Test saving and loading a device with custom parameters."""
        params_devices = ConstantStepResistiveDeviceParameters(w_max=0.987)
        params_forward = AnalogTileInputOutputParameters(inp_noise=0.321)
        params_backward = AnalogTileBackwardInputOutputParameters(inp_noise=0.456)
        params_update = AnalogTileUpdateParameters(desired_bl=78)

        # Create the device and the array.
        resistive_device = ConstantStepResistiveDevice(
            params_devices, params_forward, params_backward, params_update)

        model = self.get_model(resistive_device=resistive_device)

        # Save the model to a file.
        file = TemporaryFile()
        save(model, file)

        # Load the model.
        file.seek(0)
        new_model = load(file)
        file.close()

        # Assert over the new model tile parameters.
        parameters = new_model.analog_tile.tile.get_parameters()
        self.assertAlmostEqual(parameters.forward_io.inp_noise, 0.321)
        self.assertAlmostEqual(parameters.backward_io.inp_noise, 0.456)
        self.assertAlmostEqual(parameters.update.desired_bl, 78)

    def test_save_load_hidden_parameters(self):
        """Test saving and loading a device with hidden parameters."""
        # Create the device and the array.
        model = self.get_model()
        hidden_parameters = model.analog_tile.tile.get_hidden_parameters()

        # Save the model to a file.
        file = TemporaryFile()
        save(model, file)

        # Load the model.
        file.seek(0)
        new_model = load(file)
        file.close()

        # Assert over the new model tile parameters.
        new_hidden_parameters = new_model.analog_tile.tile.get_hidden_parameters()
        assert_array_almost_equal(hidden_parameters, new_hidden_parameters)

    def test_save_load_state_dict_hidden_parameters(self):
        """Test saving and loading via state_dict with hidden parameters."""
        # Create the device and the array.
        model = self.get_model()
        hidden_parameters = model.analog_tile.tile.get_hidden_parameters()

        # Save the model to a file.
        file = TemporaryFile()
        save(model.state_dict(), file)

        # Load the model.
        file.seek(0)
        new_model = self.get_model()
        new_model.load_state_dict(load(file))
        file.close()

        # Assert over the new model tile parameters.
        new_hidden_parameters = new_model.analog_tile.tile.get_hidden_parameters()
        assert_array_almost_equal(hidden_parameters, new_hidden_parameters)


class LinearFloatingPointSerializationTest(SerializationTestMixin, TestCase):
    """Test AnalogLinear serialization (floating point)."""

    def get_model(self, **kwargs):
        """Return a layer."""
        if 'resistive_device' not in kwargs:
            kwargs['resistive_device'] = FloatingPointResistiveDevice()
        return AnalogLinear(in_features=5, out_features=3, **kwargs)


class LinearConstantStepSerializationTest(SerializationTestMixin, TestCase):
    """Test AnalogLinear serialization (constant step)."""

    def get_model(self, **kwargs):
        """Return a layer."""
        if 'resistive_device' not in kwargs:
            kwargs['resistive_device'] = ConstantStepResistiveDevice()
        return AnalogLinear(in_features=5, out_features=3, **kwargs)


class Conv2dFloatingPointSerializationTest(SerializationTestMixin, TestCase):
    """Test AnalogConv2d serialization (floating point)."""

    def get_model(self, **kwargs):
        """Return a layer."""
        if 'resistive_device' not in kwargs:
            kwargs['resistive_device'] = FloatingPointResistiveDevice()
        return AnalogConv2d(in_channels=2, out_channels=3, kernel_size=4,
                            padding=2, **kwargs)


class Conv2dConstantStepSerializationTest(SerializationTestMixin, TestCase):
    """Test AnalogConv2d serialization (constant step)."""

    def get_model(self, **kwargs):
        """Return a layer."""
        if 'resistive_device' not in kwargs:
            kwargs['resistive_device'] = ConstantStepResistiveDevice()
        return AnalogConv2d(in_channels=2, out_channels=3, kernel_size=4,
                            padding=2, **kwargs)


@skipIf(not cuda.is_compiled(), 'not compiled with CUDA support')
class CudaLinearFloatingPointSerializationTest(SerializationTestMixin, TestCase):
    """Test AnalogLinear serialization (floating point, cuda)."""

    USE_CUDA = True

    def get_model(self, **kwargs):
        """Return a layer."""
        if 'resistive_device' not in kwargs:
            kwargs['resistive_device'] = FloatingPointResistiveDevice()
        return AnalogLinear(in_features=5, out_features=3, **kwargs).cuda()


@skipIf(not cuda.is_compiled(), 'not compiled with CUDA support')
class CudaLinearConstantStepSerializationTest(SerializationTestMixin, TestCase):
    """Test AnalogLinear serialization (constant step, cuda)."""

    USE_CUDA = True

    def get_model(self, **kwargs):
        """Return a layer."""
        if 'resistive_device' not in kwargs:
            kwargs['resistive_device'] = ConstantStepResistiveDevice()
        return AnalogLinear(in_features=5, out_features=3, **kwargs).cuda()


@skipIf(not cuda.is_compiled(), 'not compiled with CUDA support')
class CudaConv2dFloatingPointSerializationTest(SerializationTestMixin, TestCase):
    """Test AnalogConv2d serialization (floating point, cuda)."""

    USE_CUDA = True

    def get_model(self, **kwargs):
        """Return a layer."""
        if 'resistive_device' not in kwargs:
            kwargs['resistive_device'] = FloatingPointResistiveDevice()
        return AnalogConv2d(in_channels=2, out_channels=3, kernel_size=4,
                            padding=2, **kwargs).cuda()


@skipIf(not cuda.is_compiled(), 'not compiled with CUDA support')
class CudaConv2dConstantStepSerializationTest(SerializationTestMixin, TestCase):
    """Test AnalogConv2d serialization (constant step)."""

    USE_CUDA = True

    def get_model(self, **kwargs):
        """Return a layer."""
        if 'resistive_device' not in kwargs:
            kwargs['resistive_device'] = ConstantStepResistiveDevice()
        return AnalogConv2d(in_channels=2, out_channels=3, kernel_size=4,
                            padding=2, **kwargs).cuda()

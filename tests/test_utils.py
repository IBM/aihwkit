# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=too-many-locals, too-many-public-methods, no-member
"""Test for different utility functionality."""

from tempfile import TemporaryFile
from copy import deepcopy
from unittest import SkipTest

from numpy import array
from numpy.random import rand
from numpy.testing import assert_array_almost_equal, assert_raises
from torch import Tensor, load, save, device, manual_seed
from torch import abs as torch_abs
from torch.nn import Module, Sequential
from torch.nn import Linear as torch_linear
from torch.nn.functional import mse_loss
from torch.optim import SGD

from aihwkit.nn import AnalogConv2d, AnalogConv2dMapped, AnalogSequential, AnalogLinearMapped
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, InferenceRPUConfig, FloatingPointRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice, LinearStepDevice
from aihwkit.simulator.parameters import IOParameters, UpdateParameters, MappingParameter
from aihwkit.simulator.tiles.base import AnalogTileStateNames
from aihwkit.exceptions import TileError, TileModuleError
from aihwkit.nn.conversion import convert_to_analog

from .helpers.decorators import parametrize_over_layers
from .helpers.layers import (
    Conv2d,
    Conv2dCuda,
    Linear,
    LinearCuda,
    LinearMapped,
    LinearMappedCuda,
    Conv2dMapped,
    Conv2dMappedCuda,
)
from .helpers.testcases import ParametrizedTestCase, SKIP_CUDA_TESTS
from .helpers.tiles import (
    FloatingPoint,
    ConstantStep,
    Inference,
    InferenceLearnOutScaling,
    TorchInference,
    TorchInferenceIRDropT,
    Custom,
    TorchTransfer,
)

SKIP_META_PARAM_TILES = [TorchInference, TorchInferenceIRDropT, Custom, FloatingPoint]


@parametrize_over_layers(
    layers=[
        Linear,
        Conv2d,
        LinearMapped,
        LinearCuda,
        LinearMappedCuda,
        Conv2dCuda,
        Conv2dMapped,
        Conv2dMappedCuda,
    ],
    tiles=[
        FloatingPoint,
        ConstantStep,
        Inference,
        InferenceLearnOutScaling,
        TorchInference,
        TorchInferenceIRDropT,
        TorchTransfer,
        Custom,
    ],
    biases=["digital"],
)
class SerializationTest(ParametrizedTestCase):
    """Tests for serialization."""

    @staticmethod
    def train_model(model, loss_func, x_b, y_b):
        """Train the model."""
        opt = AnalogSGD(model.parameters(), lr=0.5)
        # opt.regroup_param_groups(model)
        epochs = 3
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(x_b)

            loss = loss_func(pred, y_b)
            loss.backward()
            opt.step()
        return loss

    @staticmethod
    def get_layer_and_tile_weights(model):
        """Return the weights and biases of the model and the tile and whether
        it automatically syncs"""

        if isinstance(model, AnalogLinearMapped):
            weight, bias = model.get_weights()
            return weight, bias, weight, bias, True

        if isinstance(model, AnalogConv2dMapped):
            weight, bias = model.get_weights()
            return weight, bias, weight, bias, True

        if model.weight is not None:
            weight = model.weight.data.detach().cpu().numpy()
        else:
            # we do not sync anymore
            weight, bias = model.get_weights()
            return weight, bias, weight, bias, True

        if model.bias is not None:
            bias = model.bias.data.detach().cpu().numpy()
        else:
            bias = None

        analog_weight, analog_bias = model.get_weights()
        analog_weight = analog_weight.detach().cpu().numpy()
        if analog_bias is not None:
            analog_bias = analog_bias.detach().cpu().numpy()

        return weight, bias, analog_weight.reshape(weight.shape), analog_bias, True

    @staticmethod
    def get_analog_tile(model):
        """Return a (python) analog tile of the model"""
        return list(model.analog_tiles())[0]

    def test_save_load_state_dict_train(self):
        """Test saving and loading using a state dict after training."""
        model = self.get_layer()

        # Perform an update in order to modify tile weights and biases.
        loss_func = mse_loss
        if isinstance(model, (AnalogConv2d, AnalogConv2dMapped)):
            input_x = Tensor(rand(2, 2, 3, 3)) * 0.2
            input_y = Tensor(rand(2, 3, 5, 5)) * 0.2
        else:
            input_x = Tensor(rand(2, model.in_features)) * 0.2
            input_y = Tensor(rand(2, model.out_features)) * 0.2

        if self.use_cuda:
            input_x = input_x.cuda()
            input_y = input_y.cuda()

        self.train_model(model, loss_func, input_x, input_y)

        # Keep track of the current weights and biases for comparing.
        (
            model_weights,
            model_biases,
            tile_weights,
            tile_biases,
            sync,
        ) = self.get_layer_and_tile_weights(model)

        # now the tile weights should be out of sync
        if not sync:
            assert_raises(AssertionError, assert_array_almost_equal, model_weights, tile_weights)
            if self.bias:
                assert_raises(AssertionError, assert_array_almost_equal, model_biases, tile_biases)

        # Save the model to a file.
        with TemporaryFile() as file:
            save(model.state_dict(), file)
            # Create a new model and load its state dict.
            file.seek(0)
            new_model = self.get_layer()
            new_model.load_state_dict(load(file, weights_only=False))

        # Compare the new model weights and biases. they should now be in sync
        (
            new_model_weights,
            new_model_biases,
            new_tile_weights,
            new_tile_biases,
            _,
        ) = self.get_layer_and_tile_weights(new_model)

        assert_array_almost_equal(tile_weights, new_model_weights)
        assert_array_almost_equal(tile_weights, new_tile_weights)
        if self.bias:
            assert_array_almost_equal(tile_biases, new_model_biases)
            assert_array_almost_equal(tile_biases, new_tile_biases)

    def test_save_load_state_dict_train_after(self):
        """Test saving and loading using a state dict and training after load."""
        model = self.get_layer()

        # Perform an update in order to modify tile weights and biases.
        loss_func = mse_loss
        if isinstance(model, (AnalogConv2d, AnalogConv2dMapped)):
            input_x = Tensor(rand(2, 2, 3, 3)) * 0.2
            input_y = Tensor(rand(2, 3, 5, 5)) * 0.2
        else:
            input_x = Tensor(rand(2, model.in_features)) * 0.2
            input_y = Tensor(rand(2, model.out_features)) * 0.2

        if self.use_cuda:
            input_x = input_x.cuda()
            input_y = input_y.cuda()

        self.train_model(model, loss_func, input_x, input_y)

        # Keep track of the current weights and biases for comparing.
        _, _, tile_weights, tile_biases, _ = self.get_layer_and_tile_weights(model)

        # Save the model to a file.
        with TemporaryFile() as file:
            save(model.state_dict(), file)
            # Create a new model and load its state dict.
            file.seek(0)
            new_model = self.get_layer()
            loaded = load(file, weights_only=False)
            new_model.load_state_dict(loaded)

        # Compare the new model weights and biases. they should now be in sync
        (
            new_model_weights,
            new_model_biases,
            new_tile_weights,
            new_tile_biases,
            _,
        ) = self.get_layer_and_tile_weights(new_model)

        assert_array_almost_equal(tile_weights, new_model_weights)
        assert_array_almost_equal(tile_weights, new_tile_weights)
        if self.bias:
            assert_array_almost_equal(tile_biases, new_model_biases)
            assert_array_almost_equal(tile_biases, new_tile_biases)

        new_loss = self.train_model(new_model, loss_func, input_x, input_y)

        loss = self.train_model(model, loss_func, input_x, input_y)
        if self.tile_class not in (ConstantStep, TorchTransfer):
            self.assertTensorAlmostEqual(loss, new_loss)

    def test_save_load_state_dict_train_after_old_model(self):
        """Test saving and loading using a state dict and training after load with old model."""
        model = self.get_layer()

        # Perform an update in order to modify tile weights and biases.
        loss_func = mse_loss
        if isinstance(model, (AnalogConv2d, AnalogConv2dMapped)):
            input_x = Tensor(rand(2, 2, 3, 3)) * 0.2
            input_y = Tensor(rand(2, 3, 5, 5)) * 0.2
        else:
            input_x = Tensor(rand(2, model.in_features)) * 0.2
            input_y = Tensor(rand(2, model.out_features)) * 0.2

        if self.use_cuda:
            input_x = input_x.cuda()
            input_y = input_y.cuda()

        self.train_model(model, loss_func, input_x, input_y)

        # Keep track of the current weights and biases for comparing.
        _, _, tile_weights, tile_biases, _ = self.get_layer_and_tile_weights(model)

        # Save the model to a file.
        with TemporaryFile() as file:
            save(model.state_dict(), file)
            # Create a new model and load its state dict.
            file.seek(0)
            model.load_state_dict(load(file, weights_only=False))

        # Compare the new model weights and biases. they should now be in sync
        (
            new_model_weights,
            new_model_biases,
            new_tile_weights,
            new_tile_biases,
            _,
        ) = self.get_layer_and_tile_weights(model)

        assert_array_almost_equal(tile_weights, new_model_weights)
        assert_array_almost_equal(tile_weights, new_tile_weights)
        if self.bias:
            assert_array_almost_equal(tile_biases, new_model_biases)
            assert_array_almost_equal(tile_biases, new_tile_biases)

        self.train_model(model, loss_func, input_x, input_y)

    def test_save_load_train_after(self):
        """Test saving and loading using a state dict and training after load."""
        model = self.get_layer()

        # Perform an update in order to modify tile weights and biases.
        loss_func = mse_loss
        if isinstance(model, (AnalogConv2d, AnalogConv2dMapped)):
            input_x = Tensor(rand(2, 2, 3, 3)) * 0.2
            input_y = Tensor(rand(2, 3, 5, 5)) * 0.2
        else:
            input_x = Tensor(rand(2, model.in_features)) * 0.2
            input_y = Tensor(rand(2, model.out_features)) * 0.2

        if self.use_cuda:
            input_x = input_x.cuda()
            input_y = input_y.cuda()

        # Keep track of the current weights and biases for comparing.
        _, _, tile_weights, tile_biases, _ = self.get_layer_and_tile_weights(model)

        # Save the model to a file.
        with TemporaryFile() as file:
            save(model, file)
            # Create a new model and load its state dict.
            file.seek(0)
            new_model = load(file, weights_only=False)

        # Compare the new model weights and biases. they should now be in sync
        (
            new_model_weights,
            new_model_biases,
            new_tile_weights,
            new_tile_biases,
            _,
        ) = self.get_layer_and_tile_weights(new_model)

        assert_array_almost_equal(tile_weights, new_model_weights)
        assert_array_almost_equal(tile_weights, new_tile_weights)
        if self.bias:
            assert_array_almost_equal(tile_biases, new_model_biases)
            assert_array_almost_equal(tile_biases, new_tile_biases)

        new_loss = self.train_model(new_model, loss_func, input_x, input_y)
        loss = self.train_model(model, loss_func, input_x, input_y)
        if self.tile_class not in (ConstantStep, TorchTransfer):
            self.assertTensorAlmostEqual(loss, new_loss)

    def test_save_load_model(self):
        """Test saving and loading a model directly."""
        model = self.get_layer()

        # Keep track of the current weights and biases for comparing.
        (
            model_weights,
            model_biases,
            tile_weights,
            tile_biases,
            _,
        ) = self.get_layer_and_tile_weights(model)
        assert_array_almost_equal(model_weights, tile_weights)
        if self.bias:
            assert_array_almost_equal(model_biases, tile_biases)

        # Save the model to a file.
        with TemporaryFile() as file:
            save(model, file)
            # Load the model.
            file.seek(0)
            new_model = load(file, weights_only=False)

        # Compare the new model weights and biases.
        (
            new_model_weights,
            new_model_biases,
            new_tile_weights,
            new_tile_biases,
            _,
        ) = self.get_layer_and_tile_weights(new_model)

        assert_array_almost_equal(model_weights, new_model_weights)
        assert_array_almost_equal(tile_weights, new_tile_weights)
        if self.bias:
            assert_array_almost_equal(model_biases, new_model_biases)
            assert_array_almost_equal(tile_biases, new_tile_biases)

        # Asserts over the AnalogContext of the new model.
        new_analog_tile = self.get_analog_tile(new_model)
        analog_tile = self.get_analog_tile(model)

        self.assertTrue(hasattr(new_analog_tile.analog_ctx, "analog_tile"))
        self.assertIsInstance(new_analog_tile.analog_ctx.analog_tile, analog_tile.__class__)
        self.assertTrue(new_analog_tile.is_cuda == analog_tile.is_cuda)

    def test_save_load_model_cross_device(self):
        """Test saving and loading a model directly."""

        if SKIP_CUDA_TESTS:
            raise SkipTest("CUDA not available.")

        model = self.get_layer()

        map_location = "cuda"
        if self.use_cuda:
            map_location = "cpu"

        # Keep track of the current weights and biases for comparing.
        (
            model_weights,
            model_biases,
            tile_weights,
            tile_biases,
            _,
        ) = self.get_layer_and_tile_weights(model)
        assert_array_almost_equal(model_weights, tile_weights)
        if self.bias:
            assert_array_almost_equal(model_biases, tile_biases)

        # Save the model to a file.
        with TemporaryFile() as file:
            save(model, file)
            # Load the model.
            file.seek(0)
            new_model = load(file, map_location=device(map_location), weights_only=False)

        # Compare the new model weights and biases.
        (
            new_model_weights,
            new_model_biases,
            new_tile_weights,
            new_tile_biases,
            _,
        ) = self.get_layer_and_tile_weights(new_model)

        assert_array_almost_equal(model_weights, new_model_weights)
        assert_array_almost_equal(tile_weights, new_tile_weights)
        if self.bias:
            assert_array_almost_equal(model_biases, new_model_biases)
            assert_array_almost_equal(tile_biases, new_tile_biases)

        # Asserts over the AnalogContext of the new model.
        new_analog_tile = self.get_analog_tile(new_model)
        analog_tile = self.get_analog_tile(model)

        self.assertTrue(hasattr(new_analog_tile.analog_ctx, "analog_tile"))
        self.assertIsInstance(new_analog_tile.analog_ctx.analog_tile, analog_tile.__class__)

        self.assertTrue(new_analog_tile.is_cuda != analog_tile.is_cuda)

        if analog_tile.shared_weights is not None:
            self.assertTrue(new_analog_tile.shared_weights.device.type == map_location)

    def test_save_load_meta_parameter(self):
        """Test saving and loading a device with custom parameters."""

        if self.tile_class in SKIP_META_PARAM_TILES:
            raise SkipTest("Not available")

        # Create the device and the array.
        rpu_config = SingleRPUConfig(
            forward=IOParameters(inp_noise=0.321),
            backward=IOParameters(inp_noise=0.456),
            update=UpdateParameters(desired_bl=78),
            device=ConstantStepDevice(w_max=0.987),
        )

        model = self.get_layer(rpu_config=rpu_config)

        # Save the model to a file.
        with TemporaryFile() as file:
            save(model, file)
            # Load the model.
            file.seek(0)
            new_model = load(file, weights_only=False)

        # Assert over the new model tile parameters.
        new_analog_tile = self.get_analog_tile(new_model)
        analog_tile = self.get_analog_tile(model)

        parameters = new_analog_tile.tile.get_meta_parameters()
        self.assertAlmostEqual(parameters.forward_io.inp_noise, 0.321)
        self.assertAlmostEqual(parameters.backward_io.inp_noise, 0.456)
        self.assertAlmostEqual(parameters.update.desired_bl, 78)
        self.assertTrue(new_analog_tile.is_cuda == analog_tile.is_cuda)

    def test_save_load_hidden_parameters(self):
        """Test saving and loading a device with hidden parameters."""
        # Create the device and the array.
        model = self.get_layer()
        analog_tile = self.get_analog_tile(model)
        hidden_parameters = analog_tile.tile.get_hidden_parameters()

        # Save the model to a file.
        with TemporaryFile() as file:
            save(model, file)
            # Load the model.
            file.seek(0)
            new_model = load(file, weights_only=False)

        # Assert over the new model tile parameters.
        new_analog_tile = self.get_analog_tile(new_model)
        new_hidden_parameters = new_analog_tile.tile.get_hidden_parameters()
        assert_array_almost_equal(hidden_parameters, new_hidden_parameters)

    def test_save_load_out_scaling_alpha(self):
        """Test saving and loading a device with out_scaling_alpha."""
        # Create the device and the array.
        model = self.get_layer()
        alpha = 2.0
        analog_tile = self.get_analog_tile(model)
        if analog_tile.out_scaling_alpha is None:
            analog_tile.out_scaling_alpha = Tensor([alpha])
        else:
            analog_tile.out_scaling_alpha.data = Tensor([alpha])

        # Save the model to a file.
        with TemporaryFile() as file:
            save(model, file)
            # Load the model.
            file.seek(0)
            new_model = load(file, weights_only=False)

        # Assert over the new model tile parameters.
        new_analog_tile = self.get_analog_tile(new_model)
        alpha_new = new_analog_tile.get_scales().detach().cpu()
        assert_array_almost_equal(array(alpha), array(alpha_new))

    def test_save_load_shared_weights(self):
        """Test saving and loading a device with shared_weights."""

        if isinstance(self.get_rpu_config(), FloatingPointRPUConfig):
            raise SkipTest("Not available for FP")

        # Create the device and the array.
        model = self.get_layer()
        analog_tile = self.get_analog_tile(model)
        shared_weights = None
        if analog_tile.shared_weights is not None:
            shared_weights = analog_tile.shared_weights.detach().cpu().numpy()

        # Save the model to a file.
        with TemporaryFile() as file:
            save(model, file)
            # Load the model.
            file.seek(0)
            new_model = load(file, weights_only=False)

        # Assert over the new model tile parameters.
        new_analog_tile = self.get_analog_tile(new_model)
        if shared_weights is not None:
            new_shared_weights = new_analog_tile.shared_weights
            assert_array_almost_equal(shared_weights, new_shared_weights.detach().cpu().numpy())

    def test_save_load_weight_scaling_omega(self):
        """Test saving and loading a device with weight scaling omega."""
        # Create the device and the array.
        rpu_config = SingleRPUConfig(mapping=MappingParameter(weight_scaling_omega=0.4))
        model = self.get_layer(rpu_config=rpu_config)
        analog_tile = self.get_analog_tile(model)
        alpha = analog_tile.get_scales().detach().cpu()
        self.assertNotEqual(alpha, 1.0)

        # Save the model to a file.
        with TemporaryFile() as file:
            save(model, file)
            # Load the model.
            file.seek(0)
            new_model = load(file, weights_only=False)

        # Assert over the new model tile parameters.
        new_analog_tile = self.get_analog_tile(new_model)
        alpha_new = new_analog_tile.get_scales().detach().cpu()
        assert_array_almost_equal(array(alpha), array(alpha_new))

    def test_remapping(self):
        """Test remapping of the weights."""
        # Create the device and the array.
        rpu_config = InferenceRPUConfig(
            mapping=MappingParameter(weight_scaling_omega=0.4, weight_scaling_columnwise=True)
        )

        model = self.get_layer(rpu_config=rpu_config)
        analog_tile = self.get_analog_tile(model)
        in_features = model.in_features
        out_features = model.out_features

        user_weights = Tensor(out_features, in_features).uniform_(-0.1, 0.1)
        if self.bias:
            user_biases = Tensor(out_features).uniform_(-0.1, 0.1)
        else:
            user_biases = None
        model.set_weights(user_weights, user_biases, apply_weight_scaling=True)

        alpha_initial = analog_tile.get_scales().detach().cpu()
        self.assertNotEqual(alpha_initial.max(), 1.0)

        # remap module
        model.remap_analog_weights(weight_scaling_omega=1.0)

        weights, _ = model.get_weights(apply_weight_scaling=True)
        analog_weights, _ = model.get_weights(apply_weight_scaling=False)

        assert_array_almost_equal(array(user_weights), array(weights))

        alpha = analog_tile.get_scales().detach().cpu()
        assert_raises(AssertionError, assert_array_almost_equal, array(alpha), array(alpha_initial))

        self.assertEqual(torch_abs(analog_weights).max(), 1.0)

    def test_save_load_state_dict_hidden_parameters(self):
        """Test saving and loading via state_dict with hidden parameters."""
        # Create the device and the array.
        model = self.get_layer()
        analog_tile = self.get_analog_tile(model)
        hidden_parameters = analog_tile.tile.get_hidden_parameters()

        # Save the model to a file.
        with TemporaryFile() as file:
            save(model.state_dict(), file)
            # Load the model.
            file.seek(0)
            new_model = self.get_layer()
            new_model.load_state_dict(load(file, weights_only=False))

        # Assert over the new model tile parameters.
        new_analog_tile = self.get_analog_tile(new_model)
        new_hidden_parameters = new_analog_tile.tile.get_hidden_parameters()
        assert_array_almost_equal(hidden_parameters, new_hidden_parameters)

    def test_state_dict_children_layers_sequential(self):
        """Test using the state_dict with children analog layers via Sequential."""
        children_layer = self.get_layer()
        model = Sequential(children_layer)

        # Keep track of the current weights and biases for comparing.
        (
            model_weights,
            model_biases,
            tile_weights,
            tile_biases,
            _,
        ) = self.get_layer_and_tile_weights(children_layer)

        state_dict = model.state_dict()
        lst = [
            key
            for key in state_dict.keys()
            if key.startswith("0.") and AnalogTileStateNames.ANALOG_STATE_NAME in key
        ]
        self.assertTrue(len(lst) > 0)

        # Update the state_dict of a new model.
        new_children_layer = self.get_layer()
        new_model = Sequential(new_children_layer)
        new_model.load_state_dict(model.state_dict())

        # Compare the new model weights and biases.
        (
            new_model_weights,
            new_model_biases,
            new_tile_weights,
            new_tile_biases,
            _,
        ) = self.get_layer_and_tile_weights(new_children_layer)

        assert_array_almost_equal(model_weights, new_model_weights)
        assert_array_almost_equal(tile_weights, new_tile_weights)
        if self.bias:
            assert_array_almost_equal(model_biases, new_model_biases)
            assert_array_almost_equal(tile_biases, new_tile_biases)

    def test_state_dict_children_layers_subclassing(self):
        """Test using the state_dict with children analog layers via subclassing."""

        class CustomModule(Module):
            """Module that defines its children layers via custom attributes."""

            # pylint: disable=abstract-method
            def __init__(self, layer: Module):
                super().__init__()
                self.custom_child = layer

        children_layer = self.get_layer()
        model = CustomModule(children_layer)

        # Keep track of the current weights and biases for comparing.
        (
            model_weights,
            model_biases,
            tile_weights,
            tile_biases,
            _,
        ) = self.get_layer_and_tile_weights(children_layer)

        state_dict = model.state_dict()
        lst = [
            key
            for key in state_dict.keys()
            if key.startswith("custom_child.") and AnalogTileStateNames.ANALOG_STATE_NAME in key
        ]
        self.assertTrue(len(lst) > 0)

        # Update the state_dict of a new model.
        new_children_layer = self.get_layer()
        new_model = CustomModule(new_children_layer)
        new_model.load_state_dict(model.state_dict())

        # Compare the new model weights and biases.
        (
            new_model_weights,
            new_model_biases,
            new_tile_weights,
            new_tile_biases,
            _,
        ) = self.get_layer_and_tile_weights(new_children_layer)

        assert_array_almost_equal(model_weights, new_model_weights)
        assert_array_almost_equal(tile_weights, new_tile_weights)
        if self.bias:
            assert_array_almost_equal(model_biases, new_model_biases)
            assert_array_almost_equal(tile_biases, new_tile_biases)

    def test_state_dict_analog_strict(self):
        """Test the `strict` flag for analog layers."""
        model = self.get_layer()
        state_dict = model.state_dict()

        # Remove the analog key from the state dict.
        lst = [
            key
            for key in state_dict.keys()
            if next(model.analog_tiles()).get_analog_state_name("") in key
        ]
        del state_dict[lst[0]]

        # Check that it fails when using `strict`.
        with self.assertRaises(RuntimeError) as context:
            model.load_state_dict(state_dict, strict=True)
        self.assertIn("Missing key", str(context.exception))

        # Check that it passes when not using `strict`.
        model.load_state_dict(state_dict, strict=False)

    def test_state_dict(self):
        """Test creating a new model using a state dict, without saving to disk."""
        model = self.get_layer()
        state_dict = model.state_dict()
        analog_tile = self.get_analog_tile(model)

        new_model = self.get_layer()
        new_model.load_state_dict(state_dict)
        new_analog_tile = self.get_analog_tile(new_model)

        # Asserts over the AnalogContext of the new model.
        self.assertTrue(hasattr(new_analog_tile.analog_ctx, "analog_tile"))
        self.assertIsInstance(new_analog_tile.analog_ctx.analog_tile, analog_tile.__class__)

    def test_hidden_parameter_mismatch(self):
        """Test for error if tile structure mismatches."""
        model = self.get_layer()
        state_dict = model.state_dict()
        analog_tile = self.get_analog_tile(model)

        # Create the device and the array.
        rpu_config = SingleRPUConfig(device=LinearStepDevice())  # different hidden structure

        new_model = self.get_layer(rpu_config=rpu_config)
        new_analog_tile = self.get_analog_tile(new_model)

        if "Torch" in analog_tile.__class__.__name__:
            raise SkipTest("Torch tile")

        if new_analog_tile.__class__.__name__ != analog_tile.__class__.__name__:
            with self.assertRaises(TileError):
                new_model.load_state_dict(state_dict)

    def test_load_state_load_rpu_config(self):
        """Test creating a new model using a state dict, while using a different RPU config."""

        # Create the device and the array.
        rpu_config_org = self.get_rpu_config()

        # Skipped for some
        if self.tile_class in SKIP_META_PARAM_TILES:
            raise SkipTest("Not available")

        rpu_config_org.forward.is_perfect = False
        old_value = 0.11
        rpu_config_org.forward.inp_noise = old_value

        model = self.get_layer(rpu_config=rpu_config_org)
        state_dict = model.state_dict()

        rpu_config = deepcopy(rpu_config_org)
        new_value = 0.51
        rpu_config.forward.inp_noise = new_value

        # Test restore_rpu_config=False
        new_model = self.get_layer(rpu_config=rpu_config)
        new_model.load_state_dict(state_dict, load_rpu_config=False)
        new_analog_tile = self.get_analog_tile(new_model)

        parameters = new_analog_tile.tile.get_meta_parameters()
        self.assertAlmostEqual(parameters.forward_io.inp_noise, new_value)

        # Test restore_rpu_config=True
        new_model = self.get_layer(rpu_config=rpu_config)
        new_model.load_state_dict(state_dict, load_rpu_config=True)
        new_analog_tile = self.get_analog_tile(new_model)

        parameters = new_analog_tile.tile.get_meta_parameters()
        self.assertAlmostEqual(parameters.forward_io.inp_noise, old_value)

    def test_load_state_load_rpu_config_sequential(self):
        """Test creating a new model using a state dict, while using a different RPU config."""

        # Create the device and the array.
        rpu_config_org = self.get_rpu_config()

        # Skipped for some
        if self.tile_class in SKIP_META_PARAM_TILES:
            raise SkipTest("Not available")

        rpu_config_org.forward.is_perfect = False
        old_value = 0.11
        rpu_config_org.forward.inp_noise = old_value

        model = AnalogSequential(self.get_layer(rpu_config=rpu_config_org))
        state_dict = model.state_dict()

        rpu_config = deepcopy(rpu_config_org)
        new_value = 0.51
        rpu_config.forward.inp_noise = new_value

        # Test restore_rpu_config=False
        new_model = AnalogSequential(self.get_layer(rpu_config=rpu_config))
        new_model.load_state_dict(state_dict, load_rpu_config=False)
        new_analog_tile = self.get_analog_tile(new_model[0])

        parameters = new_analog_tile.tile.get_meta_parameters()
        self.assertAlmostEqual(parameters.forward_io.inp_noise, new_value)

        # Test restore_rpu_config=True
        new_model = AnalogSequential(self.get_layer(rpu_config=rpu_config))
        new_model.load_state_dict(state_dict, load_rpu_config=True)
        new_analog_tile = self.get_analog_tile(new_model[0])

        parameters = new_analog_tile.tile.get_meta_parameters()
        self.assertAlmostEqual(parameters.forward_io.inp_noise, old_value)

    def test_load_state_load_rpu_config_wrong(self):
        """Test creating a new model using a state dict, while using a different RPU config."""

        # Skipped for FP
        if isinstance(self.get_rpu_config(), FloatingPointRPUConfig):
            raise SkipTest("Not available for FP")

        # Create the device and the array.
        model = self.get_layer()
        state_dict = model.state_dict()

        rpu_config = FloatingPointRPUConfig()

        new_model = self.get_layer(rpu_config=rpu_config)
        assert_raises(TileModuleError, new_model.load_state_dict, state_dict, load_rpu_config=False)


@parametrize_over_layers(
    layers=[Linear, LinearCuda],
    tiles=[FloatingPoint, Inference, TorchInference],
    biases=["digital"],
)
class SerializationTestExtended(ParametrizedTestCase):
    """Tests for serialization."""

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

    @staticmethod
    def get_torch_model(use_cuda: bool):
        """Returns a torch model."""
        manual_seed(4321)
        torch_model = Sequential(
            torch_linear(4, 3),
            torch_linear(3, 3),
            Sequential(torch_linear(3, 1), torch_linear(1, 1)),
        )
        if use_cuda:
            torch_model.cuda()
        return torch_model

    def test_load_state_dict_conversion(self):
        """Test loading and setting conversion with alpha."""

        # Create the device and the array.

        x_b = Tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.3, 0.1]])
        y_b = Tensor([[0.3], [0.6]])
        if self.use_cuda:
            x_b = x_b.cuda()
            y_b = y_b.cuda()

        model = self.get_torch_model(self.use_cuda)

        self.train_model_torch(model, mse_loss, x_b, y_b)

        analog_model = convert_to_analog(model, self.get_rpu_config())
        analog_loss = mse_loss(analog_model(x_b), y_b)

        with TemporaryFile() as file:
            save(analog_model.state_dict(), file)
            # Load the model.
            file.seek(0)
            model = self.get_torch_model(self.use_cuda)
            new_analog_model = convert_to_analog(model, self.get_rpu_config())
            state_dict = load(file, weights_only=False)
            new_analog_model.load_state_dict(state_dict, load_rpu_config=True)

        new_state_dict = new_analog_model.state_dict()
        for key in new_state_dict.keys():
            if not key.endswith("analog_tile_state"):
                continue

            state1 = new_state_dict[key]
            state2 = state_dict[key]
            assert_array_almost_equal(state1["analog_tile_weights"], state2["analog_tile_weights"])
            # assert_array_almost_equal(state1['analog_alpha_scale'],
            #                           state2['analog_alpha_scale'])

        new_analog_loss = mse_loss(new_analog_model(x_b), y_b)
        self.assertTensorAlmostEqual(new_analog_loss, analog_loss)

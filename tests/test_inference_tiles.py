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

"""Tests for inference tiles."""

from unittest import TestCase, skipIf

from numpy.testing import assert_array_almost_equal, assert_raises

from torch import ones
from torch import Tensor
from torch.nn.functional import mse_loss

from aihwkit.nn import AnalogLinear
from aihwkit.nn.modules.base import drift_analog_weights
from aihwkit.optim.analog_sgd import AnalogSGD
from aihwkit.simulator.configs.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import OutputWeightNoiseType
from aihwkit.simulator.rpu_base import cuda
from aihwkit.simulator.noise_models import PCMLikeNoiseModel


class InferenceTileMixin:
    """Common things for inference tile test"""

    def assertTensorAlmostEqual(self, tensor_a, tensor_b):
        """Assert that two tensors are almost equal."""
        # pylint: disable=invalid-name
        array_a = tensor_a.detach().cpu().numpy()
        array_b = tensor_b.detach().cpu().numpy()
        assert_array_almost_equal(array_a, array_b)

    def assertNotAlmostEqualTensor(self, tensor_a, tensor_b):
        """Assert that two tensors are not equal."""
        # pylint: disable=invalid-name
        assert_raises(AssertionError, self.assertTensorAlmostEqual, tensor_a, tensor_b)

    def get_model(self, cuda_if=False):
        """Trains a simple model."""
        # Prepare the datasets (input and expected output).
        x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
        y = Tensor([[1.0, 0.5], [0.7, 0.3]])

        # Define a single-layer network, using a constant step device type.
        rpu_config = InferenceRPUConfig()
        rpu_config.forward.out_res = -1.  # turn off (output) ADC discretization
        rpu_config.forward.w_noise_type = OutputWeightNoiseType.ADDITIVE_CONSTANT
        rpu_config.forward.w_noise = 0.02
        rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)

        model = AnalogLinear(4, 2, bias=True,
                             rpu_config=rpu_config)

        # Move the model and tensors to cuda if it is available.
        if cuda_if:
            x = x.cuda()
            y = y.cuda()
            model.cuda()

        # Define an analog-aware optimizer, preparing it for using the layers.
        opt = AnalogSGD(model.parameters(), lr=0.1)
        opt.regroup_param_groups(model)

        for _ in range(100):
            # Add the training Tensor to the model (input).
            pred = model(x)
            # Add the expected output Tensor.
            loss = mse_loss(pred, y)
            # Run training (backward propagation).
            loss.backward()

            opt.step()

        return model, x


class InferenceTileTest(TestCase, InferenceTileMixin):
    """Inference model tests."""

    def test_drift(self):
        """Test using realistic weights (bias)."""
        model, x = self.get_model()

        # do inference with drift
        pred_before = model(x)

        pred_last = pred_before
        for t_inference in [0., 1., 20., 1000., 1e5]:
            drift_analog_weights(model, t_inference)
            pred_drift = model(x)
            self.assertNotAlmostEqualTensor(pred_last, pred_drift)
            pred_last = pred_drift

        self.assertNotAlmostEqualTensor(model.analog_tile.alpha, ones((1,)))

    @skipIf(not cuda.is_compiled(), "No cuda available")
    def test_drift_cuda(self):
        """Test using realistic weights (bias)."""
        model, x = self.get_model(True)

        # do inference with drift
        pred_before = model(x)

        pred_last = pred_before
        for t_inference in [0., 1., 20., 1000., 1e5]:
            drift_analog_weights(model, t_inference)
            pred_drift = model(x)
            self.assertNotAlmostEqualTensor(pred_last, pred_drift)
            pred_last = pred_drift

        self.assertNotAlmostEqualTensor(model.analog_tile.alpha, ones((1,)))

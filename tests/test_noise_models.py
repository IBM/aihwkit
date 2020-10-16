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

"""Tests for general functionality of layers."""

from unittest import TestCase

from numpy.testing import assert_array_almost_equal, assert_raises

from torch import randn

from aihwkit.simulator.noise_models import PCMLikeNoiseModel, SinglePairConductanceConverter


class NoiseModelMixin:
    """Common things for noise model test"""

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


class NoiseModelTest(TestCase, NoiseModelMixin):
    """Noise model tests."""

    def test_apply_noise(self):
        """Test using realistic weights (bias)."""
        weights = randn(10, 35)

        noise_model = PCMLikeNoiseModel()
        t_inference = 100.
        noisy_weights = noise_model.apply_noise(weights, t_inference)

        self.assertNotAlmostEqualTensor(noisy_weights, weights)


class ConductanceConverterTest(TestCase, NoiseModelMixin):
    """Conductance converter test."""

    def test_single_pair_converter(self):
        """Tests the single pair converter."""
        g_max = 3.123
        g_min = 0.789

        weights = randn(10, 35)

        g_converter = SinglePairConductanceConverter(g_max=g_max, g_min=g_min)

        g_lst, params = g_converter.convert_to_conductances(weights)

        g_plus = g_lst[0].detach().cpu().numpy()
        g_minus = g_lst[1].detach().cpu().numpy()

        tolerance = 1e-6

        self.assertTrue((g_plus > g_max - tolerance).sum()
                        + (g_minus > g_max - tolerance).sum() > 0)
        self.assertTrue((g_plus < g_min - tolerance).sum()
                        + (g_minus < g_min - tolerance).sum() == 0)
        self.assertTrue((g_plus > g_max + tolerance).sum()
                        + (g_minus > g_max + tolerance).sum() == 0)

        converted_weights = g_converter.convert_back_to_weights(g_lst, params)

        self.assertTensorAlmostEqual(weights, converted_weights)

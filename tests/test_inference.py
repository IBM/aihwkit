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

"""Tests for general functionality of layers."""

from torch import randn

from aihwkit.inference import (
    PCMLikeNoiseModel,
    StateIndependentNoiseModel,
    SinglePairConductanceConverter,
)

from .helpers.testcases import AihwkitTestCase


class NoiseModelTest(AihwkitTestCase):
    """Noise model tests."""

    def test_apply_noise_pcm(self):
        """Test using realistic weights (bias)."""
        weights = randn(10, 35)

        noise_model = PCMLikeNoiseModel()
        t_inference = 100.0
        noisy_weights = noise_model.apply_noise(weights, t_inference)

        self.assertNotAlmostEqualTensor(noisy_weights, weights)

    def test_apply_noise_custom(self):
        """Test using realistic weights (bias)."""
        weights = randn(10, 35)

        noise_model = StateIndependentNoiseModel()
        t_inference = 100.0
        noisy_weights = noise_model.apply_noise(weights, t_inference)

        self.assertNotAlmostEqualTensor(noisy_weights, weights)


class ConductanceConverterTest(AihwkitTestCase):
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

        self.assertTrue(
            (g_plus > g_max - tolerance).sum() + (g_minus > g_max - tolerance).sum() > 0
        )
        self.assertTrue(
            (g_plus < g_min - tolerance).sum() + (g_minus < g_min - tolerance).sum() == 0
        )
        self.assertTrue(
            (g_plus > g_max + tolerance).sum() + (g_minus > g_max + tolerance).sum() == 0
        )

        converted_weights = g_converter.convert_back_to_weights(g_lst, params)

        self.assertTensorAlmostEqual(weights, converted_weights)

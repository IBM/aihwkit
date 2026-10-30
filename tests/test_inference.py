# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for general functionality of layers."""

from torch import randn, full, manual_seed

from aihwkit.inference import (
    PCMLikeNoiseModel,
    ReRamCMONoiseModel,
    ReRamWan2022NoiseModel,
    CustomDriftPCMLikeNoiseModel,
    StateIndependentNoiseModel,
    SinglePairConductanceConverter,
    SingleDeviceConductanceConverter,
    DualPairConductanceConverter,
    NPairConductanceConverter,
    CustomPairConductanceConverter,
)

from aihwkit.inference.noise.hermes import HermesNoiseModel

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

    def test_apply_noise_reram_cmo(self):
        """Test using realistic weights (bias)."""
        weights = randn(10, 35)

        noise_model = ReRamCMONoiseModel()
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

    def test_apply_custom_drift(self):
        """Test custom drift model with g_converter"""
        weights = randn(10, 35)

        g_min, g_max = 0.0, 25.0
        custom_drift_model = dict(
            g_lst=[g_min, 10.0, g_max],
            nu_mean_lst=[0.08, 0.05, 0.03],
            nu_std_lst=[0.03, 0.02, 0.01],
        )

        noise_model = CustomDriftPCMLikeNoiseModel(
            custom_drift_model, g_converter=SinglePairConductanceConverter(g_min=g_min, g_max=g_max)
        )
        t_inference = 100.0
        noisy_weights = noise_model.apply_noise(weights, t_inference)

        self.assertNotAlmostEqualTensor(noisy_weights, weights)

    def test_g_max_reference_explicit(self):
        """Test that an explicit g_max reference is stored and used (#830)."""
        g_target = full((2, 2), 10.0)
        drift = dict(
            g_lst=[0.0, 10.0, 40.0], nu_mean_lst=[0.08, 0.05, 0.03], nu_std_lst=[0.03, 0.02, 0.01]
        )
        coeff = [0.2, 1.9, -1.1]
        models = [
            (PCMLikeNoiseModel(g_max=25.0, prog_coeff_g_max_reference=40.0), 40.0),
            (
                PCMLikeNoiseModel(g_max=25.0, prog_coeff=coeff, prog_coeff_g_max_reference=40.0),
                40.0,
            ),
            (
                CustomDriftPCMLikeNoiseModel(
                    drift, g_max=25.0, prog_coeff=coeff, prog_coeff_g_max_reference=40.0
                ),
                40.0,
            ),
            (
                HermesNoiseModel(g_max=20.0, prog_coeff=coeff, prog_coeff_g_max_reference=40.0),
                40.0,
            ),
            (HermesNoiseModel(g_max=20.0, prog_coeff_g_max_reference=40.0), 40.0),
        ]
        for model, reference in models:
            self.assertEqual(model.prog_coeff_g_max_reference, reference)
            model.apply_programming_noise_to_conductance(g_target.clone())

        model = ReRamWan2022NoiseModel(g_max=25.0, coeff_g_max_reference=50.0)
        self.assertEqual(model.coeff_g_max_reference, 50.0)
        model.apply_programming_noise_to_conductance(g_target.clone())

    def test_g_max_reference_defaults(self):
        """Test the default g_max reference values (#830)."""
        coeff = [0.2, 1.9, -1.1]
        self.assertEqual(PCMLikeNoiseModel(g_max=20.0).prog_coeff_g_max_reference, 25.0)
        self.assertEqual(
            PCMLikeNoiseModel(g_max=20.0, prog_coeff=coeff).prog_coeff_g_max_reference, 20.0
        )
        self.assertEqual(HermesNoiseModel(num_devices=2).prog_coeff_g_max_reference, 20.7)
        self.assertEqual(HermesNoiseModel(num_devices=1).prog_coeff_g_max_reference, 10.35)
        self.assertEqual(
            HermesNoiseModel(g_max=10.0, prog_coeff=coeff).prog_coeff_g_max_reference, 10.0
        )
        self.assertEqual(ReRamWan2022NoiseModel(g_max=25.0).coeff_g_max_reference, 40.0)
        self.assertEqual(
            ReRamWan2022NoiseModel(coeff_dic={1.0: coeff}, g_max=25.0).coeff_g_max_reference,
            25.0,
        )

    def test_reram_default_reference_scaling(self):
        """Test that the 40 uS reference scales the default ReRAM noise (#830)."""
        stds = {}
        for g_max in (25.0, 40.0):
            manual_seed(0)
            model = ReRamWan2022NoiseModel(g_max=g_max)
            g_target = full((200000,), 0.5 * g_max)
            stds[g_max] = (model.apply_programming_noise_to_conductance(g_target) - g_target).std()

        self.assertAlmostEqual(float(stds[25.0] / stds[40.0]), 25.0 / 40.0, delta=0.02)


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

    def test_dual_pair_converter(self):
        """Tests the dual pair converter."""
        g_max = 3.123
        g_min = 0.789

        weights = randn(10, 35)

        g_converter = DualPairConductanceConverter(f_lst=[1.0, 3.0], g_max=g_max, g_min=g_min)

        g_lst, params = g_converter.convert_to_conductances(weights)

        tolerance = 1e-6
        for g_plus, g_minus in zip(g_lst[::2], g_lst[1::2]):

            g_plus = g_lst[0].detach().cpu().numpy()
            g_minus = g_lst[1].detach().cpu().numpy()

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

    def test_n_pair_converter(self):
        """Tests the dual pair converter."""
        g_max = 3.123
        g_min = 0.789

        weights = randn(10, 35)

        g_converter = NPairConductanceConverter(f_lst=[1.0, 2.0, 3.0], g_max=g_max, g_min=g_min)

        g_lst, params = g_converter.convert_to_conductances(weights)

        tolerance = 1e-6
        for g_plus, g_minus in zip(g_lst[::2], g_lst[1::2]):

            g_plus = g_lst[0].detach().cpu().numpy()
            g_minus = g_lst[1].detach().cpu().numpy()

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

    def test_custom_pair_converter(self):
        """Tests the dual pair converter."""
        g_max = 3.123
        g_min = 0.789

        weights = randn(10, 35)

        g_converter = CustomPairConductanceConverter(
            f_lst=[1.0],
            g_lst=[
                [g_min, g_min, g_min, (g_max - g_min) / 2 + g_min, g_max],
                [g_max, (g_max - g_min) / 2 + g_min, g_min, g_min, g_min],
            ],
            g_min=g_min,
            g_max=g_max,
            invertibility_test=False,
        )

        g_lst, params = g_converter.convert_to_conductances(weights)

        tolerance = 1e-6
        for g_plus, g_minus in zip(g_lst[::2], g_lst[1::2]):
            g_plus = g_lst[0].detach().cpu().numpy()
            g_minus = g_lst[1].detach().cpu().numpy()

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

        self.assertTensorAlmostEqual(weights, converted_weights)  # invertibility test

    def test_single_device_converter(self):
        """Tests single bidirectional switching device converter"""
        g_max = 88.19
        g_min = 9.0

        weights = randn(10, 35)
        g_converter = SingleDeviceConductanceConverter(g_max=g_max, g_min=g_min)
        g_lst, params = g_converter.convert_to_conductances(weights=weights)
        tolerance = 1e-6
        self.assertTrue((g_lst > g_max - tolerance).sum() == 0)
        self.assertTrue((g_lst < g_min - tolerance).sum() == 0)

        converted_weights = g_converter.convert_back_to_weights(g_lst, params)

        self.assertTensorAlmostEqual(weights, converted_weights)

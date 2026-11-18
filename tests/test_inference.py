# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for general functionality of layers."""

import pickle
import warnings

from parameterized import parameterized
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


def _reference_warnings(caught):
    """Filter the g_max reference override warnings out of the recorded ones."""
    return [w for w in caught if "prog_coeff_g_max_reference" in str(w.message)]


def _custom_drift_model():
    """Minimal custom drift model for ``CustomDriftPCMLikeNoiseModel``."""
    return dict(
        g_lst=[0.0, 10.0, 25.0], nu_mean_lst=[0.08, 0.05, 0.03], nu_std_lst=[0.03, 0.02, 0.01]
    )


class NoiseModelGMaxReferenceTest(AihwkitTestCase):
    """Tests for the ``prog_coeff_g_max_reference`` handling of the noise models.

    The default programming noise coefficients are fitted at a fixed
    reference g_max (25 uS for PCM, 40 uS for ReRAM) and the noise is
    scaled by ``g_max / prog_coeff_g_max_reference``.
    """

    @parameterized.expand(
        [
            ("pcm", PCMLikeNoiseModel, 25.0),
            ("reram", ReRamWan2022NoiseModel, 40.0),
        ]
    )
    def test_default_reference(self, _, model_cls, expected_reference):
        """Default coefficients use the calibrated reference regardless of g_max."""
        for g_max in [None, 25.0, 40.0, 80.0]:
            kwargs = {} if g_max is None else {"g_max": g_max}
            noise_model = model_cls(**kwargs)
            self.assertEqual(noise_model.prog_coeff_g_max_reference, expected_reference)

    def test_reram_has_no_legacy_attribute(self):
        """ReRAM stores the reference only as ``prog_coeff_g_max_reference``."""
        noise_model = ReRamWan2022NoiseModel()
        self.assertFalse(hasattr(noise_model, "coeff_g_max_reference"))

    def test_reram_explicit_reference(self):
        """Passing ``coeff_g_max_reference`` does not raise an AttributeError."""
        noise_model = ReRamWan2022NoiseModel(g_max=25.0, coeff_g_max_reference=40.0)
        g_prog = noise_model.apply_programming_noise_to_conductance(full((2, 2), 10.0))

        self.assertEqual(noise_model.prog_coeff_g_max_reference, 40.0)
        self.assertEqual(g_prog.shape, (2, 2))

    def test_pcm_explicit_reference_with_custom_coeff(self):
        """Passing ``prog_coeff`` and a reference does not raise an AttributeError."""
        noise_model = PCMLikeNoiseModel(
            g_max=40.0, prog_coeff=[0.26348, 1.9650, -1.1731], prog_coeff_g_max_reference=25.0
        )
        g_prog = noise_model.apply_programming_noise_to_conductance(full((2, 2), 10.0))

        self.assertEqual(noise_model.prog_coeff_g_max_reference, 25.0)
        self.assertEqual(g_prog.shape, (2, 2))

    @parameterized.expand(
        [
            (
                "pcm",
                PCMLikeNoiseModel,
                {"prog_coeff": [0.1, 0.2, 0.3]},
                "prog_coeff_g_max_reference",
            ),
            (
                "reram",
                ReRamWan2022NoiseModel,
                {"coeff_dic": {1.0: [0.1, 0.2, 0.3]}},
                "coeff_g_max_reference",
            ),
        ]
    )
    def test_custom_coeff_reference(self, _, model_cls, coeff_kwargs, ref_key):
        """With custom coefficients the reference defaults to g_max or the given value."""
        noise_model = model_cls(g_max=50.0, **coeff_kwargs)
        self.assertEqual(noise_model.prog_coeff_g_max_reference, 50.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            noise_model = model_cls(g_max=50.0, **coeff_kwargs, **{ref_key: 20.0})
        self.assertEqual(noise_model.prog_coeff_g_max_reference, 20.0)
        self.assertEqual(_reference_warnings(caught), [])

    @parameterized.expand(
        [
            ("pcm", PCMLikeNoiseModel, "prog_coeff_g_max_reference", 25.0),
            ("reram", ReRamWan2022NoiseModel, "coeff_g_max_reference", 40.0),
        ]
    )
    def test_override_default_reference_warns(self, _, model_cls, ref_key, default_reference):
        """Overriding the calibrated reference warns and applies the given value."""
        with self.assertWarns(UserWarning) as context:
            noise_model = model_cls(g_max=40.0, **{ref_key: 30.0})

        self.assertEqual(noise_model.prog_coeff_g_max_reference, 30.0)
        message = str(context.warning)
        self.assertIn(model_cls.__name__, message)
        self.assertIn(str(default_reference), message)
        self.assertIn("prog_coeff_g_max_reference", message)

    @parameterized.expand(
        [
            ("pcm", PCMLikeNoiseModel, "prog_coeff_g_max_reference", 25.0),
            ("reram", ReRamWan2022NoiseModel, "coeff_g_max_reference", 40.0),
        ]
    )
    def test_default_reference_value_does_not_warn(self, _, model_cls, ref_key, reference):
        """Passing the calibrated reference explicitly does not warn."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            noise_model = model_cls(g_max=25.0, **{ref_key: reference})

        self.assertEqual(noise_model.prog_coeff_g_max_reference, reference)
        self.assertEqual(_reference_warnings(caught), [])

    def test_custom_drift_pcm_override_warns(self):
        """``CustomDriftPCMLikeNoiseModel`` inherits the override warning."""
        with self.assertWarns(UserWarning) as context:
            noise_model = CustomDriftPCMLikeNoiseModel(
                _custom_drift_model(), g_max=25.0, prog_coeff_g_max_reference=40.0
            )

        self.assertEqual(noise_model.prog_coeff_g_max_reference, 40.0)
        self.assertIn("CustomDriftPCMLikeNoiseModel", str(context.warning))

    @parameterized.expand(
        [
            ("pcm", PCMLikeNoiseModel, "prog_noise_scale"),
            ("reram", ReRamWan2022NoiseModel, "noise_scale"),
        ]
    )
    def test_programming_noise_scales_with_g_max(self, _, model_cls, scale_key):
        """Doubling g_max (same relative target) doubles the programming noise."""
        # small noise scale so that no value is clamped at zero conductance
        deltas = []
        for g_max in [40.0, 80.0]:
            noise_model = model_cls(g_max=g_max, **{scale_key: 0.01})
            g_target = full((10, 10), 0.5 * g_max)
            manual_seed(42)
            g_prog = noise_model.apply_programming_noise_to_conductance(g_target)
            self.assertTrue((g_prog > 0).all())
            deltas.append(g_prog - g_target)

        self.assertTensorAlmostEqual(deltas[1], 2.0 * deltas[0])

    def test_reram_legacy_pickle(self):
        """Pickled ReRAM models with ``coeff_g_max_reference`` are mapped on load."""
        noise_model = ReRamWan2022NoiseModel(g_max=50.0, coeff_dic={1.0: [0.1, 0.2, 0.3]})
        state = dict(noise_model.__dict__)
        state["coeff_g_max_reference"] = state.pop("prog_coeff_g_max_reference")
        noise_model.__dict__ = state

        loaded = pickle.loads(pickle.dumps(noise_model))

        self.assertFalse(hasattr(loaded, "coeff_g_max_reference"))
        self.assertEqual(loaded.prog_coeff_g_max_reference, 50.0)
        loaded.apply_programming_noise_to_conductance(full((2, 2), 10.0))

    def test_reram_legacy_pickle_keeps_calibrated_reference(self):
        """Legacy pickles with default coefficients keep the 40 uS reference."""
        noise_model = ReRamWan2022NoiseModel(g_max=25.0)
        noise_model.__dict__["coeff_g_max_reference"] = 25.0  # old buggy state

        loaded = pickle.loads(pickle.dumps(noise_model))

        self.assertFalse(hasattr(loaded, "coeff_g_max_reference"))
        self.assertEqual(loaded.prog_coeff_g_max_reference, 40.0)


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

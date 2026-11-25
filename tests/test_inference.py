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


def _construct(model_cls, *args, **kwargs):
    """Construct a noise model and return it with its g_max reference warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        noise_model = model_cls(*args, **kwargs)
    reference_warnings = [w for w in caught if "prog_coeff_g_max_reference" in str(w.message)]
    return noise_model, reference_warnings


def _custom_drift_model():
    """Minimal custom drift model for ``CustomDriftPCMLikeNoiseModel``."""
    return dict(
        g_lst=[0.0, 10.0, 25.0], nu_mean_lst=[0.08, 0.05, 0.03], nu_std_lst=[0.03, 0.02, 0.01]
    )


class NoiseModelGMaxReferenceTest(AihwkitTestCase):
    """Tests for the ``g_max`` / ``prog_coeff_g_max_reference`` handling of the noise models.

    The default programming noise coefficients are fitted at a fixed
    reference g_max (25 uS for PCM, 40 uS for ReRAM) that cannot be
    changed. ``g_max`` cannot be larger than the reference, and a
    smaller ``g_max`` issues a warning.
    """

    @parameterized.expand(
        [
            ("pcm", PCMLikeNoiseModel, 25.0),
            ("reram", ReRamWan2022NoiseModel, 40.0),
        ]
    )
    def test_defaults(self, _, model_cls, reference):
        """Without arguments g_max equals the calibrated reference and nothing warns."""
        noise_model, caught = _construct(model_cls)

        self.assertEqual(noise_model.g_max, reference)
        self.assertEqual(noise_model.prog_coeff_g_max_reference, reference)
        self.assertEqual(caught, [])

    @parameterized.expand(
        [
            ("pcm", PCMLikeNoiseModel, 25.0),
            ("reram", ReRamWan2022NoiseModel, 40.0),
        ]
    )
    def test_g_max_equal_to_reference_does_not_warn(self, _, model_cls, reference):
        """Setting g_max to the calibrated reference does not warn."""
        noise_model, caught = _construct(model_cls, g_max=reference)

        self.assertEqual(noise_model.prog_coeff_g_max_reference, reference)
        self.assertEqual(caught, [])

    @parameterized.expand(
        [
            ("pcm", PCMLikeNoiseModel, "prog_coeff_g_max_reference", 25.0),
            ("reram", ReRamWan2022NoiseModel, "coeff_g_max_reference", 40.0),
        ]
    )
    def test_reference_ignored_with_default_coeff(self, _, model_cls, ref_key, reference):
        """With the default coefficients the given reference is ignored."""
        noise_model, caught = _construct(model_cls, g_max=reference, **{ref_key: 30.0})

        self.assertEqual(noise_model.prog_coeff_g_max_reference, reference)
        self.assertEqual(caught, [])

    @parameterized.expand(
        [
            ("pcm", PCMLikeNoiseModel, 25.0),
            ("reram", ReRamWan2022NoiseModel, 40.0),
        ]
    )
    def test_g_max_larger_than_reference_raises(self, _, model_cls, reference):
        """g_max larger than the calibrated reference raises an error."""
        with self.assertRaises(ValueError) as context:
            model_cls(g_max=reference + 1.0)

        message = str(context.exception)
        self.assertIn(model_cls.__name__, message)
        self.assertIn(str(reference), message)
        self.assertIn("cannot be larger", message)

    @parameterized.expand(
        [
            ("pcm", PCMLikeNoiseModel, 25.0),
            ("reram", ReRamWan2022NoiseModel, 40.0),
        ]
    )
    def test_g_max_from_converter_larger_than_reference_raises(self, _, model_cls, reference):
        """The check also applies when g_max comes from the conductance converter."""
        g_converter = SinglePairConductanceConverter(g_max=reference + 1.0)
        with self.assertRaises(ValueError):
            model_cls(g_converter=g_converter)

    @parameterized.expand(
        [
            ("pcm", PCMLikeNoiseModel, 25.0),
            ("reram", ReRamWan2022NoiseModel, 40.0),
        ]
    )
    def test_g_max_smaller_than_reference_warns(self, _, model_cls, reference):
        """g_max smaller than the calibrated reference warns and keeps the reference."""
        noise_model, caught = _construct(model_cls, g_max=reference / 2)

        self.assertEqual(noise_model.g_max, reference / 2)
        self.assertEqual(noise_model.prog_coeff_g_max_reference, reference)
        self.assertEqual(len(caught), 1)
        message = str(caught[0].message)
        self.assertIn(model_cls.__name__, message)
        self.assertIn("is smaller than", message)

    def test_reram_has_no_legacy_attribute(self):
        """ReRAM stores the reference only as ``prog_coeff_g_max_reference``."""
        noise_model = ReRamWan2022NoiseModel()
        self.assertFalse(hasattr(noise_model, "coeff_g_max_reference"))

    def test_reram_explicit_reference(self):
        """Passing ``coeff_g_max_reference`` does not raise an AttributeError."""
        noise_model, caught = _construct(
            ReRamWan2022NoiseModel, g_max=25.0, coeff_g_max_reference=40.0
        )
        g_prog = noise_model.apply_programming_noise_to_conductance(full((2, 2), 10.0))

        self.assertEqual(noise_model.prog_coeff_g_max_reference, 40.0)
        self.assertEqual(len(caught), 1)  # g_max < reference
        self.assertEqual(g_prog.shape, (2, 2))

    def test_pcm_explicit_reference_with_custom_coeff(self):
        """Passing ``prog_coeff`` and a reference does not raise an AttributeError."""
        noise_model, _ = _construct(
            PCMLikeNoiseModel,
            g_max=20.0,
            prog_coeff=[0.26348, 1.9650, -1.1731],
            prog_coeff_g_max_reference=25.0,
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
        """With custom coefficients the reference can be set and is checked against g_max."""
        # no reference: falls back to g_max
        noise_model, caught = _construct(model_cls, g_max=50.0, **coeff_kwargs)
        self.assertEqual(noise_model.prog_coeff_g_max_reference, 50.0)
        self.assertEqual(caught, [])

        # reference above g_max: applied, with a warning
        noise_model, caught = _construct(model_cls, g_max=50.0, **coeff_kwargs, **{ref_key: 60.0})
        self.assertEqual(noise_model.prog_coeff_g_max_reference, 60.0)
        self.assertEqual(len(caught), 1)

        # reference below g_max: error
        with self.assertRaises(ValueError):
            model_cls(g_max=50.0, **coeff_kwargs, **{ref_key: 20.0})

    def test_custom_drift_pcm_inherits_checks(self):
        """``CustomDriftPCMLikeNoiseModel`` applies the same g_max checks."""
        with self.assertRaises(ValueError):
            CustomDriftPCMLikeNoiseModel(_custom_drift_model(), g_max=30.0)

        noise_model, caught = _construct(
            CustomDriftPCMLikeNoiseModel, _custom_drift_model(), g_max=20.0
        )
        self.assertEqual(noise_model.prog_coeff_g_max_reference, 25.0)
        self.assertEqual(len(caught), 1)
        self.assertIn("CustomDriftPCMLikeNoiseModel", str(caught[0].message))

    @parameterized.expand(
        [
            ("pcm", PCMLikeNoiseModel, "prog_noise_scale", 25.0),
            ("reram", ReRamWan2022NoiseModel, "noise_scale", 40.0),
        ]
    )
    def test_programming_noise_scales_with_g_max(self, _, model_cls, scale_key, reference):
        """Doubling g_max (same relative target) doubles the programming noise."""
        # small noise scale so that no value is clamped at zero conductance
        deltas = []
        for g_max in [reference / 2, reference]:
            noise_model, _ = _construct(model_cls, g_max=g_max, **{scale_key: 0.01})
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
        noise_model = ReRamWan2022NoiseModel()
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

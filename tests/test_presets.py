# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for analog presets."""

from torch import Tensor, randn

from aihwkit.simulator.tiles.analog import AnalogTile
from aihwkit.simulator.presets import (
    ReRamESPreset,
    ReRamSBPreset,
    CapacitorPreset,
    EcRamPreset,
    EcRamMOPreset,
    IdealizedPreset,
    GokmenVlasovPreset,
    PCMPreset,
    ReRamES2Preset,
    ReRamSB2Preset,
    Capacitor2Preset,
    EcRam2Preset,
    EcRamMO2Preset,
    Idealized2Preset,
    ReRamES4Preset,
    ReRamSB4Preset,
    Capacitor4Preset,
    EcRam4Preset,
    EcRamMO4Preset,
    Idealized4Preset,
    TikiTakaReRamESPreset,
    TikiTakaReRamSBPreset,
    TikiTakaCapacitorPreset,
    TikiTakaEcRamPreset,
    TikiTakaEcRamMOPreset,
    TikiTakaIdealizedPreset,
    MixedPrecisionReRamESPreset,
    MixedPrecisionReRamSBPreset,
    MixedPrecisionCapacitorPreset,
    MixedPrecisionEcRamPreset,
    MixedPrecisionEcRamMOPreset,
    MixedPrecisionIdealizedPreset,
    MixedPrecisionGokmenVlasovPreset,
    MixedPrecisionPCMPreset,
    TTv2ReRamESPreset,
    TTv2ReRamSBPreset,
    TTv2CapacitorPreset,
    TTv2EcRamPreset,
    TTv2EcRamMOPreset,
    TTv2IdealizedPreset,
    FloatingPointPreset,
)
from .helpers.decorators import parametrize_over_presets
from .helpers.testcases import AihwkitTestCase


@parametrize_over_presets(
    [
        ReRamESPreset,
        ReRamSBPreset,
        CapacitorPreset,
        EcRamPreset,
        EcRamMOPreset,
        IdealizedPreset,
        GokmenVlasovPreset,
        PCMPreset,
        ReRamES2Preset,
        ReRamSB2Preset,
        Capacitor2Preset,
        EcRam2Preset,
        EcRamMO2Preset,
        Idealized2Preset,
        ReRamES4Preset,
        ReRamSB4Preset,
        Capacitor4Preset,
        EcRam4Preset,
        EcRamMO4Preset,
        Idealized4Preset,
        TikiTakaReRamESPreset,
        TikiTakaReRamSBPreset,
        TikiTakaCapacitorPreset,
        TikiTakaEcRamPreset,
        TikiTakaEcRamMOPreset,
        TikiTakaIdealizedPreset,
        MixedPrecisionReRamESPreset,
        MixedPrecisionReRamSBPreset,
        MixedPrecisionCapacitorPreset,
        MixedPrecisionEcRamPreset,
        MixedPrecisionEcRamMOPreset,
        MixedPrecisionIdealizedPreset,
        MixedPrecisionGokmenVlasovPreset,
        MixedPrecisionPCMPreset,
        TTv2ReRamESPreset,
        TTv2ReRamSBPreset,
        TTv2CapacitorPreset,
        TTv2EcRamPreset,
        TTv2EcRamMOPreset,
        TTv2IdealizedPreset,
    ]
)
class PresetTest(AihwkitTestCase):
    """Test for analog presets."""

    def test_tile_preset(self):
        """Test instantiating a tile that uses a preset."""
        out_size = 2
        in_size = 3

        rpu_config = self.preset_cls()
        analog_tile = AnalogTile(out_size, in_size, rpu_config, bias=False)

        learning_rate = 0.123

        # Use [out_size, in_size] weights.
        weights = Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        # Set some properties in the simulators.Tile.
        analog_tile.set_learning_rate(0.123)
        analog_tile.set_weights(weights)

        # Assert over learning rate.
        self.assertAlmostEqual(analog_tile.get_learning_rate(), learning_rate)
        self.assertAlmostEqual(
            analog_tile.get_learning_rate(), analog_tile.tile.get_learning_rate()
        )

        # Assert over weights and biases.
        tile_weights, tile_biases = analog_tile.get_weights()
        self.assertEqual(tuple(tile_weights.shape), (out_size, in_size))
        self.assertEqual(tile_biases, None)
        # TODO: disabled as the comparison needs to take into account noise
        # self.assertTensorAlmostEqual(tile_weights, weights)


class PresetTestFP(AihwkitTestCase):
    """Test for FP preset."""

    def test_tile_preset(self):
        """Test fwd behavior of FP preset."""
        out_size = 2
        in_size = 3
        weights = randn(out_size, in_size)
        inp = randn(in_size)
        fp_out = inp @ weights.T

        rpu_config = FloatingPointPreset()
        analog_tile = AnalogTile(out_size, in_size, rpu_config, bias=False)
        analog_tile.set_weights(weights)
        tile_out = analog_tile(inp)
        self.assertTensorAlmostEqual(fp_out, tile_out)

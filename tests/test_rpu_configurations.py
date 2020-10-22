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

"""Tests for the high level simulator devices functionality."""

from aihwkit.simulator.configs.utils import (
    IOParameters, UpdateParameters
)

from .helpers.decorators import parametrize_over_tiles
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import (
    FloatingPoint, Ideal, ConstantStep, LinearStep,
    ExpStep, Vector, Difference, Transfer,
    FloatingPointCuda, IdealCuda, ConstantStepCuda, LinearStepCuda,
    ExpStepCuda, VectorCuda, DifferenceCuda, TransferCuda
)


@parametrize_over_tiles([
    FloatingPoint,
    FloatingPointCuda
])
class RPUConfigurationsFloatingPointTest(ParametrizedTestCase):
    """Tests related to resistive processing unit configurations (floating point)."""

    def test_create_array(self):
        """Test creating an array using the mappings to bindings."""
        rpu_config = self.get_rpu_config()

        tile_params = rpu_config.device.as_bindings()

        _ = tile_params.create_array(10, 20)

    def test_config_device_parameters(self):
        """Test modifying the device parameters."""
        rpu_config = self.get_rpu_config()

        rpu_config.device.diffusion = 1.23
        rpu_config.device.lifetime = 4.56

        tile = self.get_tile(11, 22, rpu_config).tile

        # Assert over the parameters in the binding objects.
        parameters = tile.get_parameters()
        self.assertAlmostEqual(parameters.diffusion, 1.23, places=4)
        self.assertAlmostEqual(parameters.lifetime, 4.56, places=4)


@parametrize_over_tiles([
    Ideal,
    ConstantStep,
    LinearStep,
    ExpStep,
    Vector,
    Difference,
    Transfer,
    IdealCuda,
    ConstantStepCuda,
    LinearStepCuda,
    ExpStepCuda,
    VectorCuda,
    DifferenceCuda,
    TransferCuda,
])
class RPUConfigurationsTest(ParametrizedTestCase):
    """Tests related to resistive processing unit configurations."""

    def test_create_array(self):
        """Test creating an array using the mappings to bindings."""
        rpu_config = self.get_rpu_config()

        tile_params = rpu_config.as_bindings()
        device_params = rpu_config.device.as_bindings()

        _ = tile_params.create_array(10, 20, device_params)

    def test_config_device_parameters(self):
        """Test modifying the device parameters."""
        rpu_config = self.get_rpu_config()

        rpu_config.device.diffusion = 1.23
        rpu_config.device.lifetime = 4.56
        rpu_config.device.construction_seed = 192

        # TODO: don't assert over tile.get_parameters() as some of them might
        # not be present.
        _ = self.get_tile(11, 22, rpu_config).tile

    def test_config_tile_parameters(self):
        """Test modifying the tile parameters."""
        rpu_config = self.get_rpu_config()

        rpu_config.forward = IOParameters(inp_noise=0.321)
        rpu_config.backward = IOParameters(inp_noise=0.456)
        rpu_config.update = UpdateParameters(desired_bl=78)

        tile = self.get_tile(11, 22, rpu_config).tile

        # Assert over the parameters in the binding objects.
        parameters = tile.get_parameters()
        self.assertAlmostEqual(parameters.forward_io.inp_noise, 0.321)
        self.assertAlmostEqual(parameters.backward_io.inp_noise, 0.456)
        self.assertAlmostEqual(parameters.update.desired_bl, 78)

    def test_construction_seed(self):
        """Test the construction seed leads to the same tile values."""
        rpu_config = self.get_rpu_config()

        # Set the seed.
        rpu_config.device.construction_seed = 10

        tile_1 = self.get_tile(3, 4, rpu_config)
        tile_2 = self.get_tile(3, 4, rpu_config)

        hidden_parameters_1 = tile_1.get_hidden_parameters()
        hidden_parameters_2 = tile_2.get_hidden_parameters()

        # Compare old and new hidden parameters tensors.
        for (field, old), (_, new) in zip(hidden_parameters_1.items(),
                                          hidden_parameters_2.items()):

            if 'weights' in field:
                # exclude weights as these are not governed by construction seed
                continue
            self.assertTrue(old.allclose(new))

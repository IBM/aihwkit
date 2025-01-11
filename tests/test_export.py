# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=too-many-locals, too-many-public-methods, no-member
"""Test for different utility functionality."""

from tempfile import NamedTemporaryFile
from numpy.testing import assert_array_almost_equal, assert_raises

from torch import Tensor

from aihwkit.nn.modules.container import AnalogSequential
from aihwkit.nn.modules.linear import AnalogLinear
from aihwkit.simulator.configs.configs import InferenceRPUConfig
from aihwkit.exceptions import TileError

from aihwkit.utils.export import (
    fusion_export,
    fusion_import,
    _fusion_load_csv,
    _fusion_get_csv_header,
)
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter

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
from .helpers.testcases import ParametrizedTestCase, AihwkitTestCase
from .helpers.tiles import Inference


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
    tiles=[Inference],
    biases=["digital"],
)
class FusionExportTest(ParametrizedTestCase):
    """Tests for fusion export."""

    def test_fusion_export(self) -> None:
        """Test export and loading via dict."""
        model = self.get_layer()
        weights, _ = model.get_weights()
        rpu_config = self.get_rpu_config()
        if not isinstance(rpu_config, InferenceRPUConfig):
            assert_raises(TileError(), fusion_export(model))

        data, params, state_dict = fusion_export(model)

        model.reset_parameters()
        new_weights, _ = model.get_weights()
        assert_raises(AssertionError, assert_array_almost_equal, weights, new_weights)

        new_model = fusion_import(data, model, params, state_dict)

        new_weights, _ = new_model.get_weights()
        assert_array_almost_equal(weights, new_weights)

    def test_fusion_export_csv(self) -> None:
        """Test export and loading via dict."""
        model = self.get_layer()
        weights, _ = model.get_weights()

        with NamedTemporaryFile() as file:
            _, params, state_dict = fusion_export(model, file_name=file.name)
            model.reset_parameters()
            new_model = fusion_import(file.name, model, params, state_dict)

        new_weights, _ = new_model.get_weights()
        assert_array_almost_equal(weights, new_weights)


class FusionExactTest(AihwkitTestCase):
    """Tests in more detail for a given matrix"""

    def test_fusion_load_csv_single_pair(self) -> None:
        """Test export and loading via dict."""
        g_converter = SinglePairConductanceConverter(g_max=1.0, g_min=0.0)

        weights = [Tensor([[1.0, 0.2], [0.3, -0.4]]), Tensor([[-0.5, 0.6], [0.7, -1.0]])]
        conductances = [
            [1.0, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.4],
            [0.0, 0.6, 0.7, 0.0, 0.5, 0.0, 0.0, 1.0],
        ]

        rpu_config = InferenceRPUConfig()
        rpu_config.mapping.weight_scaling_columnwise = 0.0

        model = AnalogSequential(
            AnalogLinear(2, 2, False, rpu_config), AnalogLinear(2, 2, False, rpu_config)
        )

        model[0].set_weights(weights[0])
        model[1].set_weights(weights[1])
        header = _fusion_get_csv_header(model)
        with NamedTemporaryFile() as file:
            fusion_export(model, file_name=file.name, g_converter=g_converter)
            conductance_data = _fusion_load_csv(file.name, header)

        for idx, (name, _) in enumerate(model.named_analog_layers()):
            self.assertTrue(name in conductance_data)
            assert_array_almost_equal(conductance_data[name], conductances[idx])

    def test_fusion_load_csv(self) -> None:
        """Test export and loading via dict."""
        weights = [Tensor([[1.0, 0.2], [0.3, -0.4]]), Tensor([[-0.5, 0.6], [0.7, -1.0]])]
        conductances = [[40.0, 8.0, 30.0, 40.0], [33.33333, 40.0, 28.0, 40.0]]

        rpu_config = InferenceRPUConfig()
        rpu_config.mapping.weight_scaling_columnwise = 0.0

        model = AnalogSequential(
            AnalogLinear(2, 2, False, rpu_config), AnalogLinear(2, 2, False, rpu_config)
        )

        model[0].set_weights(weights[0], apply_weight_scaling=False)
        model[1].set_weights(weights[1], apply_weight_scaling=False)
        header = _fusion_get_csv_header(model)
        with NamedTemporaryFile() as file:
            fusion_export(model, file_name=file.name)
            conductance_data = _fusion_load_csv(file.name, header)

        for idx, (name, _) in enumerate(model.named_analog_layers()):
            self.assertTrue(name in conductance_data)
            print(conductance_data[name])
            assert_array_almost_equal(conductance_data[name], conductances[idx], 5)

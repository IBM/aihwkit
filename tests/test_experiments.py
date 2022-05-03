# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Experiments."""

import os
from unittest import SkipTest

from aihwkit.cloud.converter.v1.training import BasicTrainingConverter
from aihwkit.nn.modules.base import AnalogModuleBase

from .helpers.decorators import parametrize_over_experiments
from .helpers.experiments import (
    FullyConnectedFashionMNIST, FullyConnectedFashionMNISTTikiTaka,
    LeNet5FashionMNIST,
    Vgg8SVHN, Vgg8SVHNTikiTaka
)
from .helpers.testcases import AihwkitTestCase


@parametrize_over_experiments([
    FullyConnectedFashionMNIST, FullyConnectedFashionMNISTTikiTaka,
    LeNet5FashionMNIST,
    Vgg8SVHN, Vgg8SVHNTikiTaka
])
class TestBasicTraining(AihwkitTestCase):
    """Test BasicTraining Experiment."""

    def setUp(self) -> None:
        if not os.getenv('TEST_DATASET'):
            raise SkipTest('TEST_DATASET not set')

    def test_conversion_roundtrip(self):
        """Test roundtrip conversion of examples."""
        experiment_original = self.get_experiment()

        # Convert to proto.
        converter = BasicTrainingConverter()
        experiment_proto = converter.to_proto(experiment_original)

        # Convert back to Experiment.
        experiment_converted = converter.from_proto(experiment_proto)

        # Assert over the Experiments properties.
        self.assertEqual(experiment_original.dataset, experiment_converted.dataset)
        self.assertEqual(experiment_original.batch_size, experiment_converted.batch_size)
        self.assertEqual(experiment_original.loss_function, experiment_converted.loss_function)
        self.assertEqual(experiment_original.epochs, experiment_converted.epochs)

        # Compare the models by hand, as direct equality comparison is not possible.
        # self.assertEqual(experiment_original.model, experiment_converted.model)
        self.assertEqual(len(list(experiment_original.model.children())),
                         len(list(experiment_converted.model.children())))
        for layer_a, layer_b in zip(experiment_original.model.children(),
                                    experiment_converted.model.children()):
            self.assertEqual(type(layer_a), type(layer_b))
            if isinstance(layer_a, AnalogModuleBase):
                self.assertEqual(type(layer_a.analog_tile.rpu_config.device),
                                 type(layer_b.analog_tile.rpu_config.device))

# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for Experiments."""

from os import getenv
from unittest import SkipTest
from aihwkit.cloud.converter.v1.training import BasicTrainingConverter
from aihwkit.nn.modules.base import AnalogLayerBase

from .helpers.decorators import parametrize_over_experiments
from .helpers.experiments import (
    FullyConnectedFashionMNIST,
    FullyConnectedFashionMNISTTikiTaka,
    LeNet5FashionMNIST,
    # Vgg8SVHN,
    # Vgg8SVHNTikiTaka,
)
from .helpers.testcases import AihwkitTestCase


@parametrize_over_experiments(
    [
        FullyConnectedFashionMNIST,
        FullyConnectedFashionMNISTTikiTaka,
        LeNet5FashionMNIST,
        # Vgg8SVHN,
        # Vgg8SVHNTikiTaka,
    ]
)
class TestBasicTraining(AihwkitTestCase):
    """Test BasicTraining Experiment."""

    def setUp(self) -> None:
        if not getenv("TEST_DATASET"):
            raise SkipTest("TEST_DATASET not set")

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
        self.assertEqual(
            len(list(experiment_original.model.children())),
            len(list(experiment_converted.model.children())),
        )
        for layer_a, layer_b in zip(
            experiment_original.model.children(), experiment_converted.model.children()
        ):
            self.assertEqual(type(layer_a), type(layer_b))
            if isinstance(layer_a, AnalogLayerBase):
                self.assertEqual(
                    type(next(layer_a.analog_tiles()).rpu_config.device),
                    type(next(layer_b.analog_tiles()).rpu_config.device),
                )

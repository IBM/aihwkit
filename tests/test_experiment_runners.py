# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for Experiment Runners."""

import os
from io import StringIO
from unittest import SkipTest, skipIf
from unittest.mock import patch

from torch import device as torch_device
from aihwkit.experiments.runners.local import LocalRunner

from .helpers.decorators import parametrize_over_experiments
from .helpers.experiments import (
    FullyConnectedFashionMNIST,
    FullyConnectedFashionMNISTTikiTaka,
    LeNet5FashionMNIST,
    Vgg8SVHN,
    Vgg8SVHNTikiTaka,
)
from .helpers.testcases import AihwkitTestCase, SKIP_CUDA_TESTS


@parametrize_over_experiments(
    [
        FullyConnectedFashionMNIST,
        FullyConnectedFashionMNISTTikiTaka,
        LeNet5FashionMNIST,
        Vgg8SVHN,
        Vgg8SVHNTikiTaka,
    ]
)
class TestLocalRunner(AihwkitTestCase):
    """Test LocalRunner."""

    def setUp(self) -> None:
        if not os.getenv("TEST_DATASET"):
            raise SkipTest("TEST_DATASET not set")

    def test_run_example_cpu(self):
        """Test running the example using a local runner."""
        training_experiment = self.get_experiment()
        local_runner = LocalRunner(device=torch_device("cpu"))
        with patch("sys.stdout", new=StringIO()) as captured_stdout:
            result = local_runner.run(training_experiment, max_elements_train=10, stdout=True)
        # Asserts over stdout.
        self.assertIn("Epoch: ", captured_stdout.getvalue())

        # Asserts over the returned results.
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["epoch"], 0)
        self.assertIn("train_loss", result[0])
        self.assertIn("accuracy", result[0])

    @skipIf(SKIP_CUDA_TESTS, "not compiled with CUDA support")
    def test_run_example_gpu(self):
        """Test running the example using a local runner."""
        training_experiment = self.get_experiment()
        local_runner = LocalRunner(device=torch_device("cuda"))

        with patch("sys.stdout", new=StringIO()) as captured_stdout:
            result = local_runner.run(training_experiment, max_elements_train=10, stdout=True)

        # Asserts over stdout.
        self.assertIn("Epoch: ", captured_stdout.getvalue())

        # Asserts over the returned results.
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["epoch"], 0)
        self.assertIn("train_loss", result[0])
        self.assertIn("accuracy", result[0])

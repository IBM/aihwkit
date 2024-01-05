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

"""Tests for the AIHW Composer cloud runner."""

import os
from unittest import SkipTest

from aihwkit.cloud.client.entities import CloudExperiment, CloudJobStatus
from aihwkit.cloud.client.utils import ClientConfiguration
from aihwkit.experiments.experiments.training import BasicTraining
from aihwkit.experiments.experiments.inferencing import BasicInferencing
from aihwkit.experiments.runners.cloud import CloudRunner

from .helpers.decorators import parametrize_over_experiments
from .helpers.experiments import FullyConnectedFashionMNIST
from .helpers.testcases import AihwkitTestCase


class CloudRunnerTest(AihwkitTestCase):
    """Tests for the AIHW Composer CloudRunner."""

    def setUp(self) -> None:
        config = ClientConfiguration()
        if not config.token:
            raise SkipTest("API token not found")

        self.api_url = config.url
        self.api_token = config.token

    def test_list_cloud_experiments(self):
        """Test listing cloud experiments."""
        cloud_runner = CloudRunner(self.api_url, self.api_token)
        cloud_experiments = cloud_runner.list_cloud_experiments()

        self.assertIsInstance(cloud_experiments, list)
        self.assertTrue(
            all(isinstance(experiment, CloudExperiment) for experiment in cloud_experiments)
        )

    def test_get_cloud_experiment(self):
        """Test getting an experiment from an execution."""
        cloud_runner = CloudRunner(self.api_url, self.api_token)
        experiments = cloud_runner.list_cloud_experiments()
        if len(experiments) == 0:
            raise SkipTest("No executions found")

        listed_experiment = experiments[0]
        cloud_experiment = cloud_runner.get_cloud_experiment(listed_experiment.id_)

        self.assertIsInstance(cloud_experiment, CloudExperiment)
        self.assertEqual(cloud_experiment.id_, listed_experiment.id_)

    def test_get_cloud_experiment_experiment(self):
        """Test getting an experiment from a cloud experiment."""
        cloud_runner = CloudRunner(self.api_url, self.api_token)
        experiments = cloud_runner.list_cloud_experiments()
        if len(experiments) == 0:
            raise SkipTest("No executions found")

        cloud_experiment = experiments[-1].get_experiment()
        if "BasicInferencing" in str(cloud_experiment):
            self.assertIsInstance(cloud_experiment, BasicInferencing)
        else:
            self.assertIsInstance(cloud_experiment, BasicTraining)

    def test_get_cloud_experiment_result(self):
        """Test getting the result from a cloud experiment."""
        cloud_runner = CloudRunner(self.api_url, self.api_token)
        cloud_experiments = cloud_runner.list_cloud_experiments()
        if len(cloud_experiments) == 0:
            raise SkipTest("No executions found")

        for experiment in cloud_experiments:
            if experiment.job.status == CloudJobStatus.COMPLETED:
                output = experiment.get_result()
                self.assertIsInstance(output, list)
                break
        else:
            raise SkipTest("No completed executions found")


@parametrize_over_experiments([FullyConnectedFashionMNIST])
class CloudRunnerCreateTest(AihwkitTestCase):
    """Tests for the AIHW Composer CloudRunner involving experiment creation."""

    def setUp(self) -> None:
        if not os.getenv("TEST_CREATE"):
            raise SkipTest("TEST_CREATE not set")

        config = ClientConfiguration()
        if not config.token:
            raise SkipTest("API token not found")

        self.api_url = config.url
        self.api_token = config.token

    def test_create_experiment(self):
        """Test creating a new experiment."""
        cloud_runner = CloudRunner(self.api_url, self.api_token)
        cloud_experiment = self.get_experiment()

        api_experiment = cloud_runner.run(cloud_experiment, device="cpu")
        self.assertIsInstance(api_experiment, CloudExperiment)

        # Assert the experiment shows in the list.
        api_experiments = cloud_runner.list_cloud_experiments()
        api_experiment_ids = [item.id_ for item in api_experiments]
        self.assertIn(api_experiment.id_, api_experiment_ids)

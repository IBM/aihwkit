# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for the AIHW Composer API client."""

import os
from unittest import SkipTest

from aihwkit.cloud.client.entities import CloudExperiment, CloudJobStatus
from aihwkit.cloud.client.session import ApiSession
from aihwkit.cloud.client.utils import ClientConfiguration
from aihwkit.cloud.client.v1.api_client import ApiClient

from .helpers.decorators import parametrize_over_experiments
from .helpers.experiments import FullyConnectedFashionMNIST
from .helpers.testcases import AihwkitTestCase


class ApiClientTest(AihwkitTestCase):
    """Tests for the AIHW Composer API client."""

    def setUp(self) -> None:
        config = ClientConfiguration()
        if not config.token:
            raise SkipTest("API token not found")

        self.session = ApiSession(config.url, config.token)
        self.api_client = ApiClient(self.session)

    def test_experiments_list(self):
        """Test listing experiments."""
        experiments = self.api_client.experiments_list()

        self.assertIsInstance(experiments, list)
        self.assertTrue(all(isinstance(experiment, CloudExperiment) for experiment in experiments))

    def test_experiment_get(self):
        """Test getting an experiment."""
        experiments = self.api_client.experiments_list()

        if len(experiments) == 0:
            raise SkipTest("No experiments found")

        experiment = experiments[-1]
        fetched_experiment = self.api_client.experiment_get(experiment.id_)
        self.assertEqual(fetched_experiment.id_, experiment.id_)

    def test_experiment_input(self):
        """Test getting the input of an experiment."""
        experiments = self.api_client.experiments_list()
        if len(experiments) == 0:
            raise SkipTest("No experiments found")

        experiment = experiments[-1]
        input_ = self.api_client.input_get(experiment.input_id)
        self.assertIsInstance(input_, bytes)

    def test_experiment_output(self):
        """Test getting the output of an experiment."""
        experiments = self.api_client.experiments_list()
        if len(experiments) == 0:
            raise SkipTest("No experiments found")

        for experiment in experiments:
            job_ = self.api_client.job_get(experiment.job.id_)
            if job_.status == CloudJobStatus.COMPLETED:
                input_ = self.api_client.output_get(job_.output_id)
                self.assertIsInstance(input_, bytes)
                break


@parametrize_over_experiments([FullyConnectedFashionMNIST])
class ApiClientCreateTest(AihwkitTestCase):
    """Tests for the AIHW Composer API client involving experiment creation."""

    def setUp(self) -> None:
        if not os.getenv("TEST_CREATE"):
            raise SkipTest("TEST_CREATE not set")

        config = ClientConfiguration()
        if not config.token:
            raise SkipTest("API token not found")

        self.session = ApiSession(config.url, config.token)
        self.api_client = ApiClient(self.session)

    def test_create_experiment(self):
        """Test creating a new experiment."""
        experiment = self.get_experiment()
        api_experiment = self.api_client.experiment_create(
            experiment, name="test_create_experiment", device="cpu"
        )
        self.assertIsInstance(api_experiment, CloudExperiment)

        # Assert the experiment shows in the list.
        api_experiments = self.api_client.experiments_list()
        api_experiment_ids = [item.id_ for item in api_experiments]
        self.assertIn(api_experiment.id_, api_experiment_ids)

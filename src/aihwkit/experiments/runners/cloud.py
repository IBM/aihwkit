# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Base class for an Experiment Runner."""

from typing import Any, List, Optional

from aihwkit.cloud.client.entities import CloudExperiment
from aihwkit.cloud.client.exceptions import CredentialsError
from aihwkit.cloud.client.session import ApiSession
from aihwkit.cloud.client.utils import ClientConfiguration
from aihwkit.cloud.client.v1.api_client import ApiClient
from aihwkit.experiments import BasicTraining
from aihwkit.experiments.runners.base import Runner


class CloudRunner(Runner):
    """Runner that executes Experiments in the AIHW Composer cloud.

    Class that allows executing Experiments in the cloud.
    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self, api_url: Optional[str] = None, api_token: Optional[str] = None, verify: bool = True
    ):
        """Create a new ``CloudRunner``.

        Note:
            If no ``api_token`` or ``api_url`` is provided, this class will
            attempt to read them from the local configuration file (by default,
            at ``~/.config/aihwkit/aihwkit.conf`` or  environment variables
            (``AIHW_API_TOKEN``).

        Args:
            api_url: the URL of the AIHW Composer API.
            api_token: the API token for authentication.
            verify: if ``False``, disable the remote server TLS verification.

        Raises:
            CredentialsError: if no credentials could be found.
        """
        # Attempt to load credentials if not present.
        if not api_url or not api_token:
            config = ClientConfiguration()
            api_url = api_url or config.url
            api_token = api_token or config.token

            if not api_url or not api_token:
                raise CredentialsError("No credentials could be found")

        self.api_url = api_url
        self.api_token = api_token

        # Authenticate.
        self.session = ApiSession(self.api_url, self.api_token, verify)
        self.api_client = ApiClient(self.session)

    def get_cloud_experiment(self, id_: str) -> CloudExperiment:
        """Return a single cloud experiment by id.

        Args:
            id_: the identifier of the cloud experiment.

        Returns:
            A ``CloudExperiment``.
        """
        return self.api_client.experiment_get(id_)

    def list_cloud_experiments(self) -> List[CloudExperiment]:
        """Return a list of cloud experiments.

        Returns:
            A list of ``CloudExperiments``.
        """
        return self.api_client.experiments_list()

    def run(  # type: ignore[override]
        self, experiment: BasicTraining, name: str = "", device: str = "gpu", **_: Any
    ) -> CloudExperiment:
        """Run a single Experiment.

        Starts the execution of an Experiment in the cloud. Upon successful
        invocation, this method will return a ``CloudExperiment`` object that
        can be used for inspecting the status of the remote execution.

        Note:
            Please be aware that the ``experiment`` is subjected to some
            constraints compared to local running of experiments.

        Args:
            experiment: the experiment to be executed.
            name: an optional name for the experiment.
            device: the desired device.
            _: extra arguments for the runner.

        Returns:
            A ``CloudExperiment`` which represents the remote experiment.
        """
        # pylint: disable=arguments-differ
        # Generate an experiment name if not given.
        if not name:
            name = "aihwkit cloud experiment ({}, {} layers)".format(
                experiment.dataset.__name__, len(experiment.model)
            )

        return self.api_client.experiment_create(experiment, name, device)

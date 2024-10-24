# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""API client the AIHW Composer API."""

from typing import List, Dict

from aihwkit.cloud.client.entities import CloudExperiment, CloudJob
from aihwkit.cloud.client.exceptions import ApiResponseError, CredentialsError
from aihwkit.cloud.client.session import ObjectStorageSession, ApiSession
from aihwkit.cloud.client.v1.parsers import ExperimentParser, GeneralParser
from aihwkit.cloud.client.v1.stubs import ExperimentStub, InputStub, JobStub, LoginStub, OutputStub
from aihwkit.cloud.converter.v1.inferencing import BasicInferencingConverter
from aihwkit.experiments.experiments.inferencing import BasicInferencing


class InferenceApiClient:
    """API client the AIHW Composer API.

    Client for interfacing with the AIHW Composer API. Upon instantiation, the
    client will attempt to login (using the details from the ``session``).

    The functionality is provided by the stubs that are created during the
    instantiation, providing an interface that which mimic the REST API
    endpoints.
    """

    def __init__(self, session: ApiSession):
        """Create a new ``ApiClient``.

        Args:
            session: the request session to be used in the client.
        """
        self.session = session
        self.object_storage_session = ObjectStorageSession()

        # Create the helpers.
        self.converter = BasicInferencingConverter()

        # Create the stubs.
        self.experiments = ExperimentStub(self.session)
        self.inputs = InputStub(self.session)
        self.outputs = OutputStub(self.session)
        self.jobs = JobStub(self.session)
        self.login_ = LoginStub(self.session)

        # Automatically login.
        self.login()

    def login(self) -> None:
        """Login into the application.

        Raises:
            CredentialsError: if the credentials are not valid.
            ApiResponseError: if the request was not successful.
        """
        try:
            response = self.login_.post({"token": self.session.api_token})
        except ApiResponseError as ex:
            if ex.response.status_code == 400:
                try:
                    json_response = ex.response.json()
                except Exception:  # pylint: disable=broad-except
                    json_response = {}
                raise CredentialsError(
                    "Error while trying to log in: {}".format(
                        json_response.get("message", "unknown")
                    )
                ) from ex
            raise

        jwt_token = GeneralParser.parse_login(response)
        self.session.update_jwt_token(jwt_token)

    def experiments_list(self) -> List[CloudExperiment]:
        """Return a list of experiments."""
        response = self.experiments.list()

        return [ExperimentParser.parse_experiment(experiment, self) for experiment in response]

    def experiment_create(
        self,
        input_: BasicInferencing,
        analog_info: Dict,
        noise_model_info: Dict,
        name: str,
        device: str = "gpu",
    ) -> CloudExperiment:
        """Create a new experiment, queuing its execution.

        Args:
            input_: the experiment to be executed.
            analog_info: analog information.
            noise_model_info: noise information.
            name: the name of the experiment.
            device: the desired device.

        Returns:
            A ``CloudExperiment``.
        """
        # Prepare the API data.
        # debug: print('input_: ', input_)
        payload = self.converter.to_proto(input_, analog_info, noise_model_info).SerializeToString()
        # payload = self.converter.to_proto(input_)
        # debug: print('payload after convert to proto and serialize to string: \n', payload)

        # Create the experiment.
        response = self.experiments.post({"name": name, "category": "inference"})
        # debug: print('response from experiments.post: ', response)
        experiment = ExperimentParser.parse_experiment(response, self)

        # Create the input.
        response = self.inputs.post({"experiment": experiment.id_, "device": device})
        object_storage_url = response["url"]
        # debug: print ('url: \n', object_storage_url)
        _ = self.object_storage_session.put(url=object_storage_url, data=payload)

        # Create the job.
        response = self.jobs.post({"device": device, "experiment": experiment.id_})
        # debug: print('response from jobs.post: ', response)
        experiment.job = ExperimentParser.parse_job(response)
        # debug: print('In experiment_create: experiment.job: \n', experiment.job)

        return experiment

    def experiment_get(self, experiment_id: str) -> CloudExperiment:
        """Get an existing job by id.

        Args:
            experiment_id: id of the experiment.

        Returns:
            A `CloudExperiment` with the specified id.
        """
        response = self.experiments.get(experiment_id)

        return ExperimentParser.parse_experiment(response, self)

    def job_get(self, job_id: str) -> CloudJob:
        """Get an existing job by id.

        Args:
            job_id: id of the job.

        Returns:
            A `CloudJob` with the specified id.
        """
        response = self.jobs.get(job_id)

        return ExperimentParser.parse_job(response)

    def input_get(self, input_id: str) -> bytes:
        """Get an existing input by id.

        Args:
            input_id: id of the input.

        Returns:
            The input with the specified id, in protobuf format.
        """
        response = self.inputs.get(input_id)
        object_storage_url = response["url"]

        object_storage_response = self.object_storage_session.get(object_storage_url)

        return object_storage_response.content

    def output_get(self, output_id: str) -> bytes:
        """Get an existing output by id.

        Args:
            output_id: id of the output.

        Returns:
            The output with the specified id, in protobuf format.
        """
        response = self.outputs.get(output_id)
        object_storage_url = response["url"]

        object_storage_response = self.object_storage_session.get(object_storage_url)

        return object_storage_response.content

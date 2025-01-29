# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Parsers for the AIHW Composer API."""

from datetime import datetime, timezone
from typing import Any, Dict

from aihwkit.cloud.client.entities import (
    CloudExperiment,
    CloudExperimentCategory,
    CloudJobStatus,
    CloudJob,
)
from aihwkit.cloud.client.exceptions import InvalidResponseFieldError


class ExperimentParser:
    """Parser for Experiment API responses."""

    @staticmethod
    def parse_experiment(api_response: Dict, api_client: Any) -> CloudExperiment:
        """Return a CloudExperiment from an API response.

        Args:
            api_response: the response from the API.
            api_client: the client to be used in API requests.

        Returns:
            A `CloudExperiment` based on the response. Some of the fields might
            not be populated if they are not present in the response.
        """
        experiment = CloudExperiment(
            _api_client=api_client,
            id_=api_response["id"],
            name=api_response["name"],
            category=ExperimentParser.parse_experiment_category(api_response),
            created_at=ExperimentParser.parse_date_string(api_response["createdAt"]),
            input_id=None,
            job=None,
        )

        # debug
        # print('api_response: ', api_response)

        if api_response.get("input"):
            if api_response.get("input", {}).get("id"):
                experiment.input_id = api_response["input"]["id"]

        if api_response.get("job", None):
            experiment.job = ExperimentParser.parse_job(api_response["job"])

        return experiment

    @staticmethod
    def parse_job(api_response: Dict) -> CloudJob:
        """Return an CloudJob from an API response.

        Args:
            api_response: the response from the API.

        Returns:
            A `CloudJob` based on the response. Some of the fields might
            not be populated if they are not present in the response.
        """
        job = CloudJob(
            id_=api_response["id"],
            output_id=None,
            status=ExperimentParser.parse_experiment_status(api_response),
        )

        if api_response.get("output", None):
            job.output_id = api_response["output"]

        return job

    @staticmethod
    def parse_experiment_status(api_response: Dict) -> CloudJobStatus:
        """Return an Experiment status from an API response.

        Args:
            api_response: the response from the API.

        Returns:
            A value from the `CloudJobStatus` enum.

        Raises:
            InvalidResponseFieldError: if the API response contains an
                unrecognized status code.
        """
        job_status = api_response["status"]

        if job_status in ("waiting", "validating", "validated"):
            return CloudJobStatus.WAITING
        if job_status in ("running",):
            return CloudJobStatus.RUNNING
        if job_status in ("failed", "cancelled"):
            return CloudJobStatus.FAILED
        if job_status in ("completed",):
            return CloudJobStatus.COMPLETED

        raise InvalidResponseFieldError("Unsupported job status: {}".format(job_status))

    @staticmethod
    def parse_experiment_category(api_response: Dict) -> CloudExperimentCategory:
        """Return an Experiment category from an API response.

        Args:
            api_response: the response from the API.

        Returns:
            A value from the `CloudExperimentCategory` enum.

        Raises:
            InvalidResponseFieldError: if the API response contains an
                unrecognized category.
        """
        job_category = api_response["category"]

        if job_category in ("train", "trainweb"):
            return CloudExperimentCategory.BASIC_TRAINING

        if job_category in ("inference", "inferenceweb"):
            return CloudExperimentCategory.BASIC_INFERENCE

        raise InvalidResponseFieldError("Unsupported experiment category: {}".format(job_category))

    @staticmethod
    def parse_date_string(date_string: str) -> datetime:
        """Return a datetime from a date string.

        Args:
            date_string: the date string from the API.

        Returns:
            A value from the `CloudExperimentCategory` enum.
        """
        tmp_datetime = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S.%fZ")

        return tmp_datetime.replace(tzinfo=timezone.utc)


class GeneralParser:
    """Parser for generic responses."""

    # pylint: disable=too-few-public-methods

    @staticmethod
    def parse_login(api_response: Dict) -> str:
        """Return the jwt token from an API response.

        Args:
            api_response: the response from the API.

        Returns:
            A string with the jwt token.
        """
        return api_response["jwt"]

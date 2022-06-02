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

"""Data classes for the AIHW Composer API."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from aihwkit.cloud.client.exceptions import ExperimentStatusError
from aihwkit.cloud.converter.definitions.input_file_pb2 import TrainingInput
from aihwkit.cloud.converter.definitions.output_file_pb2 import TrainingOutput
from aihwkit.cloud.converter.v1.training import BasicTrainingConverter, BasicTrainingResultConverter
from aihwkit.experiments import BasicTraining


class CloudJobStatus(Enum):
    """Status for a CloudJob."""

    UNKNOWN = 0
    WAITING = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4


class CloudExperimentCategory(Enum):
    """Category of a CloudExperiment."""

    BASIC_TRAINING = 1


@dataclass
class CloudJob:
    """Representation of an API CloudJob."""

    id_: str
    output_id: Optional[str] = field(repr=False)
    status: CloudJobStatus = CloudJobStatus.UNKNOWN


@dataclass
class CloudExperiment:
    """Representation of an API Experiment."""

    _api_client: Any = field(repr=False)
    id_: str
    name: str
    category: CloudExperimentCategory = field(repr=False)
    created_at: datetime = field(repr=False)
    input_id: Optional[str] = field(repr=False)
    job: Optional[CloudJob] = field(repr=False)

    def get_experiment(self) -> BasicTraining:
        """Return a data Experiment.

        Returns:
            The experiment.

        Raises:
            ExperimentStatusError: if the Experiment is not in a valid status.
        """
        if self.status() == CloudJobStatus.UNKNOWN:
            raise ExperimentStatusError('Experiment input is not available')

        input_ = self._api_client.input_get(self.input_id)

        input_proto = TrainingInput()
        input_proto.ParseFromString(input_)

        converter = BasicTrainingConverter()
        return converter.from_proto(input_proto)

    def get_result(self) -> list:
        """Return the result of an Experiment.

        Returns:
            The experiment result.

        Raises:
            ExperimentStatusError: if the Experiment is not completed.
        """
        if self.status() != CloudJobStatus.COMPLETED:
            raise ExperimentStatusError('Output cannot be retrieved unless the '
                                        'experiment is completed')

        # Fetch the protobuf output.
        output_ = self._api_client.output_get(self.job.output_id)  # type: ignore

        # Convert from protobuf.
        training_output = TrainingOutput()
        training_output.ParseFromString(output_)
        converter = BasicTrainingResultConverter()
        output = converter.from_proto(training_output)

        return output['epochs']

    def status(self) -> CloudJobStatus:
        """Return the status of the experiment."""
        # Populate the job if not present.
        if not self.job:
            tmp_experiment = self._api_client.experiment_get(self.id_)
            self.input_id = tmp_experiment.input_id
            self.job = tmp_experiment.job

        # Fallback for Experiments without Job.
        if not self.job:
            return CloudJobStatus.UNKNOWN

        # Avoid refreshing statuses of Jobs in end states.
        if self.job.status == CloudJobStatus.COMPLETED and self.job.output_id:
            return CloudJobStatus.COMPLETED
        if self.job.status == CloudJobStatus.FAILED:
            return CloudJobStatus.FAILED

        # Refresh the status.
        self.job = self._api_client.job_get(self.job.id_)
        return self.job.status  # type: ignore

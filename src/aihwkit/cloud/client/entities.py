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

"""Data classes for the AIHW Composer API."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from aihwkit.cloud.client.exceptions import ExperimentStatusError

# pylint: disable=no-name-in-module,import-error
from aihwkit.cloud.converter.definitions.input_file_pb2 import (  # type: ignore[attr-defined]
    TrainingInput,
)
from aihwkit.cloud.converter.definitions.i_input_file_pb2 import (  # type: ignore[attr-defined]
    InferenceInput,
)
from aihwkit.cloud.converter.definitions.output_file_pb2 import (  # type: ignore[attr-defined]
    TrainingOutput,
)
from aihwkit.cloud.converter.definitions.i_output_file_pb2 import (  # type: ignore[attr-defined]
    InferencingOutput,
)
from aihwkit.cloud.converter.v1.training import BasicTrainingConverter, BasicTrainingResultConverter
from aihwkit.cloud.converter.v1.inferencing import (
    BasicInferencingConverter,
    BasicInferencingResultConverter,
)

# from aihwkit.experiments import BasicTraining, BasicInferencing


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
    BASIC_INFERENCE = 2


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

    def get_experiment(self) -> Any:
        """Return a data Experiment.

        Returns:
            The experiment.

        Raises:
            ExperimentStatusError: if the Experiment is not in a valid status.
        """
        if self.status() == CloudJobStatus.UNKNOWN:
            raise ExperimentStatusError("Experiment input is not available")

        input_ = self._api_client.input_get(self.input_id)

        if "InferenceRPUConfig" in str(input_):
            input_proto = InferenceInput()
            input_proto.ParseFromString(input_)
            proto = BasicInferencingConverter().from_proto(input_proto)
        else:
            input_proto = TrainingInput()
            input_proto.ParseFromString(input_)
            proto = BasicTrainingConverter().from_proto(input_proto)

        return proto

    def get_result(self) -> list:
        """Return the result of an Experiment.

        Returns:
            The experiment result.

        Raises:
            ExperimentStatusError: if the Experiment is not completed.
        """
        if self.status() != CloudJobStatus.COMPLETED:
            raise ExperimentStatusError(
                "Output cannot be retrieved unless the experiment is completed"
            )

        if self.category == CloudExperimentCategory.BASIC_TRAINING:
            # Fetch the protobuf output.
            output_ = self._api_client.output_get(self.job.output_id)  # type: ignore
            # Convert from protobuf.
            training_output = TrainingOutput()
            training_output.ParseFromString(output_)
            converter = BasicTrainingResultConverter()
            output = converter.from_proto(training_output)
            result = output["epochs"]
        if self.category == CloudExperimentCategory.BASIC_INFERENCE:
            output_ = self._api_client.output_get(self.job.output_id)  # type: ignore
            # Convert from protobuf.
            inferencing_output = InferencingOutput()
            inferencing_output.ParseFromString(output_)
            iconverter = BasicInferencingResultConverter()
            i_output = iconverter.result_from_proto(inferencing_output)
            result = i_output
        return result

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

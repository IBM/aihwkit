# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Runner that executes Experiments locally."""

from typing import Dict, Optional

from torch import device as torch_device
from torchvision.datasets import FashionMNIST, SVHN

from aihwkit.experiments.experiments.base import Signals
from aihwkit.experiments.runners.base import Runner
from aihwkit.experiments.experiments.inferencing import BasicInferencing
from aihwkit.experiments.runners.i_metrics import InferenceLocalMetric


class InferenceLocalRunner(Runner):
    """Runner that executes Experiments locally.

    Class that allows executing Experiments locally.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, device: Optional[torch_device] = None):
        """Create a new ``InferenceLocalRunner``.

        Args:
            device: the device where the model will be running on.
        """
        self.device = device

    def run(  # type: ignore[override]
        self,
        experiment: BasicInferencing,
        max_elements: int = 0,
        dataset_root: str = "/tmp/datasets",
        stdout: bool = False,
    ) -> Dict:
        """Run a single Experiment.

        Executes an experiment locally, in the device specified by
        ``self.device``, optionally printing information to stdout.

        Note:
            If using a dataset different than ``FashionMNIST`` or ``SVHN``,
            the runner assumes that the files for the dataset are downloaded at
            ``dataset_root``. For those two datasets, the downloading will
            take place automatically if the files are not present.

        Args:
            experiment: the experiment to be executed.
            max_elements: limit on the amount of samples to use from
                the dataset. If ``0``, no limit is applied.
            dataset_root: path for the dataset files.
            stdout: enable printing to stdout during the execution of the
                experiment.

        Returns:

            A dictionary with the inference results.
        """
        # pylint: disable=arguments-differ

        # Setup the metric helper for the experiment.
        metric = InferenceLocalMetric(stdout=stdout)
        experiment.clear_hooks()
        experiment.add_hook(Signals.INFERENCE_REPEAT_START, metric.receive_repeat_start)
        experiment.add_hook(Signals.INFERENCE_REPEAT_END, metric.receive_repeat_end)

        # Download the FashionMNIST or SVHN dataset if needed.
        if experiment.dataset == FashionMNIST:
            _ = experiment.dataset(dataset_root, download=True)
        elif experiment.dataset == SVHN:
            _ = experiment.dataset(dataset_root, download=True, split="train")
            _ = experiment.dataset(dataset_root, download=True, split="test")

        # Invoke the inference step
        return experiment.run(max_elements, dataset_root, self.device)

# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Runner that executes Experiments locally."""

from typing import Dict, List, Optional

from torch import device as torch_device
from torchvision.datasets import FashionMNIST, SVHN

from aihwkit.experiments.experiments.base import Signals
from aihwkit.experiments.runners.base import Runner
from aihwkit.experiments.experiments.training import BasicTraining
from aihwkit.experiments.runners.metrics import LocalMetric


class LocalRunner(Runner):
    """Runner that executes Experiments locally.

    Class that allows executing Experiments locally.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, device: Optional[torch_device] = None):
        """Create a new ``LocalRunner``.

        Args:
            device: the device where the model will be moved to.
        """
        self.device = device

    def run(  # type: ignore[override]
        self,
        experiment: BasicTraining,
        max_elements_train: int = 0,
        dataset_root: str = "/tmp/datasets",
        stdout: bool = False,
    ) -> List[Dict]:
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
            max_elements_train: limit on the amount of samples to use from
                the dataset. If ``0``, no limit is applied.
            dataset_root: path for the dataset files.
            stdout: enable printing to stdout during the execution of the
                experiment.

        Returns:
            A list of dictionaries with information about each
            epoch.
        """
        # pylint: disable=arguments-differ

        # Setup the metric helper for the experiment.
        metric = LocalMetric(stdout=stdout)
        experiment.clear_hooks()
        experiment.add_hook(Signals.EPOCH_START, metric.receive_epoch_start)
        experiment.add_hook(Signals.EPOCH_END, metric.receive_epoch_end)
        experiment.add_hook(Signals.TRAIN_EPOCH_END, metric.receive_train_epoch_end)
        experiment.add_hook(Signals.TRAIN_EPOCH_BATCH_END, metric.receive_train_epoch_batch_end)
        experiment.add_hook(
            Signals.VALIDATION_EPOCH_BATCH_END, metric.receive_validation_epoch_batch_end
        )
        experiment.add_hook(Signals.VALIDATION_EPOCH_END, metric.receive_validation_epoch_end)

        # Download the dataset if needed.
        if experiment.dataset == FashionMNIST:
            _ = experiment.dataset(dataset_root, download=True)
        elif experiment.dataset == SVHN:
            _ = experiment.dataset(dataset_root, download=True, split="train")
            _ = experiment.dataset(dataset_root, download=True, split="test")

        return experiment.run(max_elements_train, dataset_root, self.device)

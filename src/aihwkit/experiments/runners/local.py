# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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

    def __init__(self,
                 device: Optional[torch_device] = None):
        """Create a new ``LocalRunner``.

        Args:
            device: the device where the model will be moved to.
        """
        self.device = device

    def run(  # type: ignore[override]
            self,
            experiment: BasicTraining,
            max_elements_train: int = 0,
            dataset_root: str = '/tmp/datasets',
            stdout: bool = True
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
            A list of dictionaries with information about each epoch.
        """
        # pylint: disable=arguments-differ

        # Setup the metric helper for the experiment.
        metric = LocalMetric(stdout=stdout)
        experiment.clear_hooks()
        experiment.add_hook(Signals.EPOCH_START, metric.receive_epoch_start)
        experiment.add_hook(Signals.EPOCH_END, metric.receive_epoch_end)
        experiment.add_hook(Signals.TRAIN_EPOCH_END, metric.receive_train_epoch_end)
        experiment.add_hook(Signals.TRAIN_EPOCH_BATCH_END, metric.receive_train_epoch_batch_end)
        experiment.add_hook(Signals.VALIDATION_EPOCH_BATCH_END,
                            metric.receive_validation_epoch_batch_end)
        experiment.add_hook(Signals.VALIDATION_EPOCH_END, metric.receive_validation_epoch_end)

        # Download the dataset if needed.
        if experiment.dataset == FashionMNIST:
            _ = experiment.dataset(dataset_root, download=True)
        elif experiment.dataset == SVHN:
            _ = experiment.dataset(dataset_root, download=True, split='train')
            _ = experiment.dataset(dataset_root, download=True, split='test')

        # Move the model to the device.
        if self.device:
            if self.device.type == 'cuda':
                experiment.model = experiment.model.to(self.device)

        # Build the objects needed for training.
        training_loader, validation_loader = experiment.get_data_loaders(
            experiment.dataset, experiment.batch_size,
            max_elements_train=max_elements_train,
            dataset_root=dataset_root
        )
        optimizer = experiment.get_optimizer(experiment.learning_rate, experiment.model)

        experiment.results = experiment.train(training_loader,
                                              validation_loader,
                                              experiment.model,
                                              optimizer,
                                              experiment.loss_function(),
                                              experiment.epochs,
                                              self.device)
        return experiment.results

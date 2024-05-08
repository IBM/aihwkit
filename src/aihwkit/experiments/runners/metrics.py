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

"""Helper for retrieving Metrics of an Experiment."""

from datetime import datetime
from typing import Dict, List


class LocalMetric:
    """Metric for local experiments.

    Metric for local execution of experiments. Output to stdout can be
    controlled by the ``stdout`` parameter to the constructor.
    """

    def __init__(self, stdout: bool = True) -> None:
        self.epochs: List[Dict] = []
        self.current_epoch: Dict = {}
        self.stdout = stdout

    def receive_epoch_start(self, epoch: int) -> None:
        """Hook for `EPOCH_START`."""
        self.current_epoch = {
            "number": epoch,
            "start_time": datetime.utcnow(),
            "total_loss": 0,
            "training_images": 0,
            "validation_images": 0,
            "validation_correct": 0,
            "validation_loss": 0,
        }

    def receive_train_epoch_batch_end(self, total: int, train_loss: float) -> None:
        """Hook for `TRAIN_EPOCH_START`."""
        self.current_epoch["training_images"] += total
        self.current_epoch["total_loss"] += train_loss

    def receive_validation_epoch_batch_end(
        self, total: int, correct: int, validation_loss: float
    ) -> None:
        """Hook for `VALIDATION_EPOCH_BATCH_END`."""
        self.current_epoch["validation_images"] += total
        self.current_epoch["validation_correct"] += int(correct)
        self.current_epoch["validation_loss"] += validation_loss

    def receive_train_epoch_end(self) -> None:
        """Hook for `TRAIN_EPOCH_END`."""
        if not self.stdout:
            return

        print(
            "Epoch: {}, loss: {:.8f}".format(
                self.current_epoch["number"],
                self.current_epoch["total_loss"] / self.current_epoch["training_images"],
            )
        )

    def receive_validation_epoch_end(self) -> None:
        """Hook for `VALIDATION_EPOCH_END`."""
        if not self.stdout:
            return

        print(
            "Number of images: {}, accuracy: {:.6%}, validation loss: {:.8f}".format(
                self.current_epoch["validation_images"],
                self.current_epoch["validation_correct"] / self.current_epoch["validation_images"],
                self.current_epoch["validation_loss"] / self.current_epoch["validation_images"],
            )
        )

    def receive_epoch_end(self) -> Dict:
        """Hook for `EPOCH_END`."""
        end_time = datetime.utcnow()
        time_epoch = (end_time - self.current_epoch["start_time"]).total_seconds()
        if self.stdout:
            print("Time for epoch {}: {:.4}s".format(self.current_epoch["number"], time_epoch))
        self.epochs.append(self.current_epoch)

        return {
            "epoch": self.current_epoch["number"],
            "time_epoch": time_epoch,
            "accuracy": (
                self.current_epoch["validation_correct"] / self.current_epoch["validation_images"]
            ),
            "train_loss": (
                self.current_epoch["total_loss"] / self.current_epoch["training_images"]
            ),
            "valid_loss": (
                self.current_epoch["validation_loss"] / self.current_epoch["validation_images"]
            ),
        }

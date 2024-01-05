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
import json
from typing import Dict


class InferenceLocalMetric:
    """Metric used by the InferenceWorker Runner."""

    def __init__(self, stdout: bool = False) -> None:
        self.current_repeat: Dict = {}
        self.time_init = datetime.utcnow()
        self.stdout = stdout

    def receive_repeat_start(self, repeat: int) -> None:
        """Hook for `INFERENCE_REPEAT_START`."""
        self.current_repeat = {"number": repeat, "inference_results": []}

    def receive_repeat_end(
        self,
        t_inference_array: list,
        avg_acc_arr: list,
        std_acc_arr: list,
        avg_err_arr: list,
        avg_loss_arr: list,
        inference_repeats: int,
    ) -> Dict:
        """Hook for `INFERENCE_REPEAT_END`."""

        inf_results = []
        n_inference = len(t_inference_array)

        # The input are the arrays of avg accuracy, avg error and avg loss.
        # Create the dict entry for the items in the arrays.
        for i in range(n_inference):
            new_dict = {
                "t_inference": t_inference_array[i],
                "avg_accuracy": avg_acc_arr[i],
                "std_accuracy": std_acc_arr[i],
                "avg_error": avg_err_arr[i],
                "avg_loss": avg_loss_arr[i],
            }
            inf_results.append(new_dict)

        repeat = self.current_repeat["number"] + 1
        time_elapsed = (datetime.utcnow() - self.time_init).total_seconds()
        is_partial = bool(repeat < inference_repeats)
        partial = {
            "inference_runs": {
                "inference_repeat": repeat,
                "is_partial": is_partial,
                "time_elapsed": time_elapsed,
                "inference_results": inf_results,
            }
        }

        if self.stdout:
            print("{}".format(json.dumps(partial)))

        # Return the partial.
        return partial

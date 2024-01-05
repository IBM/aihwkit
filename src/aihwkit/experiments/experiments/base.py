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

"""Base class for an Experiment."""

from enum import Enum
from typing import Any, Callable, Dict, Optional


class Signals(Enum):
    """Signals emitted by an Experiment."""

    EXPERIMENT_START = 1
    EXPERIMENT_END = 2
    EPOCH_START = 10
    EPOCH_END = 11
    TRAIN_EPOCH_START = 20
    TRAIN_EPOCH_END = 21
    TRAIN_EPOCH_BATCH_START = 22
    TRAIN_EPOCH_BATCH_END = 23
    VALIDATION_EPOCH_START = 30
    VALIDATION_EPOCH_END = 31
    VALIDATION_EPOCH_BATCH_START = 32
    VALIDATION_EPOCH_BATCH_END = 33
    INFERENCE_START = 40
    INFERENCE_END = 41
    INFERENCE_REPEAT_START = 50
    INFERENCE_REPEAT_END = 51


class Experiment:
    """Base class for an Experiment.

    This class is used as the base class for more specific experiments. The
    experiments use ``hooks`` for reporting the different status changes to the
    ``Metrics`` during the execution of the experiment.
    """

    def __init__(self) -> None:
        self.hooks: Dict = {
            Signals.EXPERIMENT_START: [],
            Signals.EXPERIMENT_END: [],
            Signals.EPOCH_START: [],
            Signals.EPOCH_END: [],
            Signals.TRAIN_EPOCH_START: [],
            Signals.TRAIN_EPOCH_END: [],
            Signals.TRAIN_EPOCH_BATCH_START: [],
            Signals.TRAIN_EPOCH_BATCH_END: [],
            Signals.VALIDATION_EPOCH_START: [],
            Signals.VALIDATION_EPOCH_END: [],
            Signals.VALIDATION_EPOCH_BATCH_START: [],
            Signals.VALIDATION_EPOCH_BATCH_END: [],
            Signals.INFERENCE_START: [],
            Signals.INFERENCE_END: [],
            Signals.INFERENCE_REPEAT_START: [],
            Signals.INFERENCE_REPEAT_END: [],
        }

        self.results: Optional[Any] = None

    # add the specified routine to call with the specified hook key to the experiment.
    def add_hook(self, key: Signals, hook: Callable) -> None:
        """Register a hook for the experiment.

        Register a new hook for a particular signal. During the execution of
        the experiment, the ``hook`` function will be called.

        Args:
            key: signal which the hook will be registered to.
            hook: a function that will be called when the signal is emitted.
        """
        self.hooks[key].append(hook)

    def clear_hooks(self) -> None:
        """Remove all the hooks from the experiment."""
        for key in self.hooks:
            self.hooks[key] = []

    # call the routine that is associated with the specified hook key.
    def _call_hook(self, key: Signals, *args: Any, **kwargs: Any) -> Dict:
        """Invoke the hooks for a specific key."""
        ret = {}
        for hook in self.hooks[key]:
            hook_ret = hook(*args, **kwargs)
            if isinstance(hook_ret, dict):
                ret.update(hook_ret)

        return ret

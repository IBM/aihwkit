# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Base class for an Experiment Runner."""

# pylint: disable=too-few-public-methods

from typing import Any

from aihwkit.experiments.experiments.base import Experiment


class Runner:
    """Base class for an Experiment Runner."""

    def run(self, experiment: Experiment, **kwargs: Any) -> None:
        """Run a single Experiment."""
        raise NotImplementedError

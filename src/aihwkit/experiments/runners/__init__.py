# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Experiment Runners for aihwkit."""

# Convenience imports for easier access to the classes.

from aihwkit.experiments.runners.cloud import CloudRunner
from aihwkit.experiments.runners.i_cloud import InferenceCloudRunner
from aihwkit.experiments.runners.local import LocalRunner
from aihwkit.experiments.runners.i_local import InferenceLocalRunner
from aihwkit.experiments.runners.metrics import LocalMetric
from aihwkit.experiments.runners.i_metrics import InferenceLocalMetric

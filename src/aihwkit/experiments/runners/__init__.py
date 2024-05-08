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

"""Experiment Runners for aihwkit."""

# Convenience imports for easier access to the classes.

from aihwkit.experiments.runners.cloud import CloudRunner
from aihwkit.experiments.runners.i_cloud import InferenceCloudRunner
from aihwkit.experiments.runners.local import LocalRunner
from aihwkit.experiments.runners.i_local import InferenceLocalRunner
from aihwkit.experiments.runners.metrics import LocalMetric
from aihwkit.experiments.runners.i_metrics import InferenceLocalMetric

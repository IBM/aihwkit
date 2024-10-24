# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Legacy class for import the noise models for inference"""

import warnings
from aihwkit.inference import *  # pylint: disable=unused-wildcard-import, wildcard-import

warnings.warn(
    FutureWarning(
        "\n\nThe `aihwkit.simulator.noise_models` module has been superseded "
        "by the `aihwkit.inference` module. "
        "Please replace `from aihwkit.simulator.noise_models import ...` "
        "with `from aihwkit.inference import ...` in your import statement.\n"
    )
)

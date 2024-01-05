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

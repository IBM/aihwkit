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

"""Package version string."""

import os

VERSION_FILE = os.path.join(os.path.dirname(__file__), "VERSION.txt")
with open(VERSION_FILE, encoding="utf-8") as version_file:
    __version__ = version_file.read().strip()

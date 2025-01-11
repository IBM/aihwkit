# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Package version string."""

import os

VERSION_FILE = os.path.join(os.path.dirname(__file__), "VERSION.txt")
with open(VERSION_FILE, encoding="utf-8") as version_file:
    __version__ = version_file.read().strip()

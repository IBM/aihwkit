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

# pylint: disable=import-error, no-name-in-module, invalid-name

"""AIHWKIT extension """

from importlib.util import find_spec

EXTENSION_COMPILED = find_spec(".aihwkit_extension", package="aihwkit.extension") is not None

if EXTENSION_COMPILED:
    from .functions import FloatPrecisionCast
    from aihwkit.extension.aihwkit_extension import ops as extension_ops
else:
    extension_ops = None

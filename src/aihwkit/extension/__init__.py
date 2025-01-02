# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=import-error, no-name-in-module, invalid-name

"""AIHWKIT extension """

from importlib.util import find_spec

EXTENSION_COMPILED = find_spec(".aihwkit_extension", package="aihwkit.extension") is not None

if EXTENSION_COMPILED:
    from .functions import FloatPrecisionCast
    from aihwkit.extension.aihwkit_extension import ops as extension_ops
else:
    extension_ops = None

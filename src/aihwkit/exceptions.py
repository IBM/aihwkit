# -*- coding: utf-8 -*-

# (C) Copyright 2020 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Custom Exceptions for aihwkit."""


class AihwkitException(Exception):
    """Base class for exceptions related to aihwkit."""


class ModuleError(AihwkitException):
    """Exceptions related to analog neural network modules."""


class TileError(AihwkitException):
    """Exceptions related to analog tiles."""


class CudaError(AihwkitException):
    """Exceptions related to CUDA."""


class ConfigError(AihwkitException):
    """Exception related to tile configuration. """

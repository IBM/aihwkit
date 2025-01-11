# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Custom Exceptions for aihwkit."""


class AihwkitException(Exception):
    """Base class for exceptions related to aihwkit."""


class ModuleError(AihwkitException):
    """Exceptions related to analog neural network modules."""


class TileError(AihwkitException):
    """Exceptions related to analog tiles."""


class TileModuleError(TileError):
    """Exceptions related to analog tile modules."""


class ArgumentError(AihwkitException):
    """Exceptions related to wrong arguments."""


class CudaError(AihwkitException):
    """Exceptions related to CUDA."""


class ConfigError(AihwkitException):
    """Exceptions related to tile configuration."""


class AnalogBiasConfigError(ConfigError):
    """Exception that analog bias is wrongly set."""


class TorchTileConfigError(ConfigError):
    """Exceptions related to torch tile configuration."""


class CloudError(AihwkitException):
    """Exceptions related to the cloud functionality."""


class FusionExportError(CloudError):
    """Exceptions related to the fusion export functionality."""

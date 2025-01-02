# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Helper class for adding rpu_config to neural network model"""
# pylint: disable=no-name-in-module,import-error

from aihwkit.cloud.converter.definitions.i_input_file_pb2 import (  # type: ignore[attr-defined]
    AnalogProto,
)


# pylint: disable=too-few-public-methods)
class AnalogInfo:
    """Data class for fields from protobuf AnalogProto message"""

    def __init__(self, a_info: AnalogProto):  # type: ignore[valid-type]
        """Constructor for this class"""

        # all three fields are required
        self.output_noise_strength = a_info.output_noise_strength  # type: ignore[attr-defined]
        self.adc = a_info.adc  # type: ignore[attr-defined]
        self.dac = a_info.dac  # type: ignore[attr-defined]

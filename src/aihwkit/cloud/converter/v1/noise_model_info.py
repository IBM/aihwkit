# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Noise model info in rpu_config to neural network model"""

# pylint: disable=no-name-in-module,import-error
from aihwkit.cloud.converter.definitions.i_input_file_pb2 import (  # type: ignore[attr-defined]
    NoiseModelProto,
)


# pylint: disable=too-many-instance-attributes
class NoiseModelInfo:
    """Data only class for fields from protobuf NoiseModelProto message"""

    PCM = "pcm"
    GENERIC = "generic"

    def __init__(self, nm_proto: NoiseModelProto):  # type: ignore[valid-type]
        """Constructor for this class"""

        type_ = nm_proto.WhichOneof("item")  # type: ignore[attr-defined]

        info = None
        if type_ == "pcm":
            # pcm does NOT have 2 extra fields
            info = nm_proto.pcm  # type: ignore[attr-defined]
        else:
            # generic HAS 2 extra fields
            info = nm_proto.generic  # type: ignore[attr-defined]

        self.device_id = info.device_id
        self.programming_noise_scale = info.programming_noise_scale
        self.read_noise_scale = info.read_noise_scale
        self.drift_scale = info.drift_scale
        self.drift_compensation = info.drift_compensation
        self.poly_first_order_coef = info.poly_first_order_coef
        self.poly_second_order_coef = info.poly_second_order_coef
        self.poly_constant_coef = info.poly_constant_coef

        self._drift_mean = -1.1
        self._drift_std = -1.1

        # Generic device has two extra field.
        if info.device_id == self.GENERIC:
            self._drift_mean = info.drift_mean
            self._drift_std = info.drift_std

    def _assert_generic(self) -> None:
        """Check is device is generic"""

        assert self.device_id == self.GENERIC, "device_id does not have value '{}'".format(
            self.GENERIC
        )

    @property
    def drift_mean(self) -> float:
        """Enforce access to drift_mean if this is a generic device"""

        self._assert_generic()
        return self._drift_mean

    @property
    def drift_std(self) -> float:
        """Enforce access to drift_std if this is a generic device"""

        self._assert_generic()
        return self._drift_std


# pylint: enable=too-many-instance-attributes

# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Creates InferenceRPUConfig to add to nn model"""
from typing import Dict, Any
from collections import OrderedDict

from aihwkit.simulator.configs.configs import InferenceRPUConfig
from aihwkit.simulator.presets.web import (
    WebComposerInferenceRPUConfig,
    OldWebComposerInferenceRPUConfig,
)
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.inference.noise.custom import StateIndependentNoiseModel
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.cloud.converter.v1.analog_info import AnalogInfo
from aihwkit.cloud.converter.v1.noise_model_info import NoiseModelInfo
from aihwkit.exceptions import ConfigError

RPU_CLASSES = {
    "InferenceRPUConfig": InferenceRPUConfig,
    "WebComposerInferenceRPUConfig": WebComposerInferenceRPUConfig,
    "OldWebComposerInferenceRPUConfig": OldWebComposerInferenceRPUConfig,
}


# pylint: disable=too-few-public-methods
class NoiseModelDeviceIDException(Exception):
    """Exception raised if noise model device id is not correct"""


class RPUconfigInfo:
    """Data only class for RPUConfig fields"""

    def __init__(self, nm_info: NoiseModelInfo, a_info: AnalogInfo, layers: Any = None):
        """
        The only constructor for this class
        """
        self._noise_model_info = nm_info
        self._analog_info = a_info
        self._layers = layers
        self._device_id = ""

    @staticmethod
    def _get_common_rpucfg_name(layers: Any) -> Any:
        """Set common rpu config name by search all analog layers"""
        # Use default RPU config for Composer
        if layers is None:
            return "WebComposerInferenceRPUConfig"
        # Need to loop through protobuf layers and figure out
        #   common rpu_config value.
        names: Dict[str, int] = {}
        # pylint: disable=too-many-nested-blocks
        for layer_proto in layers:  # type: ignore[attr-defined]
            if layer_proto.WhichOneof("item") == "layer":
                layer = layer_proto.layer
                if layer.id.startswith("Analog"):
                    # Loop though all AttributeProto objecs in layer.arguments
                    for argument in layer.arguments:
                        if argument.name == "rpu_config":
                            # stored as UTF8 byte string in attribute s
                            arg_value = getattr(argument, "s")
                            # update count of this rpu_config in all analog layers
                            if arg_value in names:
                                names[arg_value] += 1
                            else:
                                names[arg_value] = 1
        # pylint: enable=too-many-nested-blocks
        # should have exactly on in dictionary 'names'
        if len(names) > 1:
            print(f">>> ERROR: more than one rpu_config: {names}")
            return None
        if len(names) == 1:
            # keys() returns dict_keys object, need a list
            return list(names.keys())[0].decode("UTF-8")  # type: ignore[attr-defined]
        print(">>> INFO: experiment has not analog layers")
        return ""

    def _print_rpu_config(self, rpu_config: InferenceRPUConfig, func_id: str, dev_id: str) -> None:
        """Pretty-print rpu_config"""

        # Create ordered dictionary with field name as key and value as field value
        #   Include ADC and DAC in input/output resolution
        print_order = OrderedDict(
            {
                ("forward.out_noise", rpu_config.forward.out_noise),
                (
                    "forward.out_res/adc",
                    "{}/{}".format(rpu_config.forward.out_res, self._analog_info.adc),
                ),
                (
                    "forward.inp_res/dac",
                    "{}/{}".format(rpu_config.forward.inp_res, self._analog_info.dac),
                ),
                ("noise_model", rpu_config.noise_model),
                (
                    "noise_model.read_noise_scale",
                    rpu_config.noise_model.read_noise_scale,  # type: ignore[attr-defined]
                ),
                (
                    "noise_model.prog_noise_scale",
                    rpu_config.noise_model.prog_noise_scale,  # type: ignore[attr-defined]
                ),
                ("drift_compensation", rpu_config.drift_compensation),  # type: ignore[attr-defined]
                (
                    "noise_model.drift_scale",
                    rpu_config.noise_model.drift_scale,  # type: ignore[attr-defined]
                ),
            }
        )

        # add extra fields for GENERIC device
        if dev_id == NoiseModelInfo.GENERIC:
            print_order[
                "noise_model.drift_nu_mean"
            ] = rpu_config.noise_model.drift_nu_mean  # type: ignore[attr-defined]
            print_order[
                "noise_model.drift_nu_std"
            ] = rpu_config.noise_model.drift_nu_std  # type: ignore[attr-defined]

        output = "  rpu_config: function_id={} device_id={}\n".format(func_id, dev_id)
        for key in print_order:
            output += "    {:30}: {}\n".format(key, print_order[key])
            if key == "noise_model" and dev_id == NoiseModelInfo.PCM:
                # need to print out prog_coeff in PCMLikeNoiseModel object
                for i in range(3):
                    field_name = "noise_model.prog_coeff[{}]".format(i)
                    output += "    {:30}: {}\n".format(
                        field_name,
                        rpu_config.noise_model.prog_coeff[i],  # type: ignore[attr-defined]
                    )

        print("Here is the variable content of the rpu_config:\n{}".format(output))

    def create_inference_rpu_config(
        self, func_id: str, verbose: bool = False
    ) -> InferenceRPUConfig:
        """Creates a InferenceRPUConfig class using noise and analog info"""
        # Need to find name of 'common-rpu-conf-class-name' in protobuf
        #   This should be the consistent across all layers.
        #   The Composer Validator should have already caught this but
        #   it is checked here for testcases and other unknown environments
        rpu_class_name = self._get_common_rpucfg_name(self._layers)
        print(f">>> INFO: rpu_class_name={rpu_class_name}")
        if rpu_class_name is None or len(rpu_class_name) == 0:
            raise ConfigError("class name error. see previous messages")
        rpu_config_class = None
        if rpu_class_name in RPU_CLASSES:
            rpu_config_class = RPU_CLASSES[rpu_class_name]
        else:
            raise ConfigError(
                f"rpu class name '{rpu_class_name}' not one of '{RPU_CLASSES.keys()}'"
            )

        # Dynamically create the right InferenceRPUConfig class
        rpu_config = rpu_config_class()
        # Assign values from AnalogProto
        rpu_config.forward.out_noise = self._analog_info.output_noise_strength

        # changed input/output res to use a formula. print out adc and dac
        rpu_config.forward.out_res = 1.0 / (2**self._analog_info.adc - 2)
        rpu_config.forward.inp_res = 1.0 / (2**self._analog_info.dac - 2)

        # Assign values from NoiseModelProto (CM Noise model)
        self._device_id = self._noise_model_info.device_id
        if self._device_id == NoiseModelInfo.PCM:
            rpu_config.noise_model = PCMLikeNoiseModel(
                g_max=25.0,
                prog_coeff=[
                    self._noise_model_info.poly_constant_coef,
                    self._noise_model_info.poly_first_order_coef,
                    self._noise_model_info.poly_second_order_coef,
                ],
            )
        elif self._device_id == NoiseModelInfo.GENERIC:
            rpu_config.noise_model = StateIndependentNoiseModel(
                g_max=25.0,
                prog_coeff=[
                    self._noise_model_info.poly_constant_coef,
                    self._noise_model_info.poly_first_order_coef,
                    self._noise_model_info.poly_second_order_coef,
                ],
            )

            # These are unique to generic
            rpu_config.noise_model.drift_nu_mean = (
                self._noise_model_info.drift_mean  # type: ignore[attr-defined]
            )
            rpu_config.noise_model.drift_nu_std = (
                self._noise_model_info.drift_std  # type: ignore[attr-defined]
            )
        else:
            raise NoiseModelDeviceIDException(
                "invalid noise model device id {}".format(self._device_id)
            )

        # common fields
        rpu_config.noise_model.prog_noise_scale = (  # type: ignore[attr-defined]
            self._noise_model_info.programming_noise_scale
        )
        rpu_config.noise_model.read_noise_scale = (  # type: ignore[attr-defined]
            self._noise_model_info.read_noise_scale
        )

        # Drift compensation in protobuf is boolean (bool)
        rpu_config.drift_compensation = None  # type: ignore[assignment]
        if self._noise_model_info.drift_compensation:
            rpu_config.drift_compensation = GlobalDriftCompensation()

        rpu_config.noise_model.drift_scale = (  # type: ignore[attr-defined]
            self._noise_model_info.drift_scale
        )

        if verbose:
            self._print_rpu_config(rpu_config, func_id, self._noise_model_info.device_id)
        return rpu_config

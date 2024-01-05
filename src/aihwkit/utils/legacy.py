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

"""Conversion script of legacy checkpoints (pre v0.8) to the new format."""

# pylint: disable=too-many-locals, too-many-statements, too-many-branches

from typing import Tuple, Optional
from collections import OrderedDict
from copy import deepcopy

from torch import Tensor, float32, ones
from torch.nn import Module

from aihwkit.simulator.configs.configs import InferenceRPUConfig
from aihwkit.simulator.parameters.base import RPUConfigBase
from aihwkit.simulator.parameters import PrePostProcessingParameter, WeightRemapParameter
from aihwkit.simulator.presets.web import OldWebComposerMappingParameter
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel


def convert_legacy_checkpoint(
    legacy_chkpt: "OrderedDict", model: Optional[Module] = None
) -> Tuple["OrderedDict", RPUConfigBase]:
    """Attempts to convert the fields of an legacy checkoint model so
    that it can be loaded with the new (v0.8) tile structure.

    Caution:
        Might not be fully functional in all cases.

    Important:

        Only one of the RPUConfig of any tile is return. If tiles have
        different RPUConfigs, any of them might be return in not
        particular order.

    Args:
        legacy_chkpt: loaded checkpoint (state_dict) from pre v0.8 version.
        model: Will solve more issues if instantiated model is given (that will load the stat_dict)

    Returns:
        Tuple of converted checkoint and the (one of the any) RPUConfig found in the tiles.
    """

    def check_conv_mapped(prefix1: str) -> bool:
        mod_name = prefix1[:-1]
        if mod_name in layer_dic:
            if "Conv" in layer_dic[mod_name] and "Mapped" in layer_dic[mod_name]:
                return True
        return False

    def get_key_from_ending(key_name: str, par_name: str, prefix: str) -> str:
        ending = key_name.split(par_name)[-1]
        arr = [int(val) for val in ending.split("_") if len(val) > 0]
        if len(arr) == 1 and arr[0] == 0:
            new_key = "analog_module." + par_name
        elif len(arr) == 1 and arr[0] != 0:
            new_key = "analog_module." + par_name + "." + str(arr)
        elif len(arr) > 1:
            new_key = "analog_module.array"
            if check_conv_mapped(prefix):
                new_key = "array"
            for val in arr:
                new_key += "." + str(val)
            new_key += "." + par_name
        else:
            # don't know this should not happen. Just use same ending
            new_key = "analog_module." + par_name + ending
        return new_key

    has_mapped = False
    layer_dic = {}
    if model is not None:
        for name, analog_layer in model.named_analog_layers():
            has_mapped = has_mapped or "Mapped" in analog_layer.__class__.__name__
            layer_dic[name] = analog_layer.__class__.__name__

    if not has_mapped:
        for tile in model.analog_tiles():
            tile.rpu_config.mapping.max_input_size = 0
            tile.rpu_config.mapping.max_output_size = 0

    legacy_chkpt = deepcopy(legacy_chkpt)
    for key, value in legacy_chkpt.items():
        if "analog_model.analog_tile_state" in key:
            # this is actually a new checkpoint. abort.
            rpu_config = deepcopy(value["rpu_config"])
            return legacy_chkpt, rpu_config

    for key, value in legacy_chkpt.items():
        if "analog_tile_state" in key:
            rpu_config = value["rpu_config"]
            if not isinstance(rpu_config, InferenceRPUConfig):
                continue
            if not hasattr(rpu_config, "mapping"):
                rpu_config.mapping = OldWebComposerMappingParameter()
            if "weight_scaling_omega_columnwise" in rpu_config.mapping.__dict__:
                rpu_config.mapping.weight_scaling_columnwise = rpu_config.mapping.__dict__.pop(
                    "weight_scaling_omega_columnwise"
                )
            if "learn_out_scaling_alpha" in rpu_config.mapping.__dict__:
                rpu_config.mapping.learn_out_scaling = rpu_config.mapping.__dict__.pop(
                    "learn_out_scaling_alpha"
                )
                rpu_config.mapping.out_scaling_columnwise = (
                    rpu_config.mapping.weight_scaling_columnwise
                )

            if not has_mapped:
                # need to set tile sizes to full since otherwise
                # mapping would now still occur
                rpu_config.mapping.max_input_size = 0
                rpu_config.mapping.max_output_size = 0

            if isinstance(rpu_config.noise_model, PCMLikeNoiseModel) and not hasattr(
                rpu_config.noise_model, "prog_coeff_g_max_reference"
            ):
                rpu_config.noise_model.prog_coeff_g_max_reference = rpu_config.noise_model.g_max

            if not hasattr(rpu_config, "pre_post"):
                rpu_config.pre_post = PrePostProcessingParameter()
            if not hasattr(rpu_config, "remap"):
                rpu_config.remap = WeightRemapParameter()
            if not hasattr(rpu_config.modifier, "coeffs"):
                dic = rpu_config.modifier.__dict__
                rpu_config.modifier.coeffs = [
                    dic.pop("coeff0"),
                    dic.pop("coeff1"),
                    dic.pop("coeff2"),
                ]

    chkpt = OrderedDict()
    for key, value in legacy_chkpt.items():
        name = key.split(".")[-1]
        prefix = key.split(name)[0]
        if "bias" == name:
            if len([k for k in legacy_chkpt if k.startswith(prefix + "analog_tile_state")]) > 0:
                # digital bias of an analog module. Now handled inside the TileModule
                if check_conv_mapped(prefix):
                    new_key = prefix + name
                else:
                    new_key = prefix + "analog_module." + name
                chkpt[new_key] = value
                continue

        if name.startswith("analog_tile_state"):
            # tile (array) numbers
            new_key = prefix + get_key_from_ending(name, "analog_tile_state", prefix)
            state = value
            for legacy_key in [
                "noise_model",
                "drift_compensation",
                "drift_baseline",
                "drift_readout_tensor",
                "reference_combined_weights",
                "programmed_weights",
                "nu_drift_list",
                "shared_weights",
                "image_sizes",
            ]:
                state.pop(legacy_key, None)

            # rename bias flag
            state["use_bias"] = state.pop("bias", False)

            # drift comp
            alpha = state.get("alpha", None)
            if isinstance(alpha, Tensor):
                p_key = new_key.replace("analog_tile_state", "alpha")
                chkpt[p_key] = alpha

            # will be applied
            # out_scaling_alpha = state.get('out_scaling_alpha', None)
            out_size = state["out_size"]
            rpu_config = state["rpu_config"]
            device = state["analog_ctx"].device

            # mapping scales
            p_key = new_key.replace("analog_tile_state", "mapping_scales")
            chkpt[p_key] = ones((out_size,), dtype=float32, device=device)
            if "mapping_scales" in state:
                chkpt[p_key] *= state.pop("mapping_scales", 1.0)

            #  in case of the very old alpha scale
            chkpt[p_key] *= state.pop("analog_alpha_scale", 1.0)

            chkpt[new_key] = state
            continue

        if name.startswith("analog_out_scaling_alpha"):
            # this is used for the mapped
            new_name = name.split("analog_")[-1] + "_0"
            new_key = prefix + get_key_from_ending(new_name, "out_scaling_alpha", prefix)
            chkpt[new_key] = value
            continue

        if name.startswith("analog_shared_weights"):
            # new_key = prefix + 'analog_module.' + name
            # chkpt[new_key] = value
            continue

        if name.startswith("analog_ctx"):
            new_key = prefix + get_key_from_ending(name, "analog_ctx", prefix)
            chkpt[new_key] = value
            continue

        chkpt[key] = value

    return chkpt, rpu_config

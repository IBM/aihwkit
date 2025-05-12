# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

""" Functions to convert a given model to a quantized counterpart """

from copy import deepcopy

from torch.nn import Module, Linear, Conv2d

from aihwkit.nn.conversion import convert_to_digital
from aihwkit.nn.low_precision_modules.conversion_utils import (
    DEFAULT_CONVERSIONS,
    LEAF_MODULES,
    get_module_args,
)
from aihwkit.nn.low_precision_modules.quantization_states import enable_quant_states
from aihwkit.nn.low_precision_modules.quantized_input_module import QuantizedInputModule
from aihwkit.nn.modules.base import AnalogLayerBase
from aihwkit.simulator.digital_low_precision.config_utils import convert_configs_to_kwargs_dict
from aihwkit.simulator.parameters.quantization import QuantizationMap


def convert_to_quantized(model: Module, quantization_map: QuantizationMap) -> Module:
    """High level function to perform the quantization of a model according to the QuantizationMap
    defined by the user. See the `QuantizationMap` dataclass for instructions on how to define
    its fields.

    This function calls the recursive `quantize_model` function which performs the actual
    conversion between the individual modules. It then enables the quantization states for
    all the quantized modules.

    Parameters
    ----------
    model : Module
        The model to quantize
    quantization_map : QuantizationMap
        The dataclass that contains instructions on how to quantize the model

    Returns
    -------
    torch.nn.Module
        The quantized model
    """
    quant_model = deepcopy(model)
    quant_model = quantize_model(quant_model, quantization_map)
    enable_quant_states(quant_model)
    return quant_model


def quantize_model(
    model: Module, quantization_map: QuantizationMap, model_name: str = ""
) -> Module:
    """Traverses a model recursively and replaces a module with a quantized counterpart,
    if such a conversion is defined in the quantization map. It realizes all the capabilities
    of the QuantizationMap, namely:
    - Excluding specific modules from the quantization, identified by exact state_dict
        string (`quantization_map.excluded_modules`)
    - Instance specific quantization, identified by exact state_dict string
        (`quantization_map.instance_qconfig_map`)
    - Module specific quantization, identified by the type of a module
        (`quantization_map.module_qconfig_map`)
    - Input quantization on specified modules, identified by the state_dict string
        (`quantization_map.input_activation_qconfig_map`)
    (see `QuantizationMap` and the related examples for more details on how to define these fields)

    Parameters
    ----------
    model : Module
        Model to recursively quantize
    quantization_map : QuantizationMap
        The dataclass that defines the quantization conversions as well as the
        quantization parameters for each conversion
    model_name : str, optional
        The name of the current module in the state dict of the original model, by default ""

    Returns
    -------
    Module
        Quantized model
    """
    # pylint: disable=too-many-branches

    # Exclude modules based on state_dict string
    if model_name in quantization_map.excluded_modules:
        return None

    # Early exit for quantizers and range estimators
    if isinstance(model, LEAF_MODULES):
        return None

    # Quantization code for instance- or module-specific conversions
    if (model_name in quantization_map.instance_qconfig_map) or (
        type(model) in quantization_map.module_qconfig_map
    ):
        # Select the quantized module counterpart and the quantization parameters
        # prioritizing the instance-specific case
        qmodule_config = (
            quantization_map.instance_qconfig_map[model_name]
            if model_name in quantization_map.instance_qconfig_map
            else quantization_map.module_qconfig_map[type(model)]
        )
        q_config, q_model = qmodule_config.module_qconfig, qmodule_config.quantized_module

        # Keep the original module if it's
        # not going to be used as a quantized module
        if not (q_config.activation_quant.n_bits > 0 or q_config.weight_quant.n_bits > 0):
            return model

        if type(model) in DEFAULT_CONVERSIONS:
            # If the type of the model falls under the DEFAULT_CONVERSIONS primitives,
            # then the quantized model code should use the QuantizationHijacker and as such
            # standard infrastructure is provided
            kwargs = get_module_args(model, None)
            quant_model = q_model(**kwargs, **convert_configs_to_kwargs_dict(q_config))

            quant_model.weight.data = model.weight.data
            if getattr(model, "bias", None) is not None:
                quant_model.bias.data = model.bias.data

        elif isinstance(model, AnalogLayerBase):
            # If the module is an analog layer, then use the convert_to_digital first, to
            # merge the scales with the parameters and then quantize to the counterpart.
            # NOTE: Only Linear and Conv2d are supported at the moment.
            model = convert_to_digital(model)
            assert type(model) in [Linear, Conv2d], "Only Linear and Conv2d supported at the moment"
            kwargs = get_module_args(model, None)
            quant_model = q_model(**kwargs, **convert_configs_to_kwargs_dict(q_config))

            quant_model.weight.data = model.weight.data
            if getattr(model, "bias", None) is not None:
                quant_model.bias.data = model.bias.data
        else:
            # Every other custom conversion, should conform to the following constructor structure,
            # passing the original model in the contructor along with various kwargs and the
            # quantization map
            kwargs = {
                "quantization_map": quantization_map,
                **convert_configs_to_kwargs_dict(q_config),
            }
            quant_model = q_model(model, **kwargs)
    else:
        # If the conversion is not defined, recursively call the function on the children
        quant_model = model
        for name, mod in quant_model.named_children():
            full_name = model_name + "." + name if model_name else name
            if full_name in quantization_map.excluded_modules:
                continue
            new_mod = quantize_model(mod, quantization_map, model_name=full_name)

            # If the returned module is not None, wrap it with an input quantizer
            # if such a selection is defined and swap it with the original module
            if new_mod is not None:
                if full_name in quantization_map.input_activation_qconfig_map.keys():
                    new_mod = QuantizedInputModule(
                        new_mod, quantization_map.input_activation_qconfig_map[full_name]
                    )
                setattr(quant_model, name, new_mod)

    return quant_model

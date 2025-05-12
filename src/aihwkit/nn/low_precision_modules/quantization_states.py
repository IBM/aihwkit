# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

# mypy: disable-error-code=attr-defined

"""Functions to manipulate the quantization state of a module"""

from torch.nn import Module

from aihwkit.simulator.digital_low_precision.base_quantized_classes import (
    QuantizedModule,
    _set_layer_estimate_ranges,
    _set_layer_estimate_ranges_train,
    _set_layer_fix_ranges,
    _set_layer_learn_ranges,
)


def quantized_weights(model: Module) -> None:
    """Enables quantization of the weights.

    Parameters
    ----------
    model : Module
        Model to enable weight quantization
        recursively on its modules.
    """

    def _fn(layer: Module) -> None:
        if isinstance(layer, QuantizedModule):
            layer.quantized_weights()

    model.apply(_fn)


def full_precision_weights(model: Module) -> None:
    """Places the weights in full precision.

    Parameters
    ----------
    model : Module
        Model to place the weight of its modules
        in full precision
    """

    def _fn(layer: Module) -> None:
        if isinstance(layer, QuantizedModule):
            layer.full_precision_weights()

    model.apply(_fn)


def quantized_acts(model: Module) -> None:
    """Enables quantization of the activations.

    Parameters
    ----------
    model : Module
        Model to enable activation quantization
        recursively on its modules.
    """

    def _fn(layer: Module) -> None:
        if isinstance(layer, QuantizedModule):
            layer.quantized_acts()

    model.apply(_fn)


def full_precision_acts(model: Module) -> None:
    """Places the activations in full precision.

    Parameters
    ----------
    model : Module
        Model to place the activations of its modules
        in full precision
    """

    def _fn(layer: Module) -> None:
        if isinstance(layer, QuantizedModule):
            layer.full_precision_acts()

    model.apply(_fn)


def quantized(model: Module) -> None:
    """Enables quantization on both weights and activations.

    Parameters
    ----------
    model : Module
        Model to enable activation and weight quantization
        recursively on its modules.
    """

    def _fn(layer: Module) -> None:
        if isinstance(layer, QuantizedModule):
            layer.quantized()

    model.apply(_fn)


def full_precision(model: Module) -> None:
    """Places the activations and weights in full precision.

    Parameters
    ----------
    model : Module
        Model to place the activations and weights of its modules
        in full precision
    """

    def _fn(layer: Module) -> None:
        if isinstance(layer, QuantizedModule):
            layer.full_precision()

    model.apply(_fn)


# Methods for switching quantizer quantization states
def learn_ranges(model: Module) -> None:
    """Places the quantizers of a model in `learn_ranges` mode

    Parameters
    ----------
    model : Module
        Model to place the quantizers of its modules
        in `learn_ranges` mode
    """
    model.apply(_set_layer_learn_ranges)


def fix_ranges(model: Module) -> None:
    """Places the quantizers of a model in `fix_ranges` mode

    Parameters
    ----------
    model : Module
        Model to place the quantizers of its modules
        in `fix_ranges` mode
    """
    model.apply(_set_layer_fix_ranges)


def fix_act_ranges(model: Module) -> None:
    """Places the activation quantizers of a model in `fix_ranges` mode

    Parameters
    ----------
    model : Module
        Model to place the activation quantizers of its modules
        in `fix_ranges` mode
    """

    def _fn(module: Module) -> None:
        if isinstance(module, QuantizedModule) and hasattr(module, "activation_quantizer"):
            _set_layer_fix_ranges(module.activation_quantizer)

    model.apply(_fn)


def fix_weight_ranges(model: Module) -> None:
    """Places the weight quantizers of a model in `fix_ranges` mode

    Parameters
    ----------
    model : Module
        Model to place the weight quantizers of its modules
        in `fix_ranges` mode
    """

    def _fn(module: Module) -> None:
        if isinstance(module, QuantizedModule) and hasattr(module, "weight_quantizer"):
            _set_layer_fix_ranges(module.weight_quantizer)

    model.apply(_fn)


def estimate_ranges(model: Module) -> None:
    """Places the quantizers of a model in `estimate_ranges` mode

    Parameters
    ----------
    model : Module
        Model to place the quantizers of its modules
        in `estimate_ranges` mode
    """
    model.apply(_set_layer_estimate_ranges)


def estimate_act_ranges(model: Module) -> None:
    """Places the activation quantizers of a model in `estimate_ranges` mode

    Parameters
    ----------
    model : Module
        Model to place the activation quantizers of its modules
        in `estimate_ranges` mode
    """

    def _fn(module: Module) -> None:
        if isinstance(module, QuantizedModule) and hasattr(module, "activation_quantizer"):
            _set_layer_estimate_ranges(module.activation_quantizer)

    model.apply(_fn)


def estimate_ranges_train(model: Module) -> None:
    """Places the quantizers of a model in `estimate_ranges_train` mode

    Parameters
    ----------
    model : Module
        Model to place the quantizers of its modules
        in `estimate_ranges_train` mode
    """
    model.apply(_set_layer_estimate_ranges_train)


def reset_act_ranges(model: Module) -> None:
    """Resets the activation ranges of a model to uninitialized

    Parameters
    ----------
    model : Module
        Model to reset the activation quantizers of its modules
    """

    def _fn(module: Module) -> None:
        if isinstance(module, QuantizedModule) and hasattr(module, "activation_quantizer"):
            module.activation_quantizer.reset_ranges()

    model.apply(_fn)


def set_quant_state(model: Module, weight_quant: bool, act_quant: bool) -> None:
    """Function to configure the activation and weight quantizers of a model. The model can
    be configured to either have the weights/activations in full precision or the quantization
    enabled.

    Parameters
    ----------
    model : Module
        Model to configure
    weight_quant : bool
        If True, enable weight quantization for all modules in the model.
        If False, keep the weights in full precision.
    act_quant : bool
        If True, enable activation quantization for all modules in the model.
        If False, keep the activations in full precision.
    """
    if act_quant:
        quantized_acts(model)
    else:
        full_precision_acts(model)

    if weight_quant:
        quantized_weights(model)
    else:
        full_precision_weights(model)


def enable_quant_states(model: Module) -> None:
    """Function to enable the quantization states in all modules that inherit from the
    `QuantizedModule` class.

    Parameters
    ----------
    model : Module
        Model to configure
    """

    def _fn(layer: Module) -> None:
        if isinstance(layer, QuantizedModule):
            if layer.n_bits > 0:
                layer.quantized_weights()
            else:
                layer.full_precision_weights()
            if layer.n_bits_act > 0:
                layer.quantized_acts()
            else:
                layer.full_precision_acts()

    model.apply(_fn)

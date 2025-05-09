# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

# pylint: skip-file
# type: ignore

from torch import nn

from aihwkit.simulator.digital_low_precision.base_quantized_classes import (
    QuantizedModule,
    _set_layer_learn_ranges,
    _set_layer_fix_ranges,
    _set_layer_estimate_ranges,
    _set_layer_estimate_ranges_train,
)


class QuantizedModel(nn.Module):
    """
    Parent class for a quantized model. This allows you to have convenience functions to put the
    whole model into quantization or full precision or to freeze BN. Otherwise it does not add any
    further functionality, so it is not a necessity that a quantized model uses this class.
    """

    def quantized_weights(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.quantized_weights()

        self.apply(_fn)

    def full_precision_weights(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.full_precision_weights()

        self.apply(_fn)

    def quantized_acts(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.quantized_acts()

        self.apply(_fn)

    def full_precision_acts(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.full_precision_acts()

        self.apply(_fn)

    def quantized(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.quantized()

        self.apply(_fn)

    def full_precision(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.full_precision()

        self.apply(_fn)

    # Methods for switching quantizer quantization states
    def learn_ranges(self):
        self.apply(_set_layer_learn_ranges)

    def fix_ranges(self):
        self.apply(_set_layer_fix_ranges)

    def fix_act_ranges(self):
        def _fn(module):
            if isinstance(module, QuantizedModule) and hasattr(module, "activation_quantizer"):
                _set_layer_fix_ranges(module.activation_quantizer)

        self.apply(_fn)

    def fix_weight_ranges(self):
        def _fn(module):
            if isinstance(module, QuantizedModule) and hasattr(module, "weight_quantizer"):
                _set_layer_fix_ranges(module.weight_quantizer)

        self.apply(_fn)

    def estimate_ranges(self):
        self.apply(_set_layer_estimate_ranges)

    def estimate_act_ranges(self):
        def _fn(module):
            if isinstance(module, QuantizedModule) and hasattr(module, "activation_quantizer"):
                _set_layer_estimate_ranges(module.activation_quantizer)

        self.apply(_fn)

    def estimate_ranges_train(self):
        self.apply(_set_layer_estimate_ranges_train)

    def reset_act_ranges(self):
        def _fn(module):
            if isinstance(module, QuantizedModule) and hasattr(module, "activation_quantizer"):
                module.activation_quantizer.reset_ranges()

        self.apply(_fn)

    def set_quant_state(self, weight_quant, act_quant):
        if act_quant:
            self.quantized_acts()
        else:
            self.full_precision_acts()

        if weight_quant:
            self.quantized_weights()
        else:
            self.full_precision_weights()

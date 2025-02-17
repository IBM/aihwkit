from torch import nn
from torch.nn import functional as F

from aihwkit.simulator.digital_low_precision.base_quantized_classes import (
    FP32Acts,
    QuantizedActivation,
)
from aihwkit.simulator.digital_low_precision.base_quantized_model import QuantizedModel
from aihwkit.simulator.digital_low_precision.hijacker import QuantizationHijacker
from aihwkit.simulator.digital_low_precision.quantization_manager import QuantizationManager


class QuantLinear(QuantizationHijacker, nn.Linear):
    """Quantized layer of torch.nn.Linear with weight/act quantization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_forward(self, x, weight, bias, offsets=None):
        return F.linear(x.contiguous(), weight.contiguous(), bias=bias)


class QuantConv2d(QuantizationHijacker, nn.Conv2d):
    """Quantized layer of torch.nn.Conv2d with weight/act quantization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_forward(self, x, weight, bias, offsets=None):
        return F.conv2d(
            x.contiguous(),
            weight.contiguous(),
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuantizedActivationWrapper(QuantizedActivation):
    """
    Wraps over a layer and quantized the activation.
    It also allow for tying the input and output quantizer which is helpful
    for layers such Average Pooling.
    """

    def __init__(
        self,
        layer,
        tie_activation_quantizers=False,
        input_quantizer: QuantizationManager = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tie_activation_quantizers = tie_activation_quantizers
        if input_quantizer:
            assert isinstance(input_quantizer, QuantizationManager)
            self.activation_quantizer = input_quantizer
        self.layer = layer

    def quantize_activations_no_range_update(self, x):
        if self._quant_a:
            return self.activation_quantizer.quantizer(x)
        else:
            return x

    def forward(self, x):
        x = self.layer(x)
        if self.tie_activation_quantizers:
            # The input activation quantizer is used to quantize the activation
            # but without updating the quantization range
            return self.quantize_activations_no_range_update(x)
        else:
            return self.quantize_activations(x)


class QuantLayerNorm(QuantizationHijacker, nn.LayerNorm):
    """Quantized layer of torch.nn.LayerNorm with input and weight quantization"""

    def __init__(self, *args, activation=None, **kwargs):
        super().__init__(*args, activation=activation, **kwargs)

    def run_forward(self, x, weight, bias, offsets=None):
        return F.layer_norm(
            input=x.contiguous(),
            normalized_shape=self.normalized_shape,
            weight=weight.contiguous(),
            bias=bias.contiguous(),
            eps=self.eps,
        )


class QuantEmbedding(QuantizationHijacker, nn.Embedding):
    """Quantization of the Embedding, weight quantization.
    Note: Embedding should not quantize activations, as it is simply a lookup table,
    which is already quantized.
    """

    def __init__(self, *args, activation=None, **kwargs):
        super().__init__(*args, activation=activation, **kwargs)
        # NB: Embedding should not quantize activations, as it is simply a lookup table,
        # which is already quantized.
        self.activation_quantizer = FP32Acts()

    def run_forward(self, x, weight, bias, offsets=None):
        return F.embedding(
            input=x.contiguous(),
            weight=weight.contiguous(),
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )


class QuantBatchNorm2d(QuantizedModel):
    """Quantization of the BatchNorm2d module. output activations are quantized."""

    def __init__(self, org_model, **quant_params):
        super().__init__()
        self.module = org_model

        self.act_bn_quantizer = QuantizedActivation(**quant_params)

    def forward(self, x):
        y = self.module(x)
        y_quant = self.act_bn_quantizer(y)
        return y_quant

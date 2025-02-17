import torch
from pytest import mark
from torch.nn import Linear, Sequential

from aihwkit.nn.low_precision_conversion import convert_to_quantized
from aihwkit.nn.low_precision_modules.conversion_utils import append_default_conversions
from aihwkit.nn.low_precision_modules.quantized_base_modules import QuantLinear
from aihwkit.simulator.parameters.quantization import (
    ActivationQuantConfig,
    QuantizationConfig,
    QuantizationMap,
    QuantizedModuleConfig,
)


@mark.parametrize("act_bits", [0, 4, 8])
@mark.parametrize("w_bits", [0, 4, 8])
def test_layer_quantization(act_bits, w_bits):
    input = torch.ones((1, 1024))
    torch_layer = Linear(in_features=1024, out_features=1024, bias=False)

    quant_map = QuantizationMap()
    quant_map.default_qconfig.activation_quant.n_bits = act_bits
    quant_map.default_qconfig.weight_quant.n_bits = w_bits
    append_default_conversions(quant_map)

    quant_layer = convert_to_quantized(torch_layer, quant_map)

    torch_output = torch_layer(input)
    quant_output = quant_layer(input)

    if act_bits == 0 and w_bits == 0:
        assert torch.allclose(torch_output, quant_output, atol=1e-5)
    else:
        assert not torch.allclose(torch_output, quant_output, atol=1e-5)
        if act_bits > 0:
            assert quant_output.unique().numel() <= 2**act_bits
        else:
            assert quant_output.unique().numel() > 2**act_bits


@mark.parametrize("low_bit_act_layer", [0, 1])
def test_instance_quantization(low_bit_act_layer):
    input = torch.ones((1, 1024))
    network = Sequential(
        Linear(in_features=1024, out_features=1024, bias=False),
        Linear(in_features=1024, out_features=1024, bias=False),
    )

    quant_map = QuantizationMap()
    quant_map.default_qconfig.activation_quant.n_bits = 8
    append_default_conversions(quant_map)

    quant_map.instance_qconfig_map[f"{low_bit_act_layer}"] = QuantizedModuleConfig(
        QuantLinear, QuantizationConfig(activation_quant=ActivationQuantConfig(n_bits=4))
    )

    quant_network = convert_to_quantized(network, quant_map)

    quant_output = quant_network(input)

    output_numel = quant_output.unique().numel()
    if low_bit_act_layer == 0:
        assert output_numel > 2**4 and output_numel <= 2**8
    else:
        assert output_numel <= 2**4


@mark.parametrize("actq_layer", [None, 0, 1])
def test_input_qmodule(actq_layer):
    input = torch.rand((1, 1024))
    network = Sequential(
        Linear(in_features=1024, out_features=1024, bias=False),
        Linear(in_features=1024, out_features=1024, bias=False),
    )

    quant_map = QuantizationMap()
    if actq_layer is not None:
        quant_map.input_activation_qconfig_map[f"{actq_layer}"] = ActivationQuantConfig(n_bits=4)

    quant_network = convert_to_quantized(network, quant_map)

    normal_output = network(input)
    quant_output = quant_network(input)

    if actq_layer is None:
        assert torch.allclose(normal_output, quant_output, atol=1e-5)
    else:
        assert not torch.allclose(normal_output, quant_output, atol=1e-5)


@mark.parametrize("quant_last_layer", [True, False])
def test_exclusion(quant_last_layer):
    input = torch.ones((1, 1024))
    network = Sequential(
        Linear(in_features=1024, out_features=1024, bias=False),
        Linear(in_features=1024, out_features=1024, bias=False),
    )

    quant_map = QuantizationMap()
    quant_map.default_qconfig.activation_quant.n_bits = 8
    append_default_conversions(quant_map)

    if not quant_last_layer:
        quant_map.excluded_modules.append("1")

    quant_network = convert_to_quantized(network, quant_map)

    quant_output = quant_network(input)

    output_numel = quant_output.unique().numel()
    if quant_last_layer:
        assert output_numel <= 2**8
    else:
        assert output_numel > 2**8

from torch import Tensor
from torch.nn import Module

from aihwkit.simulator.digital_low_precision.base_quantized_classes import QuantizedActivation
from aihwkit.simulator.digital_low_precision.config_utils import convert_act_config_to_kwargs_dict
from aihwkit.simulator.parameters.quantization import ActivationQuantConfig


class QuantizedInputModule(Module):
    """Wraps a module with an activation quantizer on the inputs, this offering
    the capability to quantize both the inputs and the outputs of a module.
    (NOTE: Output quantization is considered to be taken care of in the activation
    quantization of the module that is being wrapped)

    This is useful when a module is a first layer and consumes directly from the dataloader
    as well as when a layer follows a functional operation (e.g., activation function or addition)
    which did not quantize its own output down to the required size.
    """

    def __init__(self, module: Module, act_quant_config: ActivationQuantConfig):
        super().__init__()
        # The original module
        self.module = module
        # Input quantizer
        self.input_quantizer = QuantizedActivation(
            **convert_act_config_to_kwargs_dict(act_quant_config)
        )

    def forward(self, input: Tensor, *args, **kwargs):
        # Quantize inputs
        input_q = self.input_quantizer(input)
        # Feed the quantized inputs to the original module
        return self.module(input_q, *args, **kwargs)

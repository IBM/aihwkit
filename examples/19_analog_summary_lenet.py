"""aihwkit example 19: Analog summary on LeNet.

Extracts analog information in the form of a printed table.
The information can also be accessed via the returned AnalogInfo object.
"""
# pylint: disable=invalid-name

# Imports from PyTorch.
from torch import nn

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear, AnalogConv2d, AnalogSequential
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.utils.analog_info import analog_summary

# Define a single-layer network, using a constant step device type.
rpu_config = SingleRPUConfig(device=ConstantStepDevice())

channel = [16, 32, 512, 128]
model = AnalogSequential(
        AnalogConv2d(in_channels=1, out_channels=channel[0], kernel_size=5, stride=1,
                     rpu_config=rpu_config),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        AnalogConv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=5, stride=1,
                     rpu_config=rpu_config),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        nn.Tanh(),
        nn.Flatten(),
        AnalogLinear(in_features=channel[2], out_features=channel[3], rpu_config=rpu_config),
        nn.Tanh(),
        AnalogLinear(in_features=channel[3], out_features=10, rpu_config=rpu_config),
        nn.LogSoftmax(dim=1)
    )
# TODO: add mapped examples

# print(model.__class__.__name__)
analog_summary(model, (1, 1, 28, 28))

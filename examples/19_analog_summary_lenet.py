"""aihwkit example 19: Analog summary on LeNet.

Extracts analog information in the form of a printed table.
The information can also be accessed via the returned AnalogInfo object.
"""

# pylint: disable=invalid-name

# Imports from PyTorch.
from torch import nn

# Imports from aihwkit.
from aihwkit.nn.conversion import convert_to_analog_mapped
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice
from aihwkit.utils.analog_info import analog_summary

# Define a single-layer network, using a constant step device type.
rpu_config = SingleRPUConfig(device=ConstantStepDevice())
rpu_config.mapping.max_input_size = 256
rpu_config.mapping.max_output_size = 256


channel = [16, 32, 512, 128]
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=channel[0], kernel_size=5, stride=1),
    nn.Tanh(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=5, stride=1),
    nn.Tanh(),
    nn.MaxPool2d(kernel_size=2),
    nn.Tanh(),
    nn.Flatten(),
    nn.Linear(in_features=channel[2], out_features=channel[3]),
    nn.Tanh(),
    nn.Linear(in_features=channel[3], out_features=10),
    nn.LogSoftmax(dim=1),
)

analog_model = convert_to_analog_mapped(model, rpu_config=rpu_config)

analog_summary(analog_model, (1, 1, 28, 28))

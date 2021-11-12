from torch import nn 

from aihwkit.nn import AnalogLinear, AnalogConv2d, AnalogLinear, AnalogSequential
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.utils.analog_info import analog_summary

# Define a single-layer network, using a constant step device type.
rpu_config = SingleRPUConfig(device=ConstantStepDevice())

model = AnalogConv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1,
                     rpu_config=rpu_config),
     
#print(model.__class__.__name__)
analog_summary(model, (1, 1, 28, 28))
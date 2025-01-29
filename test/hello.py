import torch
import aihwkit
from aihwkit.simulator.configs import (
    RPUDataType,
    InferenceRPUConfig,
    WeightRemapType,
    WeightModifierType,
    WeightClipType,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.inference.compensation.drift import GlobalDriftCompensation, GlobalDriftCompensationWithExactReference
from aihwkit.nn import AnalogLinear

rpu_config = InferenceRPUConfig()
rpu_config.drift_compensation = GlobalDriftCompensation()
# rpu_config.drift_compensation = GlobalDriftCompensationWithExactReference()

model = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config)
model.eval()
model.drift_analog_weights(t_inference=1.0)
model.forward(torch.randn((1, 4)))
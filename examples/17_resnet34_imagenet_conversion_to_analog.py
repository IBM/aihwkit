# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 17: resnet34 CNN conversion to analog.

resnet34 inspired network based on the paper:
https://arxiv.org/abs/1512.03385
"""
# pylint: disable=invalid-name

# Imports from PyTorch.
from torchvision.models import resnet34

# Imports from aihwkit.
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.presets import TikiTakaReRamSBPreset
from aihwkit.simulator.configs import MappingParameter

# Example: Load a predefined model from pytorch library and convert to
#          its analog version.

# Load a pytorch model.
model = resnet34()
print(model)

# Define device and chip configuration used in the RPU tile
mapping = MappingParameter(
    max_input_size=512,  # analog tile size
    max_output_size=512,
    digital_bias=True,
    weight_scaling_omega=0.6,
)  # whether to use analog or digital bias
# Choose any preset or RPU configuration here
rpu_config = TikiTakaReRamSBPreset(mapping=mapping)

# Convert the model to its analog version.  This will replace
# ``Conv2d`` layers with ``AnalogConv2d`` (using simply unfolding for
# convolutions)
model = convert_to_analog(model, rpu_config)

# Note: One can also use ``convert_to_analog_mapped`` instead to
# convert e.g. ``Conv2d`` to ``AnalogConv2dMapped`` (using a special way to
# unfold over multiple tiles in a more memory efficient way
# for some analog tiles on GPU)

print(model)

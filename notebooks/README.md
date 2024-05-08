<!---
Copyright 2021, 2022, 2023, 2024 IBM Analog Hardware Acceleration Kit  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# AIHWKIT Notebooks

You can find here example notebooks provided by the AIHWKIT team.


### Pytorch notebooks

You can open any page of the documentation as a notebook in colab (there is a button directly on said pages) but they are also listed here if you need to:

| Notebook     |      Description      |   |
|:----------|:-------------|------:|
| [Analog training of LeNet5 on ReRam analog device ](https://github.com/IBM/aihwkit/blob/master/notebooks/analog_training_LeNet5_plot.ipynb)  | Training the LeNet5 neural network with MNIST dataset and the Analog SGD optimizer simulated on the analog resistive random-access memory with soft bounds (ReRam) device | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IBM/aihwkit/blob/master/notebooks/analog_training_LeNet5_plot.ipynb) |
| [Analog Training of LeNet5 with the Tiki Taka optimizer and ReRAM analog device](https://github.com/IBM/aihwkit/blob/master/notebooks/analog_training_LeNet5_TT.ipynb)  | Training the LeNet5 neural network with Tiki Taka analog optimizer on MNIST dataset, simulated on the the analog resistive random-access memory with soft bounds (ReRam) device | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IBM/aihwkit/blob/master/notebooks/analog_training_LeNet5_TT.ipynb) |
| [Analog Training of LeNet5 with hardware aware training for inference on a PCM device](https://github.com/IBM/aihwkit/blob/master/notebooks/analog_training_LeNet5_hwa.ipynb)  | Training the LeNet5 neural network with hardware aware training using the Inference RPU Config to optimize inference on a PCM device. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IBM/aihwkit/blob/master/notebooks/analog_training_LeNet5_hwa.ipynb) |
| [Experiments to study the sensitivity of neural networks to non-idealities of Crossbar implementations](https://github.com/IBM/aihwkit/blob/master/notebooks/analog_sensitivity_LeNet5.ipynb)  | Studying the sensitivity of a LeNet5 neural network to synaptic devices and peripheral circuit non-idealities. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IBM/aihwkit/blob/master/notebooks/analog_sensitivity_LeNet5.ipynb) |
| [IBM Analog Fusion chip conversion utility example](https://github.com/IBM/aihwkit/blob/master/notebooks/analog_fusion.ipynb)  | Provide examples of how to convert the weights of a neural network model to a list of conductance values that can be programmed in the Fusion chip. The notebook also shows the errors between the orginal model and the new model after programming the analog chip. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IBM/aihwkit/blob/master/notebooks/analog_fusion.ipynb) |





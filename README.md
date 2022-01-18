# IBM Analog Hardware Acceleration Kit

![PyPI](https://img.shields.io/pypi/v/aihwkit)
[![Documentation Status](https://readthedocs.org/projects/aihwkit/badge/?version=latest)](https://aihwkit.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/IBM/aihwkit.svg?branch=master)](https://travis-ci.com/IBM/aihwkit)
![PyPI - License](https://img.shields.io/pypi/l/aihwkit)
[![arXiv](https://img.shields.io/badge/arXiv-2104.02184-green.svg)](https://arxiv.org/abs/2104.02184)

## Description

_IBM Analog Hardware Acceleration Kit_ is an open source Python toolkit for
exploring and using the capabilities of in-memory computing devices in the
context of artificial intelligence.

> :warning: This library is currently in beta and under active development.
> Please be mindful of potential issues and keep an eye for improvements,
> new features and bug fixes in upcoming versions.

The toolkit consists of two main components:

### Pytorch integration

A series of primitives and features that allow using the toolkit within
[`PyTorch`]:

* Analog neural network modules (fully connected layer, 1d/2d/3d convolution
  layers, LSTM layer, sequential container).
* Analog training using torch training workflow:
  * Analog torch optimizers (SGD).
  * Analog in-situ training using customizable device models and algorithms
    (Tiki-Taka).
* Analog inference using torch inference workflow:
  * State-of-the-art statistical model of a phase-change memory (PCM) array
    calibrated on hardware measurements from a 1 million PCM devices chip.
  * Hardware-aware training with hardware non-idealities and noise
    included in the forward pass to make the trained models more
    robust during inference on Analog hardware.

### Analog devices simulator

A high-performant (CUDA-capable) C++ simulator that allows for
simulating a wide range of analog devices and crossbar configurations
by using abstract functional models of material characteristics with
adjustable parameters. Features include:

* Forward pass output-referred noise and device fluctuations, as well
  as adjustable ADC and DAC discretization and bounds
* Stochastic update pulse trains for rows and columns with finite
  weight update size per pulse coincidence
* Device-to-device systematic variations, cycle-to-cycle noise and
  adjustable asymmetry during analog update
* Adjustable device behavior for exploration of material specifications for
  training and inference
* State-of-the-art dynamic input scaling, bound management, and update
  management schemes

### Other features

Along with the two main components, the toolkit includes other
functionalities such as:

* A library of device presets that are calibrated to real hardware data and
  based on models in the literature, along with configuration that specifies a particular device and optimizer choice.
* A module for executing high-level use cases ("experiments"), such as neural
  network training with minimal code overhead.
* A utility to automatically convert a downloaded model (e.g., pre-trained) to its equivalent Analog
  model by replacing all linear/conv layers to Analog layers (e.g., for convenient hardware-aware training).
* Integration with the [AIHW Composer] platform, a no-code web experience, that allows executing
  experiments in the cloud.

## Example

### Training example

```python
from torch import Tensor
from torch.nn.functional import mse_loss

# Import the aihwkit constructs.
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD

x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = Tensor([[1.0, 0.5], [0.7, 0.3]])

# Define a network using a single Analog layer.
model = AnalogLinear(4, 2)

# Use the analog-aware stochastic gradient descent optimizer.
opt = AnalogSGD(model.parameters(), lr=0.1)
opt.regroup_param_groups(model)

# Train the network.
for epoch in range(10):
    pred = model(x)
    loss = mse_loss(pred, y)
    loss.backward()

    opt.step()
    print('Loss error: {:.16f}'.format(loss))
```

You can find more examples in the [`examples/`] folder of the project, and
more information about the library in the [documentation]. Please note that
the examples have some additional dependencies - you can install them via
`pip install -r requirements-examples.txt`.


## What is Analog AI?

In traditional hardware architecture, computation and memory are siloed in
different locations. Information is moved back and forth between computation
and memory units every time an operation is performed, creating a limitation
called the [von Neumann bottleneck].

Analog AI delivers radical performance improvements by combining compute and
memory in a single device, eliminating the von Neumann bottleneck. By leveraging
the physical properties of memory devices, computation happens at the same place
where the data is stored. Such in-memory computing hardware increases the speed
and energy-efficiency needed for next generation AI workloads.

## What is an in-memory computing chip?

An in-memory computing chip typically consists of multiple arrays of memory
devices that communicate with each other. Many types of memory devices such as
[phase-change memory] (PCM), [resistive random-access memory] (RRAM), and
[Flash memory] can be used for in-memory computing.

Memory devices have the ability to store synaptic weights in their analog
charge (Flash) or conductance (PCM, RRAM) state. When these devices are arranged
in a crossbar configuration, it allows to perform an analog matrix-vector
multiplication in a single time step, exploiting the advantages of analog
storage capability and [Kirchhoff’s circuits laws]. You can learn more about
it in our [online demo].

In deep learning, data propagation through multiple layers of a neural network
involves a sequence of matrix multiplications, as each layer can be represented
as a matrix of synaptic weights. The devices are arranged in multiple crossbar
arrays, creating an artificial neural network where all matrix multiplications
are performed in-place in an analog manner. This structure allows to run deep
learning models at reduced energy consumption.

## How to cite?

In case you are using the _IBM Analog Hardware Acceleration Kit_ for
your research, please cite the AICAS21 paper that describes the toolkit:

> Malte J. Rasch, Diego Moreda, Tayfun Gokmen, Manuel Le Gallo, Fabio Carta,
> Cindy Goldberg, Kaoutar El Maghraoui, Abu Sebastian, Vijay Narayanan.
> "A flexible and fast PyTorch toolkit for simulating training and inference on
> analog crossbar arrays" (2021 IEEE 3rd International Conference on Artificial Intelligence Circuits and Systems)
>
> https://ieeexplore.ieee.org/abstract/document/9458494

## Installation

### Installing from PyPI

The preferred way to install this package is by using the
[Python package index]:

```bash
$ pip install aihwkit
```

> :warning: Note that currently we provide CPU-only pre-built packages for
> specific combinations of architectures and versions, and in some cases a
> pre-built package might still not be available.

If you encounter any issues during download or want to compile the package
for your environment, please refer to the [advanced installation] guide.
That section describes the additional libraries and tools required for
compiling the sources, using a build system based on `cmake`.

## Authors

IBM Analog Hardware Acceleration Kit has been developed by IBM Research,
with Malte Rasch, Tayfun Gokmen, Diego Moreda, Manuel Le Gallo-Bourdeau, and Kaoutar El Maghraoui
as the initial core authors, along with many [contributors].

You can contact us by opening a new issue in the repository, or alternatively
at the ``aihwkit@us.ibm.com`` email address.

## License

This project is licensed under [Apache License 2.0].

[Apache License 2.0]: LICENSE.txt
[`CUDA Toolkit`]: https://developer.nvidia.com/accelerated-computing-toolkit
[`OpenBLAS`]: https://www.openblas.net/
[Python package index]: https://pypi.org/project/aihwkit
[`PyTorch`]: https://pytorch.org/

[`examples/`]: examples/
[documentation]: https://aihwkit.readthedocs.io/
[contributors]: https://github.com/IBM/aihwkit/graphs/contributors
[advanced installation]: https://aihwkit.readthedocs.io/en/latest/advanced_install.html

[von Neumann bottleneck]: https://en.wikipedia.org/wiki/Von_Neumann_architecture#Von_Neumann_bottleneck
[phase-change memory]: https://en.wikipedia.org/wiki/Phase-change_memory
[resistive random-access memory]: https://en.wikipedia.org/wiki/Resistive_random-access_memory
[Flash memory]: https://en.wikipedia.org/wiki/Flash_memory
[Kirchhoff’s circuits laws]: https://en.wikipedia.org/wiki/Kirchhoff%27s_circuit_laws
[online demo]: https://analog-ai-demo.mybluemix.net/
[AIHW Composer]: https://aihw-composer.draco.res.ibm.com

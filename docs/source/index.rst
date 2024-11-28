Welcome to IBM Analog Hardware Acceleration Kit's documentation!
================================================================

.. toctree::
    :maxdepth: 3
    :caption: Get started
    :hidden:


    install
    advanced_install
    using_pytorch
    glossary

.. toctree::
    :maxdepth: 3
    :caption: Analog AI Concepts
    :hidden:

    analog_ai
    analog_ai_hw
    pros_cons


.. toctree::
    :maxdepth: 3
    :caption: Cloud/Composer
    :hidden:


    composer_overview
    using_experiments

.. toctree::
    :maxdepth: 3
    :caption: Using the Simulator
    :hidden:

    using_simulator



.. toctree::
   :maxdepth: 3
   :caption: Analog DNN Training
   :hidden:

   analog_update.rst
   analog_training_presets.rst


.. toctree::
   :maxdepth: 3
   :caption: Analog DNN Inference
   :hidden:

   pcm_inference
   hwa_training

.. toctree::
   :maxdepth: 3
   :caption: Advanced Guides
   :hidden:

   design
   developer_install
   development_conventions
   roadmap
   changelog

.. toctree::
   :maxdepth: 3
   :caption: References
   :hidden:

   api_reference
   paper_references



*IBM Analog Hardware Acceleration Kit* is an open source Python toolkit for
exploring and using the capabilities of in-memory computing devices in the
context of artificial intelligence.

Components
----------

The toolkit consists of two main components:

PyTorch integration
~~~~~~~~~~~~~~~~~~~

A series of primitives and features that allow using the toolkit within PyTorch:

* Analog neural network modules (fully connected layer, 1d/2d/3d convolution
  layers, sequential container).
* Analog training using torch training workflow:

  * Analog torch optimizers (SGD).
  * Analog in-situ training using customizable device models and algorithms
    (Tiki-Taka).
* Analog inference using torch inference workflow:

  * State-of-the-art statistical model of a phase-change memory (PCM) array
    calibrated on hardware measurements from a 1 million PCM devices chip.
  * Hardware-aware training with hardware non-idealities and noise included
    in the forward pass.

Analog devices simulator
~~~~~~~~~~~~~~~~~~~~~~~~

A high-performant (CUDA-capable) C++ simulator that allows for
simulating a wide range of analog devices and crossbar configurations
by using abstract functional models of material characteristics with
adjustable parameters. Feature include:

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

Other features
~~~~~~~~~~~~~~

Along with the two main components, the toolkit includes other functionality:

* A library of device presets that are calibrated to real hardware data and
  device presets that are based on models in the literature, along with config
  preset that specify a particular device and optimizer choice.
* A module for executing high-level use cases ("experiments"), such as neural
  network training with minimal code overhead.
* Integration with the `AIHW Composer`_ platform that allows executing
  experiments in the cloud.

.. warning::
    This library is currently in beta and under active development.
    Please be mindful of potential issues and keep an eye for improvements,
    new features and bug fixes in upcoming versions.

Example
-------

::

    from torch import Tensor
    from torch.nn.functional import mse_loss

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

How to cite
-----------

In case you are using the *IBM Analog Hardware Acceleration Kit* for
your research, please cite the `AICAS21 paper`_ that describes the toolkit:

.. note::

    Malte J. Rasch, Diego Moreda, Tayfun Gokmen, Manuel Le Gallo, Fabio Carta,
    Cindy Goldberg, Kaoutar El Maghraoui, Abu Sebastian, Vijay Narayanan.
    "A flexible and fast PyTorch toolkit for simulating training and inference on
    analog crossbar arrays", 2021 IEEE 3rd International Conference on Artificial Intelligence Circuits and Systems

    https://arxiv.org/abs/2104.02184


Reference
=========

:ref:`genindex` | :ref:`modindex` | :ref:`search`

.. _AIHW Composer: https://aihw-composer.draco.res.ibm.com
.. _arXiv paper: https://arxiv.org/abs/2104.02184
.. _AICAS21 paper: https://ieeexplore.ieee.org/abstract/document/9458494

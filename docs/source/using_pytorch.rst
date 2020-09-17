Using the pytorch integration
=============================

This library exposes most of its higher-level features as `PyTorch`_ primitives,
in order to take advantage of the rest of the PyTorch framework and integrate
analog layers and other features in the regular workflow.

The following table lists the main modules that provide integration with
PyTorch:

=========================  ========
Module                     Notes
=========================  ========
:py:mod:`aihwkit.nn`       Analog Modules (layers) and Functions
:py:mod:`aihwkit.optim`    Analog Optimizers
=========================  ========

Analog layers
-------------

An **analog layer** is a neural network module that stores its weights in an
analog tile. The library current includes the following analog layers:

* :class:`~aihwkit.nn.layers.linear.AnalogLiner`:
  applies a linear transformation to the input data. It is the counterpart
  of PyTorch `nn.Linear`_ layer.

* :class:`~aihwkit.nn.layers.conv.AnalogConv2d`:
  applies a 2D convolution over an input signal composed of several input
  planes. It is the counterpart of PyTorch `nn.Conv2d`_ layer.

Using analog layers
~~~~~~~~~~~~~~~~~~~

The analog layers provided by the library can be used in a similar way to a
standard PyTorch layer, by creating an object. For example, the following
snippet would create a linear layer with 5 input features and 2 output
features::

    from aihwkit.nn import AnalogLinear

    model = AnalogLinear(5, 3)

By default, the ``AnalogLinear`` layer would use bias, and use a
:class:`~aihwkit.simulator.tiles.FloatingPointTile` tile as the underlying
tile for the analog operations. These values can be modified by passing
additional arguments to the constructor.

The analog layers will perform the ``forward`` and ``backward`` passes directly
in the underlying tile.

Overall, the layer can be combined and used as if it was a standard torch
layer. As an example, it can be mixed with existing layers::

        from aihwkit.nn import AnalogLinear
        from torch.nn import Linear, Sequential

        model = Sequential(
            AnalogLinear(2, 3),
            Linear(3, 3),
            AnalogLinear(3, 1)
        )

.. note::

    When using analog layers, please be aware that the ``Parameters`` of the
    layers (``model.weight`` and ``model.bias``) are not guaranteed to be in
    sync with the actual weights and biased used internally by the analog
    tile, as reading back the weights has a performance cost. If you need to
    ensure that the tensors are synced, please use the
    :meth:`~aihwkit.nn.modules.linear.AnalogLinear.sync_parameters_from_tile`
    method.


Customizing the analog tile properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The snippet from the previous section can be extended for specifying that the
underlying analog tile should use a ``ConstantStep`` resistive device, with
a specific value for one of its parameters (``w_min``)::

    from aihwkit.nn import AnalogLinear
    from aihwkit.simulator.devices import ConstantStepResistiveDevice
    from aihwkit.simulator.parameters import PulsedResistiveDeviceParameters

    parameters = PulsedResistiveDeviceParameters(w_min=-0.4)
    resistive_device = ConstantStepResistiveDevice(parameters)
    model = AnalogLinear(5, 3, bias=False, resistive_device=resistive_device)


You can read more about analog tiles in the :doc:`using_simulator` section.

Using CUDA
~~~~~~~~~~

If your version of the library is compiled with CUDA support, you can use
GPU-aware analog layers for improved performance::

    model = model.cuda()

This would move the layers parameters (weights and biases tensors) to CUDA
tensors, and move the analog tiles of the layers to a CUDA-enabled analog
tile.

Optimizers
----------

An **analog optimizer** is a representation of an algorithm that determines
the training strategy taking into account the particularities of the analog
layers involved. The library currently includes the following optimizers:

* :class:`~aihwkit.optim.analog_sgd.AnalogSGD`:
  implements stochastic gradient descent for analog layers. It is the
  counterpart of PyTorch `optim.SGD`_ optimizer.

Using analog optimizers
~~~~~~~~~~~~~~~~~~~~~~~

The analog layers provided by the library can be used in a similar way to a
standard PyTorch layer, by creating an object. For example, the following
snippet would create an analog-aware stochastic gradient descent optimizer
with a learning rate of ``0.1``, and set it up for using with the
analog layers of the model::

    from aihwkit.optim.analog_sgd import AnalogSGD

    optimizer = AnalogSGD(model.parameters(), lr=0.1)
    optimizer.regroup_param_groups(model)


.. note::

    The :meth:`aihwkit.optim.analog_sgd.AnalogSGD.regroup_param_groups` method
    needs to be invoked in order to set up the parameter groups, as they are
    used for handling the analog layers correctly.

The ``AnalogSGD`` optimizer will behave in the same way as the regular
``nn.SGD`` optimizer for non-analog layers in the model. For the analog layers,
the updating of the weights is performed directly in the underlying analog
tile, according to the properties set for that particular layer.

Training example
----------------

The following example combines the usage of analog layers and analog optimizer
in order to perform training::

    from torch import Tensor
    from torch.nn.functional import mse_loss

    from aihwkit.nn import AnalogLinear
    from aihwkit.optim.analog_sgd import AnalogSGD

    x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
    y = Tensor([[1.0, 0.5], [0.7, 0.3]])

    model = AnalogLinear(4, 2)
    optimizer = AnalogSGD(model.parameters(), lr=0.1)
    optimizer.regroup_param_groups(model)

    for epoch in range(10):
        pred = model(x)
        loss = mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        print("Loss error: " + str(loss))


.. _PyTorch: https://pytorch.org
.. _nn.Linear: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
.. _nn.Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
.. _optim.SGD: https://pytorch.org/docs/stable/optim.html#torch.optim.SGD

Using the PyTorch integration
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

Convolution layers
~~~~~~~~~~~~~~~~~~

+--------------------------------------------------------------+-----------------------------------------------------+---------------------+
| Analog Layer                                                 | Description                                         | PyTorch Counterpart |
+==============================================================+=====================================================+=====================+
| :class:`~aihwkit.nn.modules.linear.AnalogLinear`             | | Applies a linear transformation to the input data | `nn.Linear`_        |
|                                                              | | (layers) and Functions.                           |                     |
+--------------------------------------------------------------+-----------------------------------------------------+---------------------+
| :class:`~aihwkit.nn.modules.conv.AnalogConv1d`               | | Applies a 1D convolution over an input signal     | `nn.Conv1d`_        |
|                                                              | | composed of several input planes.                 |                     |
+--------------------------------------------------------------+-----------------------------------------------------+---------------------+
| :class:`~aihwkit.nn.modules.conv.AnalogConv2d`               | | Applies a 2D convolution over an input signal     | `nn.Conv2d`_        |
|                                                              | | composed of several input planes                  |                     |
+--------------------------------------------------------------+-----------------------------------------------------+---------------------+
| :class:`~aihwkit.nn.modules.conv.AnalogConv3d`               | | Applies a 3D convolution over an input signal     | `nn.Conv3d`_        |
|                                                              | | composed of several input planes                  |                     |
+--------------------------------------------------------------+-----------------------------------------------------+---------------------+
| :class:`~aihwkit.nn.modules.linear_mapped.AnalogLinearMapped`| | Similar to AnalogLinearMapped but constrains the  | `nn.Linear`_        |
|                                                              | | the maximal in and/or out dimension of an analog  |                     |
|                                                              | | tile and will construct multiple tiles (as many   |                     |
|                                                              | | as necessary to cover the weight matrix).         |                     |
|                                                              | | Splitting, concatenation, and partial sum addition|                     |
|                                                              | | are done in digital.                              |                     |
+--------------------------------------------------------------+-----------------------------------------------------+---------------------+
| :class:`~aihwkit.nn.modules.conv_mapped.AnalogConv1dMapped`  | | Applies a 1D convolution over an input signal     | `nn.Conv1d`_        |
|                                                              | | composed of several inputplanes, using an analog  |                     | 
|                                                              | | tile for its forward, backward and update passes. |                     |  
|                                                              | | The module will split the weight                  |                     | 
|                                                              | | matrix onto multiple tiles if necessary.          |                     |
+--------------------------------------------------------------+-----------------------------------------------------+---------------------+
| :class:`~aihwkit.nn.modules.conv_mapped.AnalogConv2dMapped`  | | Applies a 2D convolution over an input signal     | `nn.Conv2d`_        |
|                                                              | | composed of several inputplanes, using an analog  |                     | 
|                                                              | | tile for its forward, backward and update passes. |                     |  
|                                                              | | The module will split the weight                  |                     | 
|                                                              | | matrix onto multiple tiles if necessary.          |                     |
+--------------------------------------------------------------+-----------------------------------------------------+---------------------+
| :class:`~aihwkit.nn.modules.conv_mapped.AnalogConv3dMapped`  | | Applies a 3D convolution over an input signal     | `nn.Conv3d`_        |
|                                                              | | composed of several inputplanes, using an analog  |                     | 
|                                                              | | tile for its forward, backward and update passes. |                     |  
|                                                              | | The module will split the weight                  |                     | 
|                                                              | | matrix onto multiple tiles if necessary.          |                     |
+--------------------------------------------------------------+-----------------------------------------------------+---------------------+

Recurrent layers
~~~~~~~~~~~~~~~~

+-------------------------------------------------------------+-----------------------------------------------------+---------------------+
| Analog Layer                                                | Description                                         | PyTorch Counterpart |
+=============================================================+=====================================================+=====================+
| :class:`~aihwkit.nn.modules.rnn.rnn.AnalogRNN`              | | A modular RNN that uses analog tiles. Can take    |  | `nn.RNN`_        |
|                                                             | | one of three types: AnalogLSTM, AnalogGRU, or     |  | `nn.LSTM`_       |
|                                                             | | AnalogVanillaRNN                                  |  | `nn.GRU`_        |
+-------------------------------------------------------------+-----------------------------------------------------+---------------------+
| :class:`~aihwkit.nn.modules.rnn.cells.AnalogVanillaRNNCell` | An Elman RNN cell with tanh or ReLU non-linearity.  | `nn.RNNCell`_       |
+-------------------------------------------------------------+-----------------------------------------------------+---------------------+
| :class:`~aihwkit.nn.modules.rnn.cells.AnalogGRUCell`        | A gated recurrent unit (GRU) cell.                  | `nn.GRUCell`_       |
+-------------------------------------------------------------+-----------------------------------------------------+---------------------+
| :class:`~aihwkit.nn.modules.rnn.cells.AnalogLSTMCell`       | A long short-term memory (LSTM) cell.               | `nn.LSTMCell`_      |
+-------------------------------------------------------------+-----------------------------------------------------+---------------------+

Using analog layers
~~~~~~~~~~~~~~~~~~~

The analog layers provided by the library can be used in a similar way to a
standard PyTorch layer, by creating an object. For example, the following
snippet would create a linear layer with 5 input features and 2 output
features::

    from aihwkit.nn import AnalogLinear

    model = AnalogLinear(5, 3)

By default, the ``AnalogLinear`` layer would use bias, and use a
:class:`~aihwkit.simulator.tiles.floating_point.FloatingPointTile` tile as the
underlying tile for the analog operations. These values can be modified by
passing additional arguments to the constructor.

The analog layers will perform the ``forward`` and ``backward`` passes directly
in the underlying tile.

Overall, the layer can be combined and used as if it was a standard torch
layer. As an example, it can be mixed with existing layers::

        from aihwkit.nn import AnalogLinear, AnalogSequential
        from torch.nn import Linear

        model = AnalogSequential(
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
    :meth:`~aihwkit.nn.modules.base.AnalogModuleBase.set_weights` and
    :meth:`~aihwkit.nn.modules.base.AnalogModuleBase.get_weights` methods.


Customizing the analog tile properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The snippet from the previous section can be extended for specifying that the
underlying analog tile should use a ``ConstantStep`` resistive device, with
a specific value for one of its parameters (``w_min``)::

    from aihwkit.nn import AnalogLinear
    from aihwkit.simulator.configs import SingleRPUConfig
    from aihwkit.simulator.configs.devices import ConstantStepDevice

    config = SingleRPUConfig(device=ConstantStepDevice(w_min=-0.4))
    model = AnalogLinear(5, 3, bias=False, rpu_config=config)


You can read more about analog tiles in the :doc:`using_simulator` section.

Using CUDA
~~~~~~~~~~

If your version of the library is compiled with CUDA support, you can use
GPU-aware analog layers for improved performance::

    model = model.cuda()

This would move the layers parameters (weights and biases tensors) to CUDA
tensors, and move the analog tiles of the layers to a CUDA-enabled analog
tile.

.. note::

    Note that if you use analog layers that are children of other modules,
    some of the features require manually performing them on the analog layers
    directly (instead of only on the parent module).
    Please check the rest of the document for more information about using
    :class:`~aihwkit.nn.modules.container.AnalogSequential` as the parent class
    instead of ``nn.Sequential``, for convenience.

Optimizers
----------

An **analog optimizer** is a representation of an algorithm that determines
the training strategy taking into account the particularities of the analog
layers involved. The library currently includes the following optimizers:

* :class:`~aihwkit.optim.analog_optimizer.AnalogSGD`:
  implements stochastic gradient descent for analog layers. It is the
  counterpart of PyTorch `optim.SGD`_ optimizer.

Using analog optimizers
~~~~~~~~~~~~~~~~~~~~~~~

The analog layers provided by the library can be used in a similar way to a
standard PyTorch layer, by creating an object. For example, the following
snippet would create an analog-aware stochastic gradient descent optimizer
with a learning rate of ``0.1``, and set it up for using with the
analog layers of the model::

    from aihwkit.optim import AnalogSGD

    optimizer = AnalogSGD(model.parameters(), lr=0.1)
    optimizer.regroup_param_groups(model)


.. note::

    The :meth:`~aihwkit.optim.analog_optimizer.AnalogSGD.regroup_param_groups` method
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
    from aihwkit.optim import AnalogSGD

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


Using analog layers as part of other modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using analog layers in other modules, you can use the usual torch
mechanisms for including them as part of the model.

However, as a number of torch functions are applied only to the parameters and
buffers of a regular module, in some cases they would need to be applied
directly to the analog layers themselves (as opposed to applying the parent
container).

In order to bypass the need of applying the functions to the analog layers,
you can use the :class:`~aihwkit.nn.modules.container.AnalogSequential` as both
a compatible replacement for ``nn.Sequential``, and as the superclass in case
of custom analog modules. By using this convenience module, the operations are
guaranteed to be applied correctly to its children. For example::

    from aihwkit.nn import AnalogLinear, AnalogSequential

    model = AnalogSequential(
        AnalogLinear(10, 20)
    )
    model.cuda()
    model.eval()
    model.program_analog_weights()

Or in the case of custom classes::

    from aihwkit.nn import AnalogConv2d, AnalogSequential

    class Example(AnalogSequential):

        def __init__(self):
            super().__init__()

            self.feature_extractor = AnalogConv2d(
                in_channels=1, out_channels=16, kernel_size=5, stride=1
            )


.. _PyTorch:     https://pytorch.org
.. _nn.Linear:   https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
.. _nn.Conv1d:   https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
.. _nn.Conv2d:   https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
.. _nn.Conv3d:   https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
.. _optim.SGD:   https://pytorch.org/docs/stable/optim.html#torch.optim.SGD
.. _nn.RNN:      https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN
.. _nn.LSTM:     https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
.. _nn.GRU:      https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU
.. _nn.RNNCell:  https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html#torch.nn.RNNCell
.. _nn.GRUCell:  https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html#torch.nn.GRUCell
.. _nn.LSTMCell: https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html#torch.nn.LSTMCell

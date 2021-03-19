Using Experiments
=================

Since version ``0.3``, the toolkit includes support for running ``Experiments``.
An **Experiment** represents a high-level use case, such as training a neural
network, in a compact form that allows for easily running the experiment and
variations of it with ease both locally and remotely.

Experiments
-----------

The following types of Experiments are available:

=================================================================  ========
Tile class                                                         Description
=================================================================  ========
:class:`~aihwkit.experiments.experiments.training.BasicTraining`    Simple training of a neural network
=================================================================  ========

Creating an Experiment
^^^^^^^^^^^^^^^^^^^^^^

An Experiment can be created just by creating an instance of its class::

    from torchvision.datasets import FashionMNIST

    from torch.nn import Flatten, LogSoftmax, Sigmoid
    from aihwkit.nn import AnalogLinear, AnalogSequential

    from aihwkit.experiments import BasicTraining


    my_experiment = BasicTraining(
        dataset=FashionMNIST,
        model=AnalogSequential(
            Flatten(),
            AnalogLinear(784, 256, bias=True),
            Sigmoid(),
            AnalogLinear(256, 128, bias=True),
            Sigmoid(),
            AnalogLinear(128, 10, bias=True),
            LogSoftmax(dim=1)
        )
    )

Each Experiment has its own attributes, providing sensible defaults as needed.
For example, the ``BasicTraining`` Experiment allows setting attributes that
define the characteristics of the training (``dataset``, ``model``,
``batch_size``, ``loss_function``, ``epochs``, ``learning_rate``).

The created Experiment contains the definition of the operation to be performed,
but is not executed automatically: that is the role of the ``Runners``.

Runners
-------

A **Runner** is the object that controls the execution of an Experiment,
setting up the environment and providing a convenient way of starting it and
retrieving its results.

The following types of Runners are available:

========================================================  ========
Tile class                                                Description
========================================================  ========
:class:`~aihwkit.experiments.runners.local.LocalRunner`   Runner for executing experiments locally
========================================================  ========

Running an Experiment Locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to run an Experiment, the first step is creating the appropriate
runner::

    from aihwkit.experiments.runners import LocalRunner

    my_runner = LocalRunner()

.. note::

    Each runner has different configurations options depending on their type.
    For example, the ``LocalRunner`` has an option for setting the device where
    the model will be executed into, that can be used for using GPU::

        from torch import device as torch_device

        my_runner = LocalRunner(device=torch_device('cuda'))

Once the runner is created, the Experiment can be executed via::

    result = my_runner.run(my_experiment)

This will start the desired experiment, and return the results of the
experiment - in the training case, a dictionary containing the metrics for each
epoch::

    > print(result)

    [{
      'epoch': 0,
      'accuracy': 0.8289,
      'train_loss': 0.4497026850991666,
      'valid_loss': 0.07776954893999771
     },
     {
      'epoch': 1,
      'accuracy': 0.8299,
      'train_loss': 0.43052176381352103,
      'valid_loss': 0.07716381718227858
     },
     {
      'epoch': 2,
      'accuracy': 0.8392,
      'train_loss': 0.41551961805393445,
      'valid_loss': 0.07490375201140385
     },
     ...
    ]

The local runner will also print information by default while the experiment
is being executed (for example, if running the experiment in an interactive
session, as a way of tracking progress). This can be turned off by the
``stdout`` argument to the ``run()`` function::

    result = my_runner.run(my_experiment, stdout=False)

.. note::

    The local runner will automatically attempt to download the dataset if it
    is ``FashionMNIST`` or ``SVHN`` into a temporary folder. For other datasets,
    please ensure that the dataset is downloaded previously, using the
    ``dataset_root`` argument to indicate the location of the data files::

        result = my_runner.run(my_experiment, dataset_root='/some/path')


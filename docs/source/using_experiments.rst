Composer CLI
=================

Since version ``0.3``, the toolkit includes support for running ``Experiments``.
An **Experiment** represents a high-level use case, such as training a neural
network, in a compact form that allows for easily running the experiment and
variations of it with ease both locally, in the cloud and its variations.

Experiments
-----------

The following types of Experiments are available:

=====================================================================  ========
Tile class                                                             Description
=====================================================================  ========
:class:`~aihwkit.experiments.experiments.training.BasicTraining`       Simple training of a neural network
:class:`~aihwkit.experiments.experiments.training.BasicInferencing`    Simple inference of a neural network
=====================================================================  ========

Creating an Experiment
^^^^^^^^^^^^^^^^^^^^^^

A :class:`~aihwkit.experiments.experiments.training.BasicTraining`  Experiment can be created just by creating an instance of its class::

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

Similarly a :class:`~aihwkit.experiments.experiments.training.BasicInferencing`  Experiment can also be created by creating an instance of the class ::

    from torch.nn import (
    Flatten, LogSoftmax, MaxPool2d, Module, Tanh
    )
    from torchvision.datasets import FashionMNIST

    from aihwkit.nn import AnalogConv2dMapped, AnalogLinearMapped, AnalogSequential

    from aihwkit.experiments.experiments.inferencing import BasicInferencing

    DATASET = FashionMNIST
    MODEL = create_analog_lenet5_network()
    BATCH_SIZE = 8
    REPEATS = 2
    I_TIMES = 86400
    TEMPLATE_ID = 'hwa-trained-lenet5-mapped'

    my_experiment = BasicInferencing(
      dataset=DATASET,
      model = MODEL,
      batch_size = BATCH_SIZE,
      weight_template_id = TEMPLATE_ID,
      inference_repeats = REPEATS,
      inference_time = I_TIMES
    )


Each Experiment has its own attributes, providing sensible defaults as needed.
For example, the ``BasicTraining`` Experiment allows setting attributes that
define the characteristics of the training (``dataset``, ``model``,
``batch_size``, ``loss_function``, ``epochs``, ``learning_rate``).

Similarly the ``BasicInferencing`` Experiment allows setting attributes
that define the characteristics of the Inferencing experiment (``dataset``,
``model``, ``batch_size`` , ``inference_repeats`` , ``inference_time``)

The created Experiment contains the definition of the operation to be performed,
but is not executed automatically.  That is the role of the ``Runners``.

Runners
-------

A **Runner** is the object that controls the execution of an Experiment,
setting up the environment and providing a convenient way of starting it and
retrieving its results.

The following types of Runners are available:

===================================================================  ========
Tile class                                                           Description
===================================================================  ========
:class:`~aihwkit.experiments.runners.local.LocalRunner`              Runner for executing training experiments locally
:class:`~aihwkit.experiments.runners.cloud.CloudRunner`              Runner for executing training experiments in the cloud
:class:`~aihwkit.experiments.runners.i_local.InferenceLocalRunner`   Runner for executing inference experiments locally
:class:`~aihwkit.experiments.runners.i_cloud.InferenceCloudRunner`   Runner for executing inference experiments in the cloud
===================================================================  ========

Running an Experiment Locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to run an Experiment, the first step is creating the appropriate
runner, for executing a ``training exepriment`` locally we create :class:`~aihwkit.experiments.runners.local.LocalRunner`  ::

    from aihwkit.experiments.runners import LocalRunner

    my_runner = LocalRunner()

Similarly for executing a ``Inferencing Experimnet`` locally  we create :class:`~aihwkit.experiments.runners.i_local.InferenceLocalRunner` ::

    from aihwkit.experiments.runners import InferenceLocalRunner

    my_runner = InferenceLocalRunner()

.. note::

    Each runner has different configurations options depending on their type.
    For example, the ``LocalRunner`` has an option for setting the device where
    the model will be executed into, that can be used for using GPU::

        from torch import device as torch_device

        my_runner = LocalRunner(device=torch_device('cuda'))

    Similarly , the ``InferenceLocalRunner`` has also an option for setting the device
    when the model would be used for inferencing , for using the available GPU ::

        from torch import device as torch_device

        my_runner - InferenceLocalRunner(device=torch_device('cuda'))

Once the runner is created for either ``Training`` or ``Inferencing`` experiment , the Experiment can be executed via::

    result = my_runner.run(my_experiment)

This will start the desired experiment, and return the results of the
experiment - in the training case, a dictionary containing the metrics for each
epoch::

     print(result)

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

The ``LocalRunner`` for ``Training`` experiment and ``InferenceLocalRunner`` for ``Inferencing`` experiment
will also print information by default while the experiment is being executed (for example, if running
the experiment in an interactive session, as a way of tracking progress). This can be turned off by the
``stdout`` argument to the ``run()`` function::

    result = my_runner.run(my_experiment, stdout=False)

.. note::

    The local runner for both ``Training`` and ``Inferencing`` type of experiments
    will automatically attempt to download the dataset if it
    is ``FashionMNIST`` or ``SVHN`` into a temporary folder. For other datasets,
    please ensure that the dataset is downloaded previously, using the
    ``dataset_root`` argument to indicate the location of the data files::

        result = my_runner.run(my_experiment, dataset_root='/some/path')

Cloud Runner
------------

Experiments can also be run in the cloud at our companion `AIHW Composer`_
application, that allows for executing the experiments remotely using hardware
acceleration and inspect the experiments and their results visually, along
other features.

Setting up your account
^^^^^^^^^^^^^^^^^^^^^^^

The integration is provided by a Python client included in ``aihwkit`` that
allows connecting to the `AIHW Composer`_ platform. In order to be able to
run experiments in the cloud:

1. Register in the platform and generate an `API token`_ in your user page.
   This token acts as the credentials for connecting with the application.

2. Store your credentials by creating a ``~/.config/aihwkit.conf`` file with
   the following contents, replacing ``YOUR_API_TOKEN`` with the string
   from the previous step::

    [cloud]
    api_token = YOUR_API_TOKEN

3. You may need to download the SSL certificates and add them to the certificate store.

   - https://cacerts.digicert.com/DigiCertGlobalRootCA.crt.pem
   - https://cacerts.digicert.com/DigiCertTLSRSASHA2562020CA1-1.crt.pem
   - Append the certificates to the cacert.pem file

.. note::

   You can run the following command to find the location of the cacert.pem file

   $ python -c "import certifi; print(certifi.where())“

Running an Experiment in the cloud
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once your credentials are configured, running ``training`` experiments in the cloud can
be performed by using the ``CloudRunner``, in an analogous way as running
experiments locally::

    from aihwkit.experiments.runners import CloudRunner

    my_cloud_runner = CloudRunner()
    cloud_experiment = my_cloud_runner.run(my_experiment)

Similarly ``Inferencing`` experiment can also be performed in the cloud by using
the ``InferenceCloudRunner`` , in an analogous way as running experiments locally ::

    from aihwkit.experiments.runners import InferenceCloudRunner

    cloud_runner = InferenceCloudRunner()
    cloud_experiment = cloud_runner.run(my_experiment, analog_info,
                                    noise_model_info, name=NAME, device='gpu')

Instead of waiting for the experiment to be completed, the ``run()`` method
returns an object that represents a job in the cloud. As such, it has several
convenience methods:

Checking the status of a cloud experiment
"""""""""""""""""""""""""""""""""""""""""

The status of a cloud experiment for both ``Training`` and ``Inferencing`` experiments can be retrieved via::

    cloud_experiment.status()

The response will provide information about the cloud experiment:
    * ``WAITING``: if the experiment is waiting to be processed.
    * ``RUNNING``: when the experiment is being executed in the cloud.
    * ``COMPLETED``: if the experiment was executed successfully.
    * ``FAILED``: if there was an error during the execution of the experiment.

.. note::

    Some actions are only possible if the cloud experiment has finished
    successfully, for example, retrieving its results. Please also be mindful
    that some experiments can take a sizeable amount of time to be executed,
    specially during the initial versions of the platform.

Retrieving the results of a cloud experiment
""""""""""""""""""""""""""""""""""""""""""""

Once the cloud experiment (``Training`` or ``Inferencing``) completes its execution, its results can be retrieved
using::

    result = cloud_experiment.get_result()

This will display the result of executing the experiment, in a similar form as
the output of running an Experiment locally.

Retrieving the content of the experiment
""""""""""""""""""""""""""""""""""""""""

The Experiment can be retrieved using::

    experiment = cloud_experiment.get_experiment()

This will return a local Experiment (for example, a ``BasicTraining`` or ``BasicInferencing``) that
can be used locally and their properties inspected. In particular, the weights
of the model will reflect the results of the experiment.

Retrieving a previous cloud experiment
""""""""""""""""""""""""""""""""""""""

The list of experiments previously executed in the cloud can be retrieved via::

    cloud_experiments = my_cloud_runner.list_experiments()

Please see https://github.com/IBM/aihwkit/tree/master/notebooks/cli for the experiment example notebooks.


.. _AIHW Composer: https://aihw-composer.draco.res.ibm.com/
.. _API token: https://aihw-composer.draco.res.ibm.com/account

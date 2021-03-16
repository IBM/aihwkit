Installation
============

The preferred way to install this package is by using the `Python package index`_::

    pip install aihwkit


.. note::
    During the initial beta stage, we do not provide pip *wheels* (as in,
    pre-compiled binaries) for all the possible platform, version and
    architecture combinations (in particular, only CPU versions are provided).

    Please refer to the :doc:`advanced_install` page for instruction on how to
    compile the library for your environment in case you encounter errors during
    installing from pip.

The package require the following runtime libraries to be installed in your
system:

* `OpenBLAS`_: 0.3.3+
* `CUDA Toolkit`_: 9.0+ (only required for the GPU-enabled simulator [#f1]_)

.. note::
    Please note that the current pip wheels are only compatible with ``PyTorch``
    ``1.6.0``. If you need to use a different ``PyTorch`` version, please
    refer to the :doc:`advanced_install` section in order to compile a custom
    version. More details about the ``PyTorch`` compatibility can be found in
    this `issue`_.

Optional features
-----------------

The package contains optional functionality that is not installed as part of
the default installed. In order to install the extra dependencies, the
recommended way is by specifying the extra ``visualization`` dependencies::

    pip install aihwkit[visualization]

Verifying the installation
--------------------------

If the library was installed correctly, you can use the following snippet for
creating an analog layer and predicting the output::

    from torch import Tensor
    from aihwkit.nn import AnalogLinear

    model = AnalogLinear(3, 2)
    model(Tensor([[0.1, 0.2], [0.3, 0.4]]))

If you encounter any issues during the installation or executing the snippet,
please refer to the :doc:`advanced_install` section for more details and don't
hesitate on using the `issue tracker`_ for additional support.

Next steps
----------

You can read more about the PyTorch layers in the :doc:`using_pytorch`
section, and about the internal analog tiles in the :doc:`using_simulator`
section.

.. [#f1] Note that GPU support is not available in OSX, as it depends on a
   platform that has official CUDA support.

.. _OpenBLAS: https://www.openblas.net
.. _CUDA Toolkit: https://developer.nvidia.com/accelerated-computing-toolkit
.. _issue tracker: https://github.com/IBM/aihwkit/issues
.. _issue: https://github.com/IBM/aihwkit/issues/52
.. _Python package index: https://pypi.org/project/aihwkit/

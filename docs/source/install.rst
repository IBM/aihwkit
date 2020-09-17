Installation
============

The preferred way to install this package is by using the Python package index::

    pip install aihwkit


.. note::
    During the initial beta stage, we do not provide pip *wheels* (as in,
    pre-compiled binaries) for all the possible platform, version and CUDA
    architecture.

    Please refer to the :doc:`advanced_install` page for instruction on how to
    compile the library for your environment in case you encounter errors during
    installing from pip.

The packages require the following runtime libraries to be installed in your
system:

* `OpenBLAS`_: 0.3.3+
* `CUDA Toolkit`_: 9.0+ (only required for the GPU-enabled simulator [#f1]_)

Verifying the installation
--------------------------

If the library was installed correctly, you can use the following snippet for
creating an analog layer and predicting the output::

    from torch import Tensor
    from aihwkit.nn import AnalogLinear

    model = AnalogLinear(3, 2)
    model(Tensor([[0.1, 0.2], [0.3, 0.4]]))

You can read more about the Pytorch layers in the :doc:`using_pytorch`
section, and about the internal analog tiles in the :doc:`using_simulator`
section.

.. _OpenBLAS: https://www.openblas.net
.. _CUDA Toolkit: https://developer.nvidia.com/accelerated-computing-toolkit


.. [#f1] Note that GPU support is not available in OSX, as it depends on a
   platform that has official CUDA support.

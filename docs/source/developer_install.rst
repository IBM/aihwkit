Development setup
=================

This section is a complement to the :doc:`advanced_install` section, with
the goal of setting up a development environment and a development version
of the package.

For convenience, we suggest creating a `virtual environment`_ as a way to
isolate your development environment::

    $ python3 -m venv aihwkit_env
    $ cd aihwkit_env
    $ source bin/activate
    (aihwkit_env) $

Downloading the source
^^^^^^^^^^^^^^^^^^^^^^

The first step is downloading the source of the library::

    (aihwkit_env) $ git clone https://github.com/IBM/aihwkit.git
    (aihwkit_env) $ cd aihwkit

.. note::

    The following sections assume that the command line examples are executed
    in the activated ``aihwkit_env`` environment, and from the folder where the
    sources have been cloned.

Compiling the library for development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After installing the requirements listed in the :doc:`advanced_install` section,
the shared library can be compiled using the following convenience command::

    $ python setup.py build_ext --inplace

This will produce a shared library under the ``src/aihwkit/simulator``
directory, without installing the package.

As an alternative, you can use ``cmake`` directly for
finer control over the compilation and for easier debugging potential issues::

    $ mkdir build
    $ cd build
    build$ cmake ..
    build$ make

Compilation flags
^^^^^^^^^^^^^^^^^

There are several ``cmake`` options that can be used for customizing the
compilation process:

==========================  ================================================  =======
Flag                        Description                                       Default
==========================  ================================================  =======
``USE_CUDA``                Build with CUDA support                           ``OFF``
``BUILD_TEST``              Build the C++ test binaries                       ``OFF``
``RPU_BLAS``                BLAS backend of choice (``OpenBLAS`` or ``MKL``)  ``OpenBLAS``
``RPU_USE_FASTMOD``         Use fast mod                                      ``ON``
``RPU_USE_FASTRAND``        Use fastrand                                      ``OFF``
``RPU_CUDA_ARCHITECTURES``  Target CUDA architectures                         ``60``
==========================  ================================================  =======

The options can be passed both to ``setuptools`` or to ``cmake`` directly. For
example, for compiling with CUDA support::

    $ python setup.py install -DUSE_CUDA=ON -DRPU_CUDA_ARCHITECTURES="60;70"

or if using ``cmake`` directly::

    build$ cmake -DUSE_CUDA=ON -DRPU_CUDA_ARCHITECTURES="60;70" ..


.. note::
    If you are installing the package in editable mode for development (via.
    ``pip install -e .``), please be aware that under most circumstances the
    actual package sources will not be appended to the Python path. You might
    need to add the ``src/`` folder to your ``PYTHONPATH`` accordingly (for
    example, via ``export PYTHONPATH=src/``).


.. _virtual environment: https://docs.python.org/3/library/venv.html

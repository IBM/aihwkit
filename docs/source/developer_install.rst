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

Note that the build system uses a temporary ``_skbuild`` folder for caching
some steps of the compilation. While this is useful when making changes to
the source code, in some cases environment changes (such as installing a new
version of the dependencies, or switching the compiler) are not picked up
correctly and the output of the compilation can be different than expected
if the folder is present.

If the compilation was not successful, it is recommended to manually remove the
folder and re-run the compilation in a clean state via::

    $ make clean

Using the compiled version of the library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the library is compiled, the shared library will be created under the
``src/aihwkit/simulator`` directory. By default, this folder is not in the path
that Python uses for finding modules: it needs to be added to the
``PYTHONPATH`` accordingly by either:

1. Updating the environment variable for the session::

    $ export PYTHONPATH=src/

2. Prepending ``PYTHONPATH=src/`` to the commands where the library needs to
   be found::

    $ PYTHONPATH=src/ python examples/01_simple_layer.py

.. note::

    Please be aware that, if the ``PYTHONPATH`` is not modified and there is a
    version of ``aihkwit`` installed via ``pip``, by default Python will use
    the installed version, as opposed to the custom-compiled version. It is
    recommended to remove the pip-installed version via::

        $ pip uninstall aihwkit

    when developing the library, in order to minimize the risk of confusion.

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
example, for compiling and installing with CUDA support::

    $ python setup.py build_ext --inplace -DUSE_CUDA=ON -DRPU_CUDA_ARCHITECTURES="60;70"

or if using ``cmake`` directly::

    build$ cmake -DUSE_CUDA=ON -DRPU_CUDA_ARCHITECTURES="60;70" ..

Passing other ``cmake`` flags
"""""""""""""""""""""""""""""

In the same way flags specific to this project can be passed to ``setup.py``,
other generic ``cmake`` flags can be passed as well. For example, for setting
the compiler to ``clang`` in osx systems::

    $ python setup.py build_ext --inplace -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++


.. _virtual environment: https://docs.python.org/3/library/venv.html

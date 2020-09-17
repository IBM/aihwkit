Advanced installation guide
===========================

Compilation
-----------

The build system for ``aihwkit`` is based on `cmake`_, making use of
scikit-build_ for generating the Python packages.

Some of the dependencies and tools are Python-based. For convenience, we
suggest creating a `virtual environment`_ as a way to isolate your
environment::

    $ python3 -m venv aihwkit_env
    $ cd aihwkit_env
    $ source bin/activate
    (aihwkit_env) $

.. note::

    The following sections assume that the command line examples are executed
    in the activated ``aihwkit_env`` environment.

Dependencies
~~~~~~~~~~~~

For compiling ``aihwkit``, the following dependencies are required:

===============================  ========  ======
Dependency                       Version   Notes
===============================  ========  ======
C++11 compatible compiler
`cmake`_                         3.18+
`pybind11`_                      2.5.0+    Installing from ``pip`` is not supported [#f1]_
`scikit-build`_                  0.11.0+
`Python 3 development headers`_  3.6+
BLAS implementation                        `OpenBLAS`_ or `Intel MKL`_
CUDA                             9.0+      Optional, for GPU-enabled simulator
`Nvidia CUB`_                    1.8.0     Optional, for GPU-enabled simulator
`googletest`_                    1.10.0    Optional, for building the C++ tests
`PyTorch`_                       1.5+      The libtorch library and headers are needed [#f2]_
===============================  ========  ======

Please refer to your operative system documentation for instructions on how
to install the different dependencies. On a Debian-based operative system,
the following commands can be used for installing the minimal
dependencies::

    $ sudo apt-get install python3-pybind11 python3-dev libopenblas-dev
    $ pip install -r requirements.txt

On an OSX-based system, the following commands can be used for installing the
minimal dependencies (note that ``Xcode`` needs to be installed)::

    $ brew install pybind11
    $ brew install openblas
    $ pip install -r requirements.txt

Installing and compiling
~~~~~~~~~~~~~~~~~~~~~~~~

Once the dependencies are in place, the following command can be used for
installing
For compiling and installing the Python package, the following command can be
used::

    $ pip install -v aihwkit

This command will:

* download the source tarball for the library.
* invoke ``scikit-build``
* which in turn will invoke ``cmake`` for the compilation.
* execute the commands in verbose mode, for helping troubleshooting issues.

If there are any issue with the dependencies or the compilation, the output
of the command will help diagnosing the issue.

.. note::

    Please note that the instruction on this page refer to installing as an
    end user. If you are planning to contribute to the project, an alternative
    setup and tips can be found at the :doc:`developer_install` section that
    is more tuned towards the needs of a development cycle.

.. [#f1] The current (2.5.0) version of ``pybind1`` does not include the
   necessary ``cmake`` helpers on its ``pip`` release. It is recommended to either
   install ``pybind11`` using your operative system package manager or compile and
   install it manually.

.. [#f2] This library uses PyTorch as both a build dependency and a runtime
   dependency. Please ensure that your torch installation includes ``libtorch``
   and the development headers - they are included by default if installing
   torch from ``pip``.

.. _virtual environment: https://docs.python.org/3/library/venv.html

.. _cmake: https://cmake.org/
.. _Nvidia CUB: https://github.com/NVlabs/cub
.. _pybind11: https://github.com/pybind/pybind11
.. _Python 3 development headers: https://www.python.org/downloads/
.. _OpenBLAS: https://www.openblas.net
.. _Intel MKL: https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html
.. _scikit-build: https://github.com/scikit-build/scikit-build
.. _googletest: https://github.com/google/googletest
.. _PyTorch: https://pytorch.org

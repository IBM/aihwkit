Development setup
=================

The goal of this section is setting up a development environment to compile the ``aihwkit`` toolkit on a Linux environment.

The build for ``aihwkit`` is based on `cmake`_, making use of
scikit-build_ for generating the Python packages. The C++/CUDA simulator is
compiled into the ``aihwkit.simulator.rpu_base`` Python extension module: the
toolkit cannot be imported until this module has been compiled.

Some of the dependencies and tools are system-based and some are Python-based.

For convenience, we suggest creating a `conda environment <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#creating-environments>`_ as a way to isolate your development environment.  For example::

    $ conda create -n aihwkit_env python=3.12
    $ conda activate aihwkit_env
    (aihwkit_env) $

.. note::
   Please refer to https://docs.conda.io/projects/miniconda/en/latest/ for how to install `Miniconda`_ in your environment.


Download the aihwkit source
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first step is downloading the source of the library::

    (aihwkit_env) $ git clone https://github.com/IBM/aihwkit.git
    (aihwkit_env) $ cd aihwkit

.. note::

    The following sections assume that the command line examples are executed
    in the activated ``aihwkit_env`` environment, and from the folder where the
    source has been cloned.

.. _Install-the-required-packages-label:

Install the required packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For compiling ``aihwkit``, the following packages are required:

================================  ===========  ======
Dependency                        Version      Notes
================================  ===========  ======
C++17/C++20 compiler              gcc >= 10    C++17 for ``torch <= 2.13``, C++20 for ``torch >= 2.14`` (see :ref:`compilation-flags-label`)
`cmake`_                          >= 4.2.1     Installed as a Python package by ``pip install -e .``
`PyTorch`_                                     Installed by ``pip install -e .``, see :ref:`install-pytorch-label` for GPU builds
`libopenblas-dev`_                             Required with the default ``RPU_BLAS=OpenBLAS`` compiler flag.
`intel-mkl`_                                   Optional, for use with ``RPU_BLAS=MKL`` compiler flag.
                                               Alternately, you can use the ``mkl`` conda packages for this as well.
`CUDA`_                           12.0+        Optional, for the GPU-enabled simulator. 12.8+ is needed for the default
                                               ``RPU_CUDA_ARCHITECTURES`` (it includes ``120``). CUDA 13.x is supported.
`OpenMP`_                                      Optional, used automatically when found
================================  ===========  ======

Other requirements are listed in the ``pyproject.toml`` file in the ``build-system.requires``,
``project.dependencies``, ``project.optional-dependencies.dev`` and
``project.optional-dependencies.examples`` sections you can find in the ``aihwkit`` source.

Please refer to your operating system documentation for instructions on how to install different dependencies.
The following sections contain quick instructions for how to set up the conda environment in Linux
for compiling ``aihwkit``.

.. _install-pytorch-label:

Install pytorch
"""""""""""""""

If your system contains a GPU, then you want to install CUDA-enabled pytorch.
You also need to consider the installed version of the CUDA driver and of the
CUDA toolkit used for compiling ``aihwkit`` (``nvcc --version``), which should
match the CUDA version that pytorch was built with.
Please refer to `pytorch.org <https://pytorch.org/get-started/locally/>`_ for the command to install pytorch.
Pytorch is no longer published on the ``pytorch`` conda channel, so it should be
installed via ``pip`` (or from ``conda-forge``). For example:

    - GPU (replace ``cu132`` with the CUDA version of your toolkit, e.g. ``cu130``)::

      $ pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

    - CPU::

      $ pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

The C++ standard used for compiling ``aihwkit`` depends on the installed
pytorch version (see :ref:`compilation-flags-label`).

Install additional required packages
""""""""""""""""""""""""""""""""""""

Install ``mkl`` and ``mkl-include`` conda packages if you want to use ``-DRPU_BLAS=MKL`` compilation flag::

      $ conda install mkl-include mkl

Install the rest of the required packages::

      $ pip install -e .
      $ pip install -e ".[dev]"
      $ pip install -e ".[examples]"

The quotes are needed in shells such as ``zsh``, where ``[...]`` is otherwise
interpreted as a glob pattern.


Compile the library for development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After installing the requirements listed in the :ref:`Install-the-required-packages-label` section above, you can compile the ``aihwkit`` shared library.  There are several ways to compile the library.

Via python command
""""""""""""""""""
    - CPU with MKL::

      $ python setup.py build_ext -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE --inplace -DRPU_BLAS=MKL -j16 -DCMAKE_PREFIX_PATH=$CONDA_PREFIX

    - GPU with MKL::

      $ python setup.py build_ext -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE --inplace -DRPU_BLAS=MKL -j16 -DUSE_CUDA=ON -DRPU_CUDA_ARCHITECTURES="75;80" -DCMAKE_PREFIX_PATH=$CONDA_PREFIX


If you want to use ``OpenBLAS`` instead ``MKL``, you need to set ``-DRPU_BLAS=OpenBLAS``.

You may need to set ``-DRPU_CUDA_ARCHITECTURES`` to include the architecture of the GPU in your environment.
Compiling only for your own GPU also reduces the compilation time considerably.
To identify the ``CUDA_ARCH`` for your GPU using ``nvidia-smi`` in your system::

    $ export CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv | sed -n '2 p' | tr -d '.')
    $ echo $CUDA_ARCH

and then pass ``-DRPU_CUDA_ARCHITECTURES=$CUDA_ARCH``. Note that CUDA 13
dropped support for architectures older than ``75`` (Turing), and that
architecture ``120`` (Blackwell) requires CUDA 12.8 or newer.

This will produce a shared library under the ``src/aihwkit/simulator``
directory, without installing the package.  For how to use the shared library see the :ref:`use-the-library` section below.

Via make command
""""""""""""""""

As an alternative, you can use ``make`` to compile the ``aihwkit`` shared library, for example:

    - CPU with OpenBLAS::

      $ make build_inplace

    - CPU with MKL::

      $ make build_inplace_mkl

    - GPU with MKL::

      $ make build_inplace_cuda

Additional ``cmake`` flags can be appended via the ``flags`` variable, for example::

    $ make build_inplace_cuda flags="-DRPU_CUDA_ARCHITECTURES=$CUDA_ARCH"

.. note::
   If you are using ``mkl`` in the conda environment then you also need to
   ``export CMAKE_PREFIX_PATH=$CONDA_PREFIX``. The ``*_mkl`` targets also pass
   ``$MKLROOT`` to ``cmake``, which can be set to the root of a standalone
   Intel MKL / oneAPI installation instead.

The ``make`` targets ``pytest``, ``pycodestyle``, ``mypy`` and ``pylint`` run the
test suite and the lint checks used in the continuous integration.

Via cmake command
"""""""""""""""""

For finer control over the compilation and for easier debugging potential issues, you can use ``cmake`` directly
(with the environment containing ``torch`` and ``pybind11`` activated)::

    $ mkdir build
    $ cd build
    build$ cmake ..
    build$ make

Note that the build system uses a temporary ``_skbuild`` folder for caching
some steps of the compilation. While this is useful when making changes to
the source code, in some cases environment changes (such as installing a new
version of the dependencies, switching the compiler or the Python environment, or
changing ``cmake`` flags) are not picked up correctly and the output of the
compilation can be different than expected if the folder is present.

If the compilation was not successful, it is recommended to manually remove the
folder and re-run the compilation in a clean state via::

    $ make clean

Via CUDA-enabled docker image
"""""""""""""""""""""""""""""

As an alternative to a regular install, a CUDA-enabled docker image can also be
built using the ``CUDA.Dockerfile`` included in the repository. The image
compiles ``aihwkit`` with ``MKL`` (from the Intel oneAPI image) on top of the
``nvidia/cuda`` Ubuntu 24.04 images.

In order to build the image, first identify the ``CUDA_ARCH`` for your GPU
using ``nvidia-smi`` in your local machine::

    export CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv | sed -n '2 p' | tr -d '.')
    echo $CUDA_ARCH

The image can be built via::

    docker build \
    --tag aihwkit:cuda \
    --build-arg USERNAME=${USER} \
    --build-arg USERID=$(id -u $USER) \
    --build-arg GROUPID=$(id -g $USER) \
    --build-arg CUDA_ARCH=${CUDA_ARCH} \
    --file CUDA.Dockerfile .

If building your image against a different GPU architecture, please make sure to
update the ``CUDA_ARCH`` build argument accordingly (it defaults to ``75``). The
CUDA version of the base images can be changed with the ``CUDA_VERSION`` build
argument (it defaults to ``13.3.0``), and should be supported by the CUDA driver
of the host.

.. _use-the-library:

Use the compiled library
^^^^^^^^^^^^^^^^^^^^^^^^

Once the library is compiled, the shared library will be created under the
``src/aihwkit/simulator`` directory when you are using ``inplace`` option. By default, this folder is not in the path
that Python uses for finding modules: it needs to be added to the
``PYTHONPATH`` accordingly by either:

1. Updating the environment variable for the session.  For example::

    $ export PYTHONPATH=src/

2. Prepending ``PYTHONPATH=src/`` to the commands where the library needs to
   be found. For example::

    $ PYTHONPATH=src/ python examples/01_simple_layer.py

.. note::

    Please be aware that, if the ``PYTHONPATH`` is not modified and there is a
    version of ``aihkwit`` installed via ``pip``, by default Python will use
    the installed version, as opposed to the custom-compiled version. It is
    recommended to remove the pip-installed version via::

        $ pip uninstall aihwkit

    when developing the library, in order to minimize the risk of confusion.

.. _compilation-flags-label:

Compilation flags
^^^^^^^^^^^^^^^^^

There are several ``cmake`` options that can be used for customizing the
compilation process:

==========================  ========================================================  ==================
Flag                        Description                                               Default
==========================  ========================================================  ==================
``USE_CUDA``                Build with CUDA support                                   ``$USE_CUDA`` env
                                                                                      (``OFF`` if unset)
``RPU_BLAS``                BLAS backend of choice (``OpenBLAS`` or ``MKL``)          ``OpenBLAS``
``RPU_CUDA_ARCHITECTURES``  Target CUDA architectures                                 ``75;80;89;120``
``RPU_CXX_STANDARD``        C++ standard (``17`` or ``20``)                           auto-detected
``BUILD_TEST``              Build the C++ test binaries                               ``OFF``
``BUILD_EXTENSION``         Build the additional C++ torch operators
                            (``aihwkit.extension``)                                   ``OFF``
``RPU_USE_TORCH_BUFFERS``   Use torch (cached) memory buffers in the CUDA simulator   ``ON``
``RPU_USE_FASTMOD``         Use fast mod                                              ``OFF``
``RPU_USE_FASTRAND``        Use fastrand                                              ``OFF``
``RPU_DEBUG``               Enable debug printing                                     ``OFF``
``RPU_USE_DOUBLE``          Experimental: add ``double`` support                      ``OFF``
``RPU_USE_FP16``            Experimental: add half precision support (CUDA only)      ``OFF``
==========================  ========================================================  ==================

The C++ standard is taken from the installed ``PyTorch`` headers when
``RPU_CXX_STANDARD`` is not set: ``torch`` releases up to ``2.13`` are built
with C++17, ``torch >= 2.14`` requires C++20 (and hence a C++20-capable
compiler, e.g. ``gcc >= 10`` or ``CUDA >= 12`` for the GPU build).

The options can be passed both to ``setuptools`` or to ``cmake`` directly. For
example, for compiling with CUDA support::

    $ python setup.py build_ext --inplace -DUSE_CUDA=ON -DRPU_CUDA_ARCHITECTURES="75;80"

or if using ``cmake`` directly::

    build$ cmake -DUSE_CUDA=ON -DRPU_CUDA_ARCHITECTURES="75;80" ..

Passing other ``cmake`` flags
"""""""""""""""""""""""""""""

In the same way flags specific to this project can be passed to ``setup.py``,
other generic ``cmake`` flags can be passed as well. For example, for setting
the compiler to ``clang`` in osx systems::

    $ python setup.py build_ext --inplace -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++

Environment variables
"""""""""""""""""""""

The following environment variables are taken into account during the build
process:

=====================================  ===========================================================
Environment variable                   Description
=====================================  ===========================================================
``USE_CUDA``                           Default value of the ``USE_CUDA`` ``cmake`` option
``VIRTUAL_ENV``                        If present, its ``include``/``lib`` folders are added
                                       to the search paths and it is used as ``CMAKE_PREFIX_PATH``
``CONDA_PREFIX``, ``OpenBLAS_HOME``    Additional locations for finding ``OpenBLAS``
``MKLROOT``                            Location of ``MKL`` used by the ``make *_mkl`` targets
``CUDACXX``                            Path to the ``nvcc`` compiler, if not in the ``PATH``
=====================================  ===========================================================

.. _cmake: https://cmake.org/
.. _Nvidia CUB: https://github.com/NVlabs/cub
.. _pybind11: https://github.com/pybind/pybind11
.. _Python 3 development headers: https://www.python.org/downloads/
.. _libopenblas-dev: https://www.openblas.net
.. _intel-mkl: https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html
.. _scikit-build: https://github.com/scikit-build/scikit-build
.. _googletest: https://github.com/google/googletest
.. _PyTorch: https://pytorch.org
.. _OpenMP: https://openmp.llvm.org
.. _OpenBLAS - Visual Studio: https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio
.. _MS Visual Studio 2019: https://visualstudio.microsoft.com/vs/
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _CUDA: https://developer.nvidia.com/cuda-toolkit

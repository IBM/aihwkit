Development setup
=================

The goal of this section is setting up a development environment to compile the ``aihwkit`` toolkit on a Linux environment.

The build for ``aihwkit`` is based on `cmake`_, making use of
scikit-build_ for generating the Python packages.

Some of the dependencies and tools are system-based and some are Python-based.

For convenience, we suggest creating a `conda environment <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#creating-environments>`_ as a way to isolate your development environment.  For example::

    $ conda create -n aihwkit_env
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

============================  ========  ======
Dependency                    Version   Notes
============================  ========  ======
C++11 compatible compiler               
`libopenblas-dev`_                      Optional, for use with RPU_BLAS=OpenBLAS compiler flag.
`intel-mkl`_                            Optional, for use with RPU_BLAS=MKL compiler flag.
                                        Alternately, you can use mkl conda packages for this as well.
`CUDA`_                       11.3+     Optional, for GPU-enabled simulator
============================  ========  ======

Other requirements are listed in the ``requirements.txt``, ``requirements-dev.txt``, ``requirements-examples.txt`` in the ``aihwkit`` source.

Please refer to your operating system documentation for instructions on how to install different dependencies.
The following sections contain quick instructions for how to set up the conda environment in Linux
for compiling ``aihwkit``.

Install pytorch
"""""""""""""""

If your system contains GPU, then you want to install CUDA-enabled pytorch.
The minimum required version of Torch/Pytorch is specified in the ``requirements.txt`` file. You also need to consider the installed version of the CUDA driver in the installation of pytorch.
Please refer to `pytorch.org <https://pytorch.org/>`_ for the command to install pytorch. For example:

    - GPU::      

      $ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

    - CPU::

      $ conda install pytorch torchvision torchaudio cpuonly -c pytorch


The installation of pytorch conda package would also install additional required packages such as mkl-service, libgcc-ng, blas, etc.

Install additional required packages
""""""""""""""""""""""""""""""""""""

Install ``mkl-include`` conda package if you want to use ``-DRPU_BLAS=MKL`` compilation flag::

      $ conda install mkl-include

Install the rest of the required packages::

      $ pip install -r requirements.txt
      $ pip install -r requirements-dev.txt
      $ pip install -r requirements-example.txt
     

Compile the library for development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After installing the requirements listed in the :ref:`Install-the-required-packages-label` section above, you can compile the ``aihwkit`` shared library.  There are several ways to compile the library.

Via python command
""""""""""""""""""
    - CPU with MKL::   

      $ python setup.py build_ext -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE --inplace -DRPU_BLAS=MKL -j16 -DCMAKE_PREFIX_PATH=$CONDA_PREFIX

    - GPU with MKL::

      $ python setup.py build_ext -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE --inplace -DRPU_BLAS=MKL -j16 -DUSE_CUDA=ON -DRPU_CUDA_ARCHITECTURES="60;70" -DCMAKE_PREFIX_PATH=$CONDA_PREFIX


If you want to use ``OpenBLAS`` instead ``MKL``, you need to set ``-DRPU_BLAS=OpenBLAS``.

You may need to set ``-DRPU_CUDA_ARCHITECTURES`` to include the architecture of the GPU in your environment.
To identify the ``CUDA_ARCH`` for your GPU using ``nvidia-smi`` in your system::

    $ export CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv | sed -n '2 p' | tr -d '.')
    $ echo $CUDA_ARCH

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


Via cmake command
"""""""""""""""""

For finer control over the compilation and for easier debugging potential issues, you can use ``cmake`` directly::

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

Via CUDA-enabled docker image
"""""""""""""""""""""""""""""

As an alternative to a regular install, a CUDA-enabled docker image can also be
built using the ``CUDA.Dockerfile`` included in the repository.

In order to build the image, first identify the ``CUDA_ARCH`` for your GPU
using ` `nvidia-smi`` in your local machine::

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
update the ``CUDA_ARCH`` build argument accordingly.

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
``RPU_CUDA_ARCHITECTURES``  Target CUDA architectures                         ``60;70;75;80``
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

Environment variables
"""""""""""""""""""""

The following environment variables are taken into account during the build
process:

============================  ================================================
Environment variable          Description
============================  ================================================
``TORCH_VERSION_SPECIFIER``   If present, sets the ``PyTorch`` dependency version in the built Python package
============================  ================================================

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

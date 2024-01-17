Advanced installation guide
===========================

Install the aihwkit conda package
---------------------------------

At this time, the conda package is only available for the Linux environment. You can use the
following steps as an example for how to install the aihwkit conda package.

There is a conda package for aihwkit available in conda-forge. 
It can be  installed in a conda environment running on a Linux or WSL in a Windows system.  

Install any one of the conda packages as shown below.

  - CPU::

    $ conda install -c conda-forge aihwkit

  - GPU::

    $ conda install -c conda-forge aihwkit-gpu

Install the aihwkit using pip
---------------------------------
AIHWKIT can also be installed using pip commands as shown below.

 - CPU::

    $ pip install aihwkit

 - GPU::

    $ wget https://aihwkit-gpu-demo.s3.us-east.cloud-object-storage.appdomain.cloud/aihwkit-0.8.0+cuda117-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 

   then::
    
    $ pip install aihwkit-0.8.0+cuda117-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

.. note::

    Please note that the instructions on this page refer to installing as an
    end user. If you are planning to contribute to the project, an alternative
    setup and tips can be found at the :doc:`developer_install` section that
    is more tuned towards the needs of a development cycle.

.. _cmake: https://cmake.org/
.. _Nvidia CUB: https://github.com/NVlabs/cub
.. _pybind11: https://github.com/pybind/pybind11
.. _Python 3 development headers: https://www.python.org/downloads/
.. _OpenBLAS: https://www.openblas.net
.. _Intel MKL: https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html
.. _scikit-build: https://github.com/scikit-build/scikit-build
.. _googletest: https://github.com/google/googletest
.. _PyTorch: https://pytorch.org
.. _OpenMP: https://openmp.llvm.org
.. _OpenBLAS - Visual Studio: https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio
.. _MS Visual Studio 2019: https://visualstudio.microsoft.com/vs/
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _Cuda: https://developer.nvidia.com/cuda-toolkit

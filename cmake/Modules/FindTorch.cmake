# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Try to find the Torch library and headers.
#  TORCH_INCLUDE_DIRS - the torch include directory
#  TORCH_LIB_DIR      - the torch shared library

# Require Python in order to find the site-packages location.
find_package(Python REQUIRED)

# Attempt to locate torch/extension.h.
find_path(TORCH_EXTENSION_H_INCLUDE_DIR
  NAMES torch/extension.h
  DOC "The directory where Torch includes (extension.h) reside"
  PATHS "${Python_SITELIB}/torch/include"
        "${Python_STDARCH}/torch/include"
        "${Python_SITEARCH}/torch/include"
)

if(TORCH_EXTENSION_H_INCLUDE_DIR STREQUAL "TORCH_EXTENSION_H_INCLUDE_DIR-NOTFOUND")
  # Attempt to locate the path by invoking python.
  execute_process(COMMAND python -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE CUSTOM_PYTHON_TORCH_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
  set(CUSTOM_PYTHON_TORCH_PATH "")
endif()

# extension.h
find_path(TORCH_EXTENSION_H_INCLUDE_DIR REQUIRED
  NAMES torch/extension.h
  DOC "The directory where Torch includes (extension.h) reside"
  PATHS "${Python_SITELIB}/torch/include"
        "${Python_STDARCH}/torch/include"
        "${Python_SITEARCH}/torch/include"
        "${CUSTOM_PYTHON_TORCH_PATH}/include"
)

# torch.h
find_path(TORCH_H_INCLUDE_DIR REQUIRED
  NAMES torch/torch.h
  DOC "The directory where Torch includes (torch.h) reside"
  PATHS ${TORCH_EXTENSION_H_INCLUDE_DIR}/torch/csrc/api/include
)

# libtorch_python.so
find_library(TORCH_PYTHON_SO torch_python REQUIRED
    PATHS "${Python_SITELIB}/torch/lib"
          "${Python_STDARCH}/torch/lib"
          "${Python_SITEARCH}/torch/lib"
          "${CUSTOM_PYTHON_TORCH_PATH}/lib"
)

# Set variables.
set(TORCH_INCLUDE_DIRS ${TORCH_EXTENSION_H_INCLUDE_DIR} ${TORCH_H_INCLUDE_DIR})
get_filename_component(TORCH_LIB_DIR ${TORCH_PYTHON_SO} DIRECTORY)

find_package_handle_standard_args(Torch DEFAULT_MSG
  TORCH_INCLUDE_DIRS
  TORCH_LIB_DIR
)

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# Try to find the Torch library and headers.
#  TORCH_INCLUDE_DIRS - the torch include directory
#  TORCH_LIB_DIR      - the torch shared library

# Require Python in order to find the site-packages location.
find_package(Python REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import pathlib, torch; print(pathlib.Path(torch.__file__).parent)"
  RESULT_VARIABLE TORCH_PYTHON_RESULT
  OUTPUT_VARIABLE CUSTOM_PYTHON_TORCH_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT TORCH_PYTHON_RESULT EQUAL 0 OR
   NOT EXISTS "${CUSTOM_PYTHON_TORCH_PATH}/include/torch/extension.h")
  message(FATAL_ERROR
    "Could not locate PyTorch headers for ${Python_EXECUTABLE}: "
    "${CUSTOM_PYTHON_TORCH_PATH}")
endif()

find_path(TORCH_EXTENSION_H_INCLUDE_DIR
  NAMES torch/extension.h
  PATHS "${CUSTOM_PYTHON_TORCH_PATH}/include"
  NO_DEFAULT_PATH
  REQUIRED
)

find_path(TORCH_H_INCLUDE_DIR
  NAMES torch/torch.h
  PATHS "${TORCH_EXTENSION_H_INCLUDE_DIR}/torch/csrc/api/include"
  NO_DEFAULT_PATH
  REQUIRED
)

find_library(TORCH_PYTHON_SO
  NAMES torch_python
  PATHS "${CUSTOM_PYTHON_TORCH_PATH}/lib"
  NO_DEFAULT_PATH
  REQUIRED
)

# Set variables.
set(TORCH_INCLUDE_DIRS ${TORCH_EXTENSION_H_INCLUDE_DIR} ${TORCH_H_INCLUDE_DIR})
get_filename_component(TORCH_LIB_DIR ${TORCH_PYTHON_SO} DIRECTORY)

find_package_handle_standard_args(Torch DEFAULT_MSG
  TORCH_INCLUDE_DIRS
  TORCH_LIB_DIR
)

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# Try to find the CUB library and headers.
#  CUB_FOUND        - system has CUB
#  CUB_INCLUDE_DIRS - the CUB include directory

find_path(CUB_INCLUDE_DIR
  NAMES cub/cub.cuh
  DOC "The directory where CUB includes reside"
)

set(CUB_INCLUDE_DIRS ${CUB_INCLUDE_DIR})

find_package_handle_standard_args(CUB
  FOUND_VAR CUB_FOUND
  REQUIRED_VARS CUB_INCLUDE_DIR
)

mark_as_advanced(CUB_FOUND)

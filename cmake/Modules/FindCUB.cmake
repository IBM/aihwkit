# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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

# This file has been adapted from pytorch.
# https://github.com/pytorch/pytorch

# Copyright (c) 2016-, Facebook Inc. All rights reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

SET(Open_BLAS_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/include/openblas
  /usr/include/openblas-base
  /usr/local/include
  /usr/local/include/openblas
  /usr/local/include/openblas-base
  /usr/local/opt/openblas/include
  /opt/OpenBLAS/include
  /opt/include/OpenBLAS
  $ENV{OpenBLAS_HOME}
  $ENV{OpenBLAS_HOME}/include
  $ENV{OPENBLAS_ROOT_DIR}
  $ENV{OPENBLAS_ROOT_DIR}/include
  $ENV{CONDA_PREFIX}/include
)

SET(Open_BLAS_LIB_SEARCH_PATHS
  /lib/
  /lib/openblas-base
  /lib64/
  /usr/lib
  /usr/lib/openblas-base
  /usr/lib64
  /usr/local/lib
  /usr/local/lib64
  /usr/local/opt/openblas/lib
  /opt/OpenBLAS/lib
  /opt/lib
  /opt/lib/OpenBLAS
  $ENV{OpenBLAS}
  $ENV{OpenBLAS}/lib
  $ENV{OpenBLAS_HOME}
  $ENV{OpenBLAS_HOME}/lib
  $ENV{OPENBLAS_ROOT_DIR}
  $ENV{OPENBLAS_ROOT_DIR}/lib
  $ENV{CONDA_PREFIX}/lib
)

FIND_PATH(OpenBLAS_INCLUDE_DIR NAMES openblas_config.h PATHS ${Open_BLAS_INCLUDE_SEARCH_PATHS})
if(WIN32)
    # The official OpenBLAS github releases (-x64) name the library
    # `libopenblas.*`, whereas the default `CMAKE_FIND_LIBRARY_PREFIXES` does
    # not contain `lib`.
    FIND_LIBRARY(OpenBLAS_LIB NAMES openblas libopenblas PATHS ${Open_BLAS_LIB_SEARCH_PATHS})
else()
    FIND_LIBRARY(OpenBLAS_LIB NAMES openblas PATHS ${Open_BLAS_LIB_SEARCH_PATHS})
endif()
# maybe add the openblasp64 if available

SET(OpenBLAS_FOUND ON)

#    Check include files
IF(NOT OpenBLAS_INCLUDE_DIR)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find OpenBLAS include. Turning OpenBLAS_FOUND off")
ENDIF()

#    Check libraries
IF(NOT OpenBLAS_LIB)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find OpenBLAS lib. Turning OpenBLAS_FOUND off")
ENDIF()

IF (OpenBLAS_FOUND)
  IF (NOT OpenBLAS_FIND_QUIETLY)
    MESSAGE(STATUS "Found OpenBLAS libraries: ${OpenBLAS_LIB}")
    MESSAGE(STATUS "Found OpenBLAS include: ${OpenBLAS_INCLUDE_DIR}")
  ENDIF (NOT OpenBLAS_FIND_QUIETLY)
ELSE (OpenBLAS_FOUND)
  IF (OpenBLAS_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find OpenBLAS")
  ENDIF (OpenBLAS_FIND_REQUIRED)
ENDIF (OpenBLAS_FOUND)

MARK_AS_ADVANCED(
    OpenBLAS_INCLUDE_DIR
    OpenBLAS_LIB
    OpenBLAS
)

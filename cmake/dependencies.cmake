# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# For informative purposes.
if(SKBUILD)
  message(STATUS "Invoking cmake through scikit-build")
endif()

# RPATH handling
# see https://cmake.org/Wiki/CMake_RPATH_handling
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(_rpath_portable_origin "@loader_path")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
else()
  set(_rpath_portable_origin $ORIGIN)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
endif(APPLE)
# Use separate rpaths during build and install phases
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
# Don't use the install-rpath during the build phase
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${_rpath_portable_origin}")


# Threads
if(USE_THREADS)
  find_package(Threads REQUIRED)
  list(APPEND RPU_DEPENDENCY_LIBS ${CMAKE_THREAD_LIBS_INIT})
endif()


# OpenMP
find_package(OpenMP QUIET)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  include_directories(SYSTEM ${OpenMP_CXX_INCLUDE_DIR})
else()
  message(STATUS "OpenMP could not be found. Disabling OpenMP support.")
endif()

# BLAS
message(STATUS "The BLAS backend of choice:" ${RPU_BLAS})

if(RPU_BLAS STREQUAL "OpenBLAS")
  find_package(OpenBLAS REQUIRED)
  include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
  list(APPEND RPU_DEPENDENCY_LIBS ${OpenBLAS_LIB})
  add_compile_definitions(RPU_USE_OPENBLAS)
elseif(RPU_BLAS STREQUAL "MKL")
  find_package(MKL REQUIRED)
  include_directories(SYSTEM ${MKL_INCLUDE_DIR})
  list(APPEND RPU_DEPENDENCY_LIBS ${MKL_LIBRARIES} )
  if(USE_OMP)
    list(APPEND RPU_DEPENDENCY_LIBS ${MKL_OPENMP_LIBRARY} )
  endif()
  if(NOT MSVC) # not sure why this is not found for linux. Maybe also windows?
    find_package(AVX) # checks AVX and AVX2
    set(MKL_LIB_PATH "${MKL_LIBRARIES}")
    list(FILTER MKL_LIB_PATH INCLUDE REGEX ".*mkl_core.*")
    list(GET MKL_LIB_PATH 0 MKL_LIB_PATH)
    get_filename_component(MKL_LIB_PATH ${MKL_LIB_PATH} DIRECTORY)
    if(CXX_AVX2_FOUND)
      message(STATUS "AVX compiler support found")
      file(GLOB tmp "${MKL_LIB_PATH}/*mkl_avx2*")
      list(APPEND RPU_DEPENDENCY_LIBS "${tmp}")
    endif()
  endif()
  add_compile_definitions(RPU_USE_MKL)
  message(STATUS "MKL include for RPU is ${RPU_DEPENDENCY_LIBS}")
else()
  message(FATAL_ERROR "Invalid BLAS backend: ${RPU_BLAS}")
endif()

# Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})  # order matters (before pybind)

set(ignoreMe "${Python3_EXECUTABLE}${Python3_FIND_REGISTRY}${Python3_INCLUDE_DIR}${Python3_NumPy_INCLUDE_DIRS}${Python3_ROOT_DIR}${Python_INCLUDE_DIR}${Python_NumPy_INCLUDE_DIRS}")

# Find pybind11Config.cmake
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE CUSTOM_PYTHON_PYBIND11_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
set(pybind11_DIR ${CUSTOM_PYTHON_PYBIND11_PATH})

find_package(pybind11 CONFIG REQUIRED)
include_directories(${pybind11_INCLUDE_DIR})

# Pytorch
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
link_directories(${TORCH_LIB_DIR})

if (CMAKE_COMPILER_IS_GNUCXX)
  # check for pytorch's ABI
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print('1' if torch._C._GLIBCXX_USE_CXX11_ABI else '0')"
    OUTPUT_VARIABLE OUTPUT_GNU_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  add_compile_definitions("_GLIBCXX_USE_CXX11_ABI=${OUTPUT_GNU_ABI}")
  message(STATUS "Set _GLIBCXX_USE_CXX11_ABI=${OUTPUT_GNU_ABI}")
endif()

# Set compile definitions
if(RPU_USE_FASTRAND)
  add_compile_definitions(RPU_USE_FASTRAND)
endif()

if(RPU_USE_FASTMOD)
  add_compile_definitions(RPU_USE_FASTMOD)
endif()

if(RPU_DEBUG)
  add_compile_definitions(RPU_DEBUG)
endif()

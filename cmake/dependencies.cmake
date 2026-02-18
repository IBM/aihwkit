# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

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
find_package(Python3 COMPONENTS Interpreter Development.Module)
include_directories(${Python3_INCLUDE_DIRS})  # order matters (before pybind)

set(ignoreMe "${Python3_EXECUTABLE}${Python3_FIND_REGISTRY}${Python3_INCLUDE_DIR}${Python3_NumPy_INCLUDE_DIRS}${Python3_ROOT_DIR}${Python_INCLUDE_DIR}${Python_NumPy_INCLUDE_DIRS}")
set(RPU_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")

if(NOT RPU_PYTHON_EXECUTABLE AND DEFINED PYTHON_EXECUTABLE AND NOT "${PYTHON_EXECUTABLE}" STREQUAL "")
  # Keep compatibility with legacy FindPython variables if provided by callers.
  set(RPU_PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()

if(NOT RPU_PYTHON_EXECUTABLE)
  message(FATAL_ERROR "Could not determine a Python executable for CMake helper commands.")
endif()

# Find pybind11Config.cmake
execute_process(COMMAND "${RPU_PYTHON_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    RESULT_VARIABLE PYBIND11_CMAKE_DIR_RESULT
    OUTPUT_VARIABLE CUSTOM_PYTHON_PYBIND11_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE PYBIND11_CMAKE_DIR_ERROR)
if(NOT PYBIND11_CMAKE_DIR_RESULT EQUAL 0 OR CUSTOM_PYTHON_PYBIND11_PATH STREQUAL "")
  message(FATAL_ERROR "Failed to query pybind11 CMake dir with ${RPU_PYTHON_EXECUTABLE}: ${PYBIND11_CMAKE_DIR_ERROR}")
endif()
set(pybind11_DIR ${CUSTOM_PYTHON_PYBIND11_PATH})

find_package(pybind11 CONFIG REQUIRED)
include_directories(${pybind11_INCLUDE_DIR})

# Pytorch
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
link_directories(${TORCH_LIB_DIR})

if (CMAKE_COMPILER_IS_GNUCXX)
  # Prefer ABI from Torch CMake config (works even when Python build isolation
  # makes `import torch` unavailable in helper subprocesses).
  set(OUTPUT_GNU_ABI "")
  if(DEFINED TORCH_CXX_FLAGS AND NOT "${TORCH_CXX_FLAGS}" STREQUAL "")
    string(REGEX MATCH "-D_GLIBCXX_USE_CXX11_ABI=[01]" TORCH_ABI_DEFINE "${TORCH_CXX_FLAGS}")
    if(NOT TORCH_ABI_DEFINE STREQUAL "")
      string(REGEX REPLACE ".*=([01]).*" "\\1" OUTPUT_GNU_ABI "${TORCH_ABI_DEFINE}")
    endif()
  endif()

  if(NOT OUTPUT_GNU_ABI MATCHES "^[01]$")
    # Fallback: query torch Python runtime if ABI is not present in TORCH_CXX_FLAGS.
    execute_process(COMMAND "${RPU_PYTHON_EXECUTABLE}" -c "import torch; print('1' if torch._C._GLIBCXX_USE_CXX11_ABI else '0')"
      RESULT_VARIABLE TORCH_GNU_ABI_RESULT
      OUTPUT_VARIABLE OUTPUT_GNU_ABI
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_VARIABLE TORCH_GNU_ABI_ERROR)
    if(NOT TORCH_GNU_ABI_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to determine torch CXX11 ABI. TORCH_CXX_FLAGS='${TORCH_CXX_FLAGS}'. Python probe error with ${RPU_PYTHON_EXECUTABLE}: ${TORCH_GNU_ABI_ERROR}")
    endif()
  endif()

  if(NOT OUTPUT_GNU_ABI MATCHES "^[01]$")
    message(FATAL_ERROR "Invalid torch CXX11 ABI value: '${OUTPUT_GNU_ABI}'. Expected 0 or 1.")
  endif()
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${OUTPUT_GNU_ABI})
  message(STATUS "Set _GLIBCXX_USE_CXX11_ABI=${OUTPUT_GNU_ABI} (TORCH_CXX_FLAGS='${TORCH_CXX_FLAGS}')")
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

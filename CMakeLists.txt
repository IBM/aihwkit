# (C) Copyright 2020 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

cmake_minimum_required(VERSION 3.18.0)
project(aihwkit C CXX)

# Project options.
option(BUILD_TEST "Build C++ test binaries" OFF)
option(USE_CUDA "Build with CUDA support" OFF)
option(RPU_DEBUG "Enable debug printing" OFF)
option(RPU_USE_FASTMOD "Use fast mod" ON)
option(RPU_USE_FASTRAND "Use fastrand" OFF)
set(RPU_BLAS "OpenBLAS" CACHE STRING "BLAS backend of choice (OpenBLAS, MKL)")
set(RPU_CUDA_ARCHITECTURES "60" CACHE STRING "Target CUDA architectures")

# Internal variables.
set(CUDA_TARGET_PROPERTIES POSITION_INDEPENDENT_CODE ON
                           CUDA_RESOLVE_DEVICE_SYMBOLS ON
                           CUDA_SEPARABLE_COMPILATION ON
                           CXX_STANDARD 11
                           POSITION_INDEPENDENT_CODE ON)

# Append the virtualenv library path to cmake.
if(DEFINED ENV{VIRTUAL_ENV})
  include_directories("$ENV{VIRTUAL_ENV}/include")
  link_directories("$ENV{VIRTUAL_ENV}/lib")
  set(CMAKE_PREFIX_PATH "$ENV{VIRTUAL_ENV}")
endif()

# Check for dependencies.
include(cmake/dependencies.cmake)
include(cmake/dependencies_cuda.cmake)
include(cmake/dependencies_test.cmake)

# Set compilation flags.
if(WIN32)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-narrowing -Wno-strict-overflow")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ftree-vectorize")
endif()

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
if (APPLE)
  string(APPEND CMAKE_CXX_FLAGS " -fvisibility=hidden")
endif()

# Add rpucuda sources and target.
add_subdirectory(src/rpucuda)
include_directories(SYSTEM src/rpucuda)

add_library(RPU_CPU ${RPU_CPU_SRCS})

target_link_libraries(RPU_CPU ${RPU_DEPENDENCY_LIBS})
if(WIN32)
  target_link_libraries(RPU_CPU c10.lib torch_cpu.lib)
endif()

set_target_properties(RPU_CPU PROPERTIES CXX_STANDARD 11
                                         POSITION_INDEPENDENT_CODE ON)

if(USE_CUDA)
  add_subdirectory(src/rpucuda/cuda)
  include_directories(SYSTEM src/rpucuda/cuda)
  add_library(RPU_GPU ${RPU_GPU_SRCS})

  target_link_libraries(RPU_GPU RPU_CPU cublas curand ${RPU_DEPENDENCY_LIBS})
  if(WIN32)
    target_link_libraries(RPU_GPU c10_cuda.lib torch_cuda.lib)
  endif(WIN32)

  set_target_properties(RPU_GPU PROPERTIES ${CUDA_TARGET_PROPERTIES})
  set_property(TARGET RPU_GPU PROPERTY CUDA_ARCHITECTURES ${RPU_CUDA_ARCHITECTURES})

  if(${CUDAToolkit_VERSION_MAJOR} LESS 11)
    # The "cub" target only exists if cub was downloaded during build.
    if(TARGET cub)
        add_dependencies(RPU_GPU cub)
    endif()
  endif()

endif(USE_CUDA)

# Add aihwkit targets.
add_subdirectory(src/aihwkit/simulator)

# Add tests.
if(BUILD_TEST)
  enable_testing()

  foreach(test_src ${RPU_CPU_TEST_SRCS} ${RPU_GPU_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} gtest gmock)

    # Link to main library.
    if(${test_src} IN_LIST RPU_CPU_TEST_SRCS)
      target_link_libraries(${test_name} RPU_CPU ${RPU_DEPENDENCY_LIBS})
    else()
      target_link_libraries(${test_name} RPU_GPU RPU_CPU cublas curand ${RPU_DEPENDENCY_LIBS})
      set_target_properties(${test_name} PROPERTIES ${CUDA_TARGET_PROPERTIES})
      set_property(TARGET ${test_name} PROPERTY CUDA_ARCHITECTURES ${RPU_CUDA_ARCHITECTURES})
    endif()

    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    set_target_properties(${test_name} PROPERTIES FOLDER tests)
    add_dependencies(RPU_CPU GTest)
  endforeach()
endif()
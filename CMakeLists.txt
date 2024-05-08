# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
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
option(BUILD_EXTENSION "Build additional C++ tools" OFF)
option(USE_CUDA "Build with CUDA support" $ENV{USE_CUDA})

# experimental precision flags
option(RPU_USE_FP16 "EXPERIMENTAL: Build FP16 support (only available with CUDA)" OFF)
option(RPU_USE_DOUBLE "EXPERIMENTAL: Build DOUBLE support" OFF)
option(RPU_PARAM_FP16 "EXPERIMENTAL: Use FP16 for (4 + 2) CUDA params" OFF)
option(RPU_BFLOAT_AS_FP16 "EXPERIMENTAL: Use bfloat instead of half for FP16 (only supported for A100+, CUDA 12)" OFF)

option(RPU_DEBUG "Enable debug printing" OFF)
option(RPU_USE_FASTMOD "Use fast mod" OFF)
option(RPU_USE_FASTRAND "Use fastrand" OFF)
option(RPU_USE_TORCH_BUFFERS "Use torch buffers for RPUCuda" ON)


set(RPU_BLAS "OpenBLAS" CACHE STRING "BLAS backend of choice (OpenBLAS, MKL)")
set(RPU_CUDA_ARCHITECTURES "70;75;80" CACHE STRING "Target CUDA architectures")

# Internal variables.
set(CUDA_TARGET_PROPERTIES POSITION_INDEPENDENT_CODE ON
                           CUDA_RESOLVE_DEVICE_SYMBOLS ON
                           CUDA_SEPARABLE_COMPILATION ON
                           CXX_STANDARD 17)

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
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-narrowing -Wno-strict-overflow")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ftree-vectorize")
endif()

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

set_target_properties(RPU_CPU PROPERTIES CXX_STANDARD 17
  POSITION_INDEPENDENT_CODE ON)

add_compile_definitions(RPU_USE_WITH_TORCH)
if (RPU_USE_DOUBLE)
  add_compile_definitions(RPU_USE_DOUBLE)
  message(STATUS "Add DOUBLE as RPU number type.")
endif(RPU_USE_DOUBLE)


if(USE_CUDA)
  add_subdirectory(src/rpucuda/cuda)
  include_directories(SYSTEM src/rpucuda/cuda)
  add_library(RPU_GPU ${RPU_GPU_SRCS})

  if (RPU_USE_FP16)
    add_compile_definitions(RPU_USE_FP16)
    if (RPU_BFLOAT_AS_FP16)
      add_compile_definitions(RPU_BFLOAT_AS_FP16)
      message(STATUS "Add BFLOAT16 as RPU number type.")
    else (RPU_BFLOAT_AS_FP16)
      message(STATUS "Add FP16 as RPU number type.")
    endif (RPU_BFLOAT_AS_FP16)
  endif(RPU_USE_FP16)


  if (RPU_PARAM_FP16)
    add_compile_definitions(RPU_PARAM_FP16)
    message(STATUS "Use FP16 parameters for CUDA (for all RPU number types).")
  endif(RPU_PARAM_FP16)

  target_link_libraries(RPU_GPU RPU_CPU cublas curand ${RPU_DEPENDENCY_LIBS})
  if(WIN32)
    target_link_libraries(RPU_GPU c10_cuda.lib torch_cuda.lib)
  endif(WIN32)

  set_target_properties(RPU_GPU PROPERTIES ${CUDA_TARGET_PROPERTIES})
  set_property(TARGET RPU_GPU PROPERTY CUDA_ARCHITECTURES ${RPU_CUDA_ARCHITECTURES})

  if (RPU_USE_TORCH_BUFFERS)
    if (BUILD_TEST)
      # we could just link torch to the tests in principle
      message(FATAL_ERROR "Cannot use torch buffers when BUILD_TEST=ON. Set RPU_USE_TORCH_BUFFERS=OFF")
    endif(BUILD_TEST)

    add_compile_definitions(RPU_TORCH_CUDA_BUFFERS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=186")
  endif(RPU_USE_TORCH_BUFFERS)

  if(${CUDAToolkit_VERSION_MAJOR} LESS 11)
    # The "cub" target only exists if cub was downloaded during build.
    if(TARGET cub)
        add_dependencies(RPU_GPU cub)
    endif()
  endif()
endif(USE_CUDA)

# Add aihwkit targets.
add_subdirectory(src/aihwkit/simulator)

# Add extension targets.
if (BUILD_EXTENSION)
  add_subdirectory(src/aihwkit/extension)

  add_library(AIHWKIT_EXTENSION_OPS ${AIHWKIT_EXTENSION_OPS_CPU_SRCS})

  set_target_properties(AIHWKIT_EXTENSION_OPS PROPERTIES CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON)

  target_link_libraries(AIHWKIT_EXTENSION_OPS torch_python c10 torch_cpu)
  target_include_directories(AIHWKIT_EXTENSION_OPS PRIVATE src/aihwkit/extension/extension_src)
  
  if(WIN32)
    target_link_libraries(AIHWKIT_EXTENSION_OPS c10.lib torch_cpu.lib)
  endif()

  if(USE_CUDA)
    add_library(AIHWKIT_EXTENSION_OPS_GPU ${AIHWKIT_EXTENSION_OPS_GPU_SRCS})
    target_link_libraries(AIHWKIT_EXTENSION_OPS_GPU AIHWKIT_EXTENSION_OPS c10_cuda torch_cuda cudart)
    target_include_directories(AIHWKIT_EXTENSION_OPS_GPU PRIVATE src/aihwkit/extension/extension_src)
    
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=186")
    set_target_properties(AIHWKIT_EXTENSION_OPS_GPU PROPERTIES ${CUDA_TARGET_PROPERTIES})
    set_property(TARGET AIHWKIT_EXTENSION_OPS_GPU PROPERTY CUDA_ARCHITECTURES ${RPU_CUDA_ARCHITECTURES})

    if(WIN32)
      target_link_libraries(AIHWKIT_EXTENSION_OPS_GPU c10_cuda.lib torch_cuda.lib)
    endif(WIN32)

  endif()

  set(extension_module_name aihwkit_extension)
  pybind11_add_module(${extension_module_name} MODULE ${EXTENSION_BINDINGS_SRCS})
  target_link_libraries(${extension_module_name} PRIVATE torch_python)
  target_include_directories(${extension_module_name} PRIVATE src/aihwkit/extension/extension_src)
  target_include_directories(${extension_module_name} PRIVATE src/aihwkit/extension/extension_src/ops)
  set_target_properties(${extension_module_name} PROPERTIES CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON)

  if (USE_CUDA)
    target_link_libraries(${extension_module_name} PRIVATE AIHWKIT_EXTENSION_OPS_GPU)
  else()
    target_link_libraries(${extension_module_name} PRIVATE AIHWKIT_EXTENSION_OPS)
  endif()

  install(TARGETS ${extension_module_name} DESTINATION "src/aihwkit/extension")

  add_custom_command(TARGET ${extension_module_name}
    COMMAND stubgen --module "src/aihwkit/extension" --output .
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
endif(BUILD_EXTENSION)



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

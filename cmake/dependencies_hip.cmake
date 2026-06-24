# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# ROCm/HIP support, mirroring dependencies_cuda.cmake. The CUDA-spelled device
# sources compile under HIP through the cuda_to_hip.h shim; RPU_USE_CUDA stays
# defined so the device tile code and CudaAnalogTile pybind exposure are built.
if(USE_HIP)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH})
    else()
      set(ROCM_PATH "/opt/rocm")
    endif()
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

  enable_language(HIP)

  find_package(hip REQUIRED)
  find_package(hipblas REQUIRED)
  find_package(hiprand REQUIRED)
  find_package(hipcub REQUIRED)

  add_compile_definitions(RPU_USE_CUDA)
  add_compile_definitions(USE_HIP)

  # Detect the torch hipify generation. hipify v1 renames CUDA spellings to HIP
  # (c10::cuda::getCurrentCUDAStream is removed, only c10::hip::getCurrentHIPStream
  # exists); hipify v2 masquerades, keeping c10::cuda::getCurrentCUDAStream as the
  # public API while c10::hip::getCurrentHIPStream stays guarded by USE_ROCM (which
  # this build does not define). rpu_base_tiles_cuda.cpp selects the c10 stream
  # symbol from this, so probe the torch the build already uses.
  execute_process(COMMAND "${RPU_PYTHON_EXECUTABLE}" -c "from torch.utils.hipify import __version__ as v; print(v)"
      RESULT_VARIABLE TORCH_HIPIFY_VERSION_RESULT
      OUTPUT_VARIABLE TORCH_HIPIFY_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
  if(TORCH_HIPIFY_VERSION_RESULT EQUAL 0 AND NOT TORCH_HIPIFY_VERSION STREQUAL "")
    message(STATUS "torch hipify version: ${TORCH_HIPIFY_VERSION}")
    if(TORCH_HIPIFY_VERSION VERSION_GREATER_EQUAL "2.0.0")
      set(RPU_TORCH_HIPIFY_V2 ON)
    endif()
  else()
    message(STATUS "torch hipify version: unknown (assuming v1 stream API)")
  endif()
endif()

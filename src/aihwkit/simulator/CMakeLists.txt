# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

set(python_module_name rpu_base)

file(GLOB RPU_BINDINGS_SRCS rpu_base_src/*.cpp)
if (USE_HIP)
  # The binding TUs pull in HIP device headers transitively from RPU_GPU and
  # inherit its --offload-arch usage requirement, which only the HIP (clang)
  # driver understands; compile them with the HIP toolchain.
  set_source_files_properties(${RPU_BINDINGS_SRCS} PROPERTIES LANGUAGE HIP)
endif()
pybind11_add_module(${python_module_name} MODULE ${RPU_BINDINGS_SRCS})
target_link_libraries(${python_module_name} PRIVATE torch_python)
set_target_properties(${python_module_name} PROPERTIES CXX_STANDARD ${RPU_CXX_STANDARD})

if (USE_CUDA OR USE_HIP)
  target_link_libraries(${python_module_name} PRIVATE RPU_GPU)
else()
  target_link_libraries(${python_module_name} PRIVATE RPU_CPU)
endif()

if (USE_HIP)
  # HIP device-code linking does not finalize LTO, which yields an empty
  # PyInit_ and an ImportError; keep IPO off for the pybind module.
  set_target_properties(${python_module_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

  # rpu_base_tiles_cuda.cpp picks the c10 current-stream symbol by hipify
  # generation (see cmake/dependencies_hip.cmake); v2 keeps the c10::cuda
  # spelling, v1 (undefined) uses c10::hip.
  if(RPU_TORCH_HIPIFY_V2)
    target_compile_definitions(${python_module_name} PRIVATE TORCH_HIPIFY_V2)
  endif()
endif()

install(TARGETS ${python_module_name} DESTINATION "src/aihwkit/simulator")

add_custom_command(TARGET ${python_module_name}
  COMMAND stubgen --module "src/aihwkit/simulator" --output .
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

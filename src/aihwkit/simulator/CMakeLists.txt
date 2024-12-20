# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

set(python_module_name rpu_base)

file(GLOB RPU_BINDINGS_SRCS rpu_base_src/*.cpp)
pybind11_add_module(${python_module_name} MODULE ${RPU_BINDINGS_SRCS})
target_link_libraries(${python_module_name} PRIVATE torch_python)
set_target_properties(${python_module_name} PROPERTIES CXX_STANDARD 17)

if (USE_CUDA)
  target_link_libraries(${python_module_name} PRIVATE RPU_GPU)
else()
  target_link_libraries(${python_module_name} PRIVATE RPU_CPU)
endif()

install(TARGETS ${python_module_name} DESTINATION "src/aihwkit/simulator")

add_custom_command(TARGET ${python_module_name}
  COMMAND stubgen --module "src/aihwkit/simulator" --output .
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# CPU ops files
file(GLOB AIHWKIT_EXTENSION_OPS_CPU_SRCS extension_src/ops/*.cpp)
set(AIHWKIT_EXTENSION_OPS_CPU_SRCS ${AIHWKIT_EXTENSION_OPS_CPU_SRCS} PARENT_SCOPE)

# GPU ops files
file(GLOB AIHWKIT_EXTENSION_OPS_GPU_SRCS extension_src/ops/*.cu)
set(AIHWKIT_EXTENSION_OPS_GPU_SRCS ${AIHWKIT_EXTENSION_OPS_GPU_SRCS} PARENT_SCOPE)

# CPU extension files
file(GLOB EXTENSION_BINDINGS_SRCS extension_src/*.cpp)
set(EXTENSION_BINDINGS_SRCS ${EXTENSION_BINDINGS_SRCS} PARENT_SCOPE)



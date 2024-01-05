# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# CPU ops files
file(GLOB AIHWKIT_EXTENSION_OPS_CPU_SRCS extension_src/ops/*.cpp)
set(AIHWKIT_EXTENSION_OPS_CPU_SRCS ${AIHWKIT_EXTENSION_OPS_CPU_SRCS} PARENT_SCOPE)

# GPU ops files
file(GLOB AIHWKIT_EXTENSION_OPS_GPU_SRCS extension_src/ops/*.cu)
set(AIHWKIT_EXTENSION_OPS_GPU_SRCS ${AIHWKIT_EXTENSION_OPS_GPU_SRCS} PARENT_SCOPE)

# CPU extension files
file(GLOB EXTENSION_BINDINGS_SRCS extension_src/*.cpp)
set(EXTENSION_BINDINGS_SRCS ${EXTENSION_BINDINGS_SRCS} PARENT_SCOPE)



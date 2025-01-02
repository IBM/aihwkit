# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# Simulator main files.
file(GLOB RPU_CPU_SRCS *.cpp)
list(FILTER RPU_CPU_SRCS EXCLUDE REGEX ".*_test.cpp$")
set(RPU_CPU_SRCS ${RPU_CPU_SRCS} PARENT_SCOPE)

# Simulator test files.
file(GLOB RPU_CPU_TEST_SRCS *_test.cpp)
set(RPU_CPU_TEST_SRCS ${RPU_CPU_TEST_SRCS} PARENT_SCOPE)

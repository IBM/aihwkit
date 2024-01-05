# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Simulator main files.
file(GLOB RPU_CPU_SRCS *.cpp)
list(FILTER RPU_CPU_SRCS EXCLUDE REGEX ".*_test.cpp$")
set(RPU_CPU_SRCS ${RPU_CPU_SRCS} PARENT_SCOPE)

# Simulator test files.
file(GLOB RPU_CPU_TEST_SRCS *_test.cpp)
set(RPU_CPU_TEST_SRCS ${RPU_CPU_TEST_SRCS} PARENT_SCOPE)

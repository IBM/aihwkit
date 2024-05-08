# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

if(BUILD_TEST)
  include(ExternalProject)
  ExternalProject_Add(GTest
    URL               https://github.com/google/googletest/archive/release-1.11.0.zip
    URL_HASH          MD5=52943a59cefce0ae0491d4d2412c120b
    CMAKE_ARGS        "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI\=0"
    INSTALL_COMMAND   ""
  )

  ExternalProject_Get_Property(GTest source_dir)
  ExternalProject_Get_Property(GTest binary_dir)
  set(GTest_INCLUDE_DIR ${source_dir}/googletest/include)
  set(GTest_LIBRARY_DIR ${binary_dir}/lib)

  include_directories(SYSTEM ${GTest_INCLUDE_DIR})
  link_directories(SYSTEM ${GTest_LIBRARY_DIR})
endif()

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# CUDA
if(USE_CUDA)
  enable_language(CUDA)

  # CUDA Toolkit
  find_package(CUDAToolkit)

  # CUB (Only for CUDA v. less than 11)
  if(${CUDAToolkit_VERSION_MAJOR} LESS 11)
    find_package(CUB QUIET)
    if(CUB_FOUND)
      include_directories(SYSTEM ${CUB_INCLUDE_DIRS})
    else()
      include(ExternalProject)
      ExternalProject_Add(cub
        URL               https://github.com/NVlabs/cub/archive/1.8.0.zip
        URL_HASH          MD5=a821b9dffbc9d1bacf1c8db2a59094bf
        GIT_TAG        origin/release/1.2.3
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
      )

      ExternalProject_Get_Property(cub source_dir)
      set(CUB_INCLUDE_DIR ${source_dir})
      include_directories(SYSTEM ${CUB_INCLUDE_DIR})
    endif(CUB_FOUND)
  endif()

  include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  add_compile_definitions(RPU_USE_CUDA)

endif()

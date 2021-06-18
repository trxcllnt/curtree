#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

###################################################################################################
# - CMake properties ------------------------------------------------------------------------------

if(${PROJECT_NAME}_USE_CCACHE)
    find_program(CCACHE_PROGRAM_PATH ccache)
    if(CCACHE_PROGRAM_PATH)
        message(STATUS "Using ccache: ${CCACHE_PROGRAM_PATH}")
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM_PATH}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM_PATH}")
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM_PATH}")
    endif()
endif()

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(${PROJECT_NAME}_CMAKE_C_FLAGS "")
set(${PROJECT_NAME}_CMAKE_CXX_FLAGS "")
set(${PROJECT_NAME}_CMAKE_CUDA_FLAGS "")

if(CMAKE_COMPILER_IS_GNUCXX)
    option(${PROJECT_NAME}_CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)
    list(APPEND ${PROJECT_NAME}_CMAKE_CXX_FLAGS -Wall -Werror -Wno-unknown-pragmas -Wno-error=deprecated-declarations)
    if(${PROJECT_NAME}_CMAKE_CXX11_ABI)
        message(STATUS "Enabling the GLIBCXX11 ABI")
    else()
        message(STATUS "Disabling the GLIBCXX11 ABI")
        list(APPEND ${PROJECT_NAME}_CMAKE_C_FLAGS -D_GLIBCXX_USE_CXX11_ABI=0)
        list(APPEND ${PROJECT_NAME}_CMAKE_CXX_FLAGS -D_GLIBCXX_USE_CXX11_ABI=0)
        list(APPEND ${PROJECT_NAME}_CMAKE_CUDA_FLAGS -Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=0)
    endif()
endif()

list(APPEND ${PROJECT_NAME}_CMAKE_C_FLAGS -fdiagnostics-color=always)
list(APPEND ${PROJECT_NAME}_CMAKE_CXX_FLAGS -fdiagnostics-color=always)
list(APPEND ${PROJECT_NAME}_CMAKE_CUDA_FLAGS -Xcompiler=-fdiagnostics-color=always)

set(NO_SYSTEM_FROM_IMPORTED ON)

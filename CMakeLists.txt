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

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(CU_RTREE VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

include(ConfigureCXX)
include(ConfigureCUDA)
include(get_cpm)

CPMFindPackage(NAME     rmm
    VERSION             21.08.00
    GIT_REPOSITORY      https://github.com/rapidsai/rmm.git
    GIT_TAG             branch-21.08
    GIT_SHALLOW         TRUE
    UPDATE_DISCONNECTED FALSE
    OPTIONS             "BUILD_TESTS OFF"
                        "BUILD_BENCHMARKS OFF")

function(_set_target_properties_and_options TARGET)
    set_target_properties(${TARGET}
        PROPERTIES PREFIX                              ""
                   BUILD_RPATH                         "\$ORIGIN"
                   INSTALL_RPATH                       "\$ORIGIN"
                   CXX_STANDARD                        17
                   CXX_STANDARD_REQUIRED               ON
                   CUDA_STANDARD                       17
                   CUDA_STANDARD_REQUIRED              ON
                   NO_SYSTEM_FROM_IMPORTED             ON
                   POSITION_INDEPENDENT_CODE           ON
                   INTERFACE_POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(${TARGET}
        PRIVATE "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_CMAKE_C_FLAGS}>>"
                "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CMAKE_CXX_FLAGS}>>"
                "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${${PROJECT_NAME}_CMAKE_CUDA_FLAGS}>>"
    )
    target_include_directories(${TARGET}
        PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
    )
    target_link_libraries(${TARGET} rmm::rmm)
endfunction()

add_library(curtree SHARED src/utility.cpp)
add_executable(rtree_test_thrust src/rtree_test_thrust.cu)
add_executable(rtree_test_frontier src/rtree_test_frontier.cu)

_set_target_properties_and_options(curtree)
_set_target_properties_and_options(rtree_test_thrust)
_set_target_properties_and_options(rtree_test_frontier)

target_link_libraries(rtree_test_thrust curtree)
target_link_libraries(rtree_test_frontier curtree)

# Create a symlink to compile_commands.json for the llvm-vs-code-extensions.vscode-clangd plugin
execute_process(COMMAND
    ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)

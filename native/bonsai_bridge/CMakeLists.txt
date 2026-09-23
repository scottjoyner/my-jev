cmake_minimum_required(VERSION 3.20)

project(my_jev_bonsai_bridge LANGUAGES C CXX)

set(PRISM_LLAMA_ROOT "" CACHE PATH "Path to PrismML-Eng/llama.cpp checkout")

if(NOT PRISM_LLAMA_ROOT)
    message(FATAL_ERROR "PRISM_LLAMA_ROOT is required")
endif()

get_filename_component(PRISM_LLAMA_ROOT "${PRISM_LLAMA_ROOT}" ABSOLUTE)

if(NOT EXISTS "${PRISM_LLAMA_ROOT}/include/llama.h")
    message(FATAL_ERROR "PrismML llama.h not found under PRISM_LLAMA_ROOT")
endif()

if(NOT EXISTS "${PRISM_LLAMA_ROOT}/src/llama-ext.h")
    message(FATAL_ERROR "PrismML staging llama-ext.h not found under PRISM_LLAMA_ROOT")
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)

add_subdirectory(
    "${PRISM_LLAMA_ROOT}"
    "${CMAKE_BINARY_DIR}/prism-llama"
    EXCLUDE_FROM_ALL
)

add_executable(
    my-jev-bonsai-bridge
    bonsai_activation_bridge.cpp
)

target_compile_features(
    my-jev-bonsai-bridge
    PRIVATE
    cxx_std_17
)

target_include_directories(
    my-jev-bonsai-bridge
    PRIVATE
    "${PRISM_LLAMA_ROOT}/include"
    "${PRISM_LLAMA_ROOT}/src"
    "${PRISM_LLAMA_ROOT}/ggml/include"
)

target_link_libraries(
    my-jev-bonsai-bridge
    PRIVATE
    llama
    ggml
)

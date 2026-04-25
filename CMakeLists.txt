cmake_minimum_required(VERSION 3.22)
project(ultimateCGMD LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)

# Global CUDA flags
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Core library
add_library(core STATIC
    src/core/system.cu
)
target_include_directories(core PUBLIC ${CMAKE_SOURCE_DIR}/src)
set_target_properties(core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(tests)

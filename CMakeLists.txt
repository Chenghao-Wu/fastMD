cmake_minimum_required(VERSION 3.22)
project(fastMD LANGUAGES CXX CUDA)

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
    src/core/morton.cu
    src/neighbor/tile_list.cu
    src/neighbor/verlet_list.cu
    src/force/lj.cu
    src/force/table.cu
    src/force/fene.cu
    src/force/angle.cu
    src/integrate/langevin.cu
    src/integrate/nose_hoover.cu
    src/analysis/thermo.cu
    src/analysis/rg.cu
    src/analysis/correlator.cu
    src/io/dump.cu
    src/io/lammps_data.cpp
    src/io/table_parser.cpp
)
target_include_directories(core PUBLIC ${CMAKE_SOURCE_DIR}/src)
set_target_properties(core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Main executable
add_executable(fastMD src/main.cu src/io/config.cpp)
target_include_directories(fastMD PRIVATE src)
target_link_libraries(fastMD core CUDA::cudart CUDA::curand)
set_target_properties(fastMD PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(tests)

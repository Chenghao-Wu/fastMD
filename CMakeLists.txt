cmake_minimum_required(VERSION 3.22)

# Pre-flight: check for nvcc before project() enables CUDA language
find_program(NVCC_PROGRAM nvcc)
if(NOT NVCC_PROGRAM)
    message(FATAL_ERROR
        "CUDA Toolkit not found — nvcc is not on PATH.\n"
        "fastmd requires CUDA >= 11.0 with nvcc on PATH.\n"
        "Install CUDA from: https://developer.nvidia.com/cuda-downloads\n"
        "If CUDA is installed, add it to PATH, e.g.:\n"
        "  export PATH=/usr/local/cuda/bin:\$PATH\n"
        "Then try building again."
    )
endif()
message(STATUS "Found nvcc: ${NVCC_PROGRAM}")

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

# CMake does not auto-add -DNDEBUG for CUDA release builds
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -DNDEBUG")

# Core library
add_library(core STATIC
    src/core/system.cu
    src/core/morton.cu
    src/neighbor/tile_list.cu
    src/neighbor/verlet_list.cu
    src/force/lj.cu
    src/force/lj_coul_long.cu
    src/force/ewald_recip.cu
    src/force/ewald_excl.cu
    src/force/bond_harmonic.cu
    src/force/dihedral_opls.cu
    src/force/table.cu
    src/force/fene.cu
    src/force/angle.cu
    src/force/bond_table.cu
    src/force/angle_table.cu
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

# Install the fastMD binary into the Python package for scikit-build-core wheel packaging
set_target_properties(fastMD PROPERTIES OUTPUT_NAME fastmd)
install(TARGETS fastMD DESTINATION python/fastmd/_binary)

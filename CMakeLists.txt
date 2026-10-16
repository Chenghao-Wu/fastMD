cmake_minimum_required(VERSION 3.22)

# Pre-flight: check for nvcc before project() enables CUDA language
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    find_program(NVCC_PROGRAM nvcc)
    if(NOT NVCC_PROGRAM)
        message(FATAL_ERROR
            "CUDA Toolkit not found — nvcc is not on PATH.\n"
            "fastmd requires CUDA >= 11.0 with nvcc on PATH.\n"
            "Install CUDA from: https://developer.nvidia.com/cuda-downloads\n"
            "If CUDA is installed, add it to PATH, e.g.:\n"
            "  export PATH=/usr/local/cuda/bin:\$PATH\n"
            "or pass -DCMAKE_CUDA_COMPILER=/path/to/nvcc."
        )
    endif()
    message(STATUS "Found nvcc: ${NVCC_PROGRAM}")
endif()

# Remember whether the user chose GPU architectures (cache var or CUDAARCHS env)
# before project() fills in the compiler default.
if(DEFINED CMAKE_CUDA_ARCHITECTURES OR DEFINED ENV{CUDAARCHS})
    set(FASTMD_USER_CUDA_ARCH ON)
endif()

project(fastMD LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT FASTMD_USER_CUDA_ARCH)
    # Real SASS for common data-center/consumer GPUs plus PTX for the newest
    # supported architecture so future GPUs can JIT. Override with e.g.
    #   -DCMAKE_CUDA_ARCHITECTURES=86   (fast single-arch build)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
        set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real 90-real 120)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
        set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real 90)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
        set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75-real 80)
    endif()
endif()
message(STATUS "fastMD CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

find_package(CUDAToolkit REQUIRED)

# Tests are built by default for a top-level developer build, but not when
# the project is built as a Python wheel (scikit-build) or as a subproject.
if(DEFINED SKBUILD OR NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(_fastmd_tests_default OFF)
else()
    set(_fastmd_tests_default ON)
endif()
option(FASTMD_BUILD_TESTS "Build the fastMD unit tests (fetches GoogleTest)" ${_fastmd_tests_default})

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
    src/core/particles.cu
    src/core/topology.cu
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
    src/io/config.cpp
    src/io/lammps_data.cpp
    src/io/table_parser.cpp
    src/sim/simulation.cu
)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
# PUBLIC so host-only translation units (config.cpp, lammps_data.cpp) and
# every consumer get the CUDA include path and runtime.
target_link_libraries(core PUBLIC CUDA::cudart CUDA::curand)
set_target_properties(core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# Main executable (the installed command is `fastmd`)
add_executable(fastmd src/main.cu)
target_link_libraries(fastmd PRIVATE core)
set_target_properties(fastmd PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(FASTMD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Wheel builds (scikit-build-core) place the binary inside the Python package
# (site-packages/fastmd/_binary/fastmd); plain CMake installs go to bin/.
if(DEFINED SKBUILD)
    install(TARGETS fastmd DESTINATION fastmd/_binary)
else()
    install(TARGETS fastmd DESTINATION bin)
endif()

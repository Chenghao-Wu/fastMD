include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()

function(add_cuda_test name)
    add_executable(${name} ${name}.cu)
    target_link_libraries(${name} GTest::gtest_main CUDA::cudart)
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_cuda_test(test_pbc)
add_cuda_test(test_morton)
target_sources(test_morton PRIVATE ${CMAKE_SOURCE_DIR}/src/core/morton.cu)

add_cuda_test(test_tile_list)
target_sources(test_tile_list PRIVATE
    ${CMAKE_SOURCE_DIR}/src/neighbor/tile_list.cu
    ${CMAKE_SOURCE_DIR}/src/core/morton.cu)

add_cuda_test(test_lj)
target_sources(test_lj PRIVATE
    ${CMAKE_SOURCE_DIR}/src/force/lj.cu
    ${CMAKE_SOURCE_DIR}/src/neighbor/verlet_list.cu
    ${CMAKE_SOURCE_DIR}/src/core/morton.cu)

add_cuda_test(test_integrator)
target_sources(test_integrator PRIVATE
    ${CMAKE_SOURCE_DIR}/src/integrate/langevin.cu
    ${CMAKE_SOURCE_DIR}/src/force/lj.cu
    ${CMAKE_SOURCE_DIR}/src/neighbor/verlet_list.cu
    ${CMAKE_SOURCE_DIR}/src/core/morton.cu
    ${CMAKE_SOURCE_DIR}/src/core/system.cu)
target_link_libraries(test_integrator CUDA::curand)

add_cuda_test(test_fene)
target_sources(test_fene PRIVATE ${CMAKE_SOURCE_DIR}/src/force/fene.cu)

add_cuda_test(test_angle)
target_sources(test_angle PRIVATE ${CMAKE_SOURCE_DIR}/src/force/angle.cu)

add_cuda_test(test_thermo)
target_sources(test_thermo PRIVATE
    ${CMAKE_SOURCE_DIR}/src/analysis/thermo.cu
    ${CMAKE_SOURCE_DIR}/src/core/system.cu)

add_cuda_test(test_correlator)
target_sources(test_correlator PRIVATE ${CMAKE_SOURCE_DIR}/src/analysis/correlator.cu)

add_cuda_test(test_lammps_data)
target_sources(test_lammps_data PRIVATE ${CMAKE_SOURCE_DIR}/src/io/lammps_data.cpp)

add_cuda_test(test_rg)
target_sources(test_rg PRIVATE ${CMAKE_SOURCE_DIR}/src/analysis/rg.cu)

add_cuda_test(test_nose_hoover)
target_sources(test_nose_hoover PRIVATE
    ${CMAKE_SOURCE_DIR}/src/integrate/nose_hoover.cu
    ${CMAKE_SOURCE_DIR}/src/force/lj.cu
    ${CMAKE_SOURCE_DIR}/src/neighbor/verlet_list.cu
    ${CMAKE_SOURCE_DIR}/src/core/morton.cu
    ${CMAKE_SOURCE_DIR}/src/core/system.cu)

add_cuda_test(test_table_parser)
target_sources(test_table_parser PRIVATE
    ${CMAKE_SOURCE_DIR}/src/io/table_parser.cpp)

add_cuda_test(test_config)
target_sources(test_config PRIVATE
    ${CMAKE_SOURCE_DIR}/src/io/config.cpp
    ${CMAKE_SOURCE_DIR}/src/io/table_parser.cpp)

add_cuda_test(test_table)
target_sources(test_table PRIVATE
    ${CMAKE_SOURCE_DIR}/src/force/lj.cu
    ${CMAKE_SOURCE_DIR}/src/force/table.cu
    ${CMAKE_SOURCE_DIR}/src/neighbor/verlet_list.cu
    ${CMAKE_SOURCE_DIR}/src/core/morton.cu
    ${CMAKE_SOURCE_DIR}/src/io/table_parser.cpp)

add_cuda_test(test_mass)
target_sources(test_mass PRIVATE
    ${CMAKE_SOURCE_DIR}/src/integrate/langevin.cu
    ${CMAKE_SOURCE_DIR}/src/analysis/thermo.cu
    ${CMAKE_SOURCE_DIR}/src/core/system.cu)

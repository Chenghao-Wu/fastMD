find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

set(FASTMD_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fixtures)

# add_fastmd_test(<name> [CPU]) — CPU tests exercise host code only and run
# on machines without a GPU (CI); every other test is labelled "gpu".
function(add_fastmd_test name)
    cmake_parse_arguments(ARG "CPU" "" "" ${ARGN})
    add_executable(${name} ${name}.cu)
    target_link_libraries(${name} PRIVATE core GTest::gtest_main)
    target_compile_definitions(${name} PRIVATE
        FASTMD_TEST_DATA_DIR="${FASTMD_TEST_DATA_DIR}")
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME ${name} COMMAND ${name}
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    if(ARG_CPU)
        set_tests_properties(${name} PROPERTIES LABELS cpu)
    else()
        set_tests_properties(${name} PROPERTIES LABELS gpu)
    endif()
endfunction()

# Host-only tests
add_fastmd_test(test_table_parser CPU)
add_fastmd_test(test_config CPU)
add_fastmd_test(test_lammps_data CPU)

# Device tests
add_fastmd_test(test_pbc)
add_fastmd_test(test_morton)
add_fastmd_test(test_tile_list)
add_fastmd_test(test_lj)
add_fastmd_test(test_verlet)
add_fastmd_test(test_integrator)
add_fastmd_test(test_fene)
add_fastmd_test(test_angle)
add_fastmd_test(test_thermo)
add_fastmd_test(test_correlator)
add_fastmd_test(test_rg)
add_fastmd_test(test_nose_hoover)
add_fastmd_test(test_table)
add_fastmd_test(test_mass)
add_fastmd_test(test_bond_harmonic)
add_fastmd_test(test_dihedral_opls)
add_fastmd_test(test_ewald)

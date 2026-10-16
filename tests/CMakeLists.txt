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
add_fastmd_test(test_trajectory CPU)

# Device tests
add_fastmd_test(test_pbc)
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
add_fastmd_test(test_bonded_gather)
add_fastmd_test(test_ewald)
add_fastmd_test(test_pme)
add_fastmd_test(test_wall)
add_fastmd_test(test_simulation)

# Python interface (built with -DFASTMD_BUILD_PYTHON=ON; needs numpy)
if(TARGET _core)
    add_test(NAME test_python_api
             COMMAND ${Python_EXECUTABLE} -m unittest -v test_simulation_api
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python)
    set_tests_properties(test_python_api PROPERTIES
        LABELS gpu
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python;FASTMD_SOURCE_DIR=${PROJECT_SOURCE_DIR};FASTMD_BINARY=$<TARGET_FILE:fastmd>")
endif()

# Analysis module, trajectory reader and system builders (numpy; no GPU)
if(TARGET _core)
    add_test(NAME test_python_analysis
             COMMAND ${Python_EXECUTABLE} -m unittest -v test_analysis
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python)
    set_tests_properties(test_python_analysis PROPERTIES
        LABELS cpu
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python;FASTMD_SOURCE_DIR=${PROJECT_SOURCE_DIR}")
endif()

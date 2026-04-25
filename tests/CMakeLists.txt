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

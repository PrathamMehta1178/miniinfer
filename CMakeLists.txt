cmake_minimum_required(VERSION 3.16)
project(MiniInfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)  # benchmarks without optimization are meaningless
endif()

# -march=native lets the compiler use your CPU's vector (AVX) instructions.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
add_compile_options(-Wall -Wextra)

find_package(OpenMP REQUIRED)

add_library(miniinfer src/kernels.cpp src/model.cpp)
target_include_directories(miniinfer PUBLIC include)
target_link_libraries(miniinfer PUBLIC OpenMP::OpenMP_CXX)

add_executable(test_kernels tests/test_kernels.cpp)
target_link_libraries(test_kernels miniinfer)

add_executable(test_model tests/test_model.cpp)
target_link_libraries(test_model miniinfer)

add_executable(bench bench/bench.cpp)
target_link_libraries(bench miniinfer)

enable_testing()
add_test(NAME kernels COMMAND test_kernels)
add_test(NAME model COMMAND test_model ${CMAKE_SOURCE_DIR}/data)

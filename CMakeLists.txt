project(athenak_test_kernel)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(${CMAKE_SOURCE_DIR}/kokkos)

add_executable(athenak_test_kernel src/main.cpp src/data.hpp)
target_include_directories(athenak_test_kernel PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(athenak_test_kernel Kokkos::kokkos)
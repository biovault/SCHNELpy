cmake_minimum_required (VERSION 2.8.11)
project (py_hdi)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB HeaderFiles hdi/*.h)
file(GLOB SourceFiles hdi/*.cpp)

add_subdirectory (hdi/pybind11)

find_library(lz4 REQUIRED)

find_package(OpenMP REQUIRED)

pybind11_add_module (numpy_to_hsne ${HeaderFiles} ${SourceFiles})

target_link_libraries (numpy_to_hsne lz4 OpenMP::OpenMP_CXX)


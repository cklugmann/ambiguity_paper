cmake_minimum_required(VERSION 3.14)
project(amb_pdf)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.10.4
)
FetchContent_MakeAvailable(pybind11)

find_package(OpenMP)

add_library(amb_pdf_native STATIC amb_pdf.cpp)
target_include_directories(amb_pdf_native PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(OpenMP_CXX_FOUND)
    target_compile_options(amb_pdf_native PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(amb_pdf_native PUBLIC OpenMP::OpenMP_CXX)
endif()

pybind11_add_module(amb_pdf bindings.cpp)
target_link_libraries(amb_pdf PRIVATE amb_pdf_native)

set_target_properties(amb_pdf PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../seldon"
)


cmake_minimum_required(VERSION 3.10)
project(InternetRouting)

set(CMAKE_CXX_STANDARD 20)

include_directories(src/algo src/metrics src/semiring src/utilities)

file(GLOB_RECURSE SOURCES "src/*.cpp")

find_package(OpenMP REQUIRED)

add_executable(main ${SOURCES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
else()
    message(FATAL_ERROR "OpenMP was not found. Make sure OpenMP is installed and try again.")
endif()

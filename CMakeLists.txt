cmake_minimum_required(VERSION 3.10)
project(InternetRouting)

set(CMAKE_CXX_STANDARD 20)

include_directories(src/routing/algo src/routing/metrics src/routing/semiring src/routing/utilities)

file(GLOB_RECURSE SOURCES "src/*.cpp")

find_package(OpenMP REQUIRED)

add_executable(main ${SOURCES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
endif()

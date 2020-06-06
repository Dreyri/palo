cmake_minimum_required(VERSION 3.9)
project(ausarbeitung CXX)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

find_package(OpenMP REQUIRED)

set(SRC
    01_Original.cpp
    MoveParticles.cpp)

add_executable(ausarbeitung ${SRC})
target_link_libraries(ausarbeitung PRIVATE OpenMP::OpenMP_CXX)
set_property(TARGET ausarbeitung PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
target_compile_features(ausarbeitung PRIVATE cxx_std_17)
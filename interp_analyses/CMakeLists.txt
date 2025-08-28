# This file shows how to build and link a user project against GTSAM using CMake
###################################################################################
# To create your own project, replace "example" with the actual name of your project
cmake_minimum_required(VERSION 3.5)
project(GPAnalyses CXX)

# Point to the custom build directory
set(GTSAM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build")
# Find GTSAM, either from a local build, or from a Debian/Ubuntu package.
find_package(GTSAM REQUIRED)

# Include example headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../examples)
# add execs
add_executable(SimpleInterpExample simple_traj/SimpleInterpExample.cpp)


# By using CMake exported targets, a simple "link" dependency introduces the
# include directories (-I) flags, and add any other
# required build flags (e.g. C++11, etc.)
target_link_libraries(SimpleInterpExample PRIVATE gtsam;yaml-cpp)
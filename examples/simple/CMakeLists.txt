cmake_minimum_required(VERSION 3.8)

#####################################################################
project(saxpy VERSION 1.0 LANGUAGES CXX)
#####################################################################

find_package(Hemi 2.0 REQUIRED)
message(STATUS "Configured Hemi with GPU = ${HEMI_ENABLE_GPU} (${HEMI_GPU_LANG})")

# Define the source to compile:
add_executable(saxpy saxpy.cpp)
HemiLink(saxpy)

install(TARGETS saxpy DESTINATION bin)

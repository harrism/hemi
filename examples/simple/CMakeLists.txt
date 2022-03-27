cmake_minimum_required(VERSION 3.8)

#####################################################################
project(saxpy VERSION 1.0 LANGUAGES CXX)
#####################################################################

find_package(Hemi 2.0 REQUIRED)

# Define the source to compile:
add_executable(saxpy saxpy.cpp)
target_link_libraries(saxpy PUBLIC Hemi::hemi)

if(ENABLE_GPU)
    set_property(SOURCE saxpy.cpp PROPERTY LANGUAGE CUDA)
endif()

install(TARGETS saxpy DESTINATION bin)

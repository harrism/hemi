cmake_minimum_required(VERSION 3.8)

#####################################################################
project(hemi VERSION 2.0.1 LANGUAGES CXX)
#####################################################################

add_library(HemiCPU INTERFACE)
target_compile_features(HemiCPU INTERFACE cxx_std_11)
target_compile_definitions(HemiCPU INTERFACE HEMI_CUDA_DISABLE)
target_include_directories(HemiCPU
                           INTERFACE
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                            $<INSTALL_INTERFACE:include>
                           )

add_library(HemiCUDA INTERFACE)
target_compile_features(HemiCUDA INTERFACE cxx_std_11)
target_compile_options(HemiCUDA INTERFACE --expt-extended-lambda)
target_include_directories(HemiCUDA
                           INTERFACE
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                            $<INSTALL_INTERFACE:include>
                           )

# Don't craete the Hemi target here.
# It needs to be included by our dependencies instead.
# include(Hemi.cmake)

#####################################################################
install(DIRECTORY "${CMAKE_SOURCE_DIR}/hemi"
        DESTINATION "include"
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.inl"
)

####### None of the stuff below is needed or used ##################
####### if you install Hemi with CPM.             ##################

set(installable_libs HemiCPU HemiCUDA)

# Attach these libraries to the HemiTargets list of exported libs.
install(TARGETS ${installable_libs}
        DESTINATION lib
        EXPORT HemiTargets)

# Note: we choose the following location for cmake dependency info:
# <prefix>/lib/cmake/Hemi/
# install the targets to export
install(EXPORT HemiTargets
  FILE HemiTargets.cmake
  DESTINATION lib/cmake/Hemi
)

# Create a config helper so others can call find_package(Hemi)
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/HemiConfig.cmake"
  INSTALL_DESTINATION "lib/cmake/Hemi"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
# generate the version file for the config file
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/HemiConfigVersion.cmake"
  VERSION "${hemi_VERSION_MAJOR}.${hemi_VERSION_MINOR}"
  COMPATIBILITY AnyNewerVersion
)

# install the configuration files
install(FILES
  Hemi.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/HemiConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/HemiConfigVersion.cmake
  DESTINATION lib/cmake/Hemi
)

cmake_minimum_required(VERSION 3.8)

#####################################################################
project(hemi VERSION 2.0 LANGUAGES CXX)
#####################################################################

add_library(cpu INTERFACE)
target_compile_features(cpu INTERFACE cxx_std_11)
target_compile_definitions(cpu INTERFACE HEMI_CUDA_DISABLE)
target_include_directories(cpu
                           INTERFACE
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                            $<INSTALL_INTERFACE:include>
                           )

add_library(cuda INTERFACE)
target_compile_features(cuda INTERFACE cxx_std_11)
target_compile_options(cuda INTERFACE --expt-extended-lambda)
target_include_directories(cuda
                           INTERFACE
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                            $<INSTALL_INTERFACE:include>
                           )

#####################################################################
set(installable_libs cpu cuda)
#####################################################################

# Attach these libraries to the HemiTargets list of exported libs.
install(TARGETS ${installable_libs}
        DESTINATION lib
        EXPORT HemiTargets)

#####################################################################
install(DIRECTORY "${CMAKE_SOURCE_DIR}/hemi"
        DESTINATION "include"
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.inl"
)

# Note: we choose the following location for cmake dependency info:
# <prefix>/lib/cmake/Hemi/
# install the targets to export
install(EXPORT HemiTargets
  FILE HemiTargets.cmake
  NAMESPACE Hemi::
  DESTINATION lib/cmake/Hemi
)

# Create a config helper so others can call find_package(Hemi::hemi)
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/HemiConfig.cmake"
  INSTALL_DESTINATION "lib/cmake/Hemi"
  NO_SET_AND_CHECK_MACRO
  )
# generate the version file for the config file
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/HemiConfigVersion.cmake"
  VERSION "${hemi_VERSION_MAJOR}.${hemi_VERSION_MINOR}"
  COMPATIBILITY AnyNewerVersion
)
# install the configuration file
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/HemiConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/HemiConfigVersion.cmake
  DESTINATION lib/cmake/Hemi
  )
# allow this project to be used from the build directory (before install)
export(EXPORT HemiTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/HemiTargets.cmake"
)

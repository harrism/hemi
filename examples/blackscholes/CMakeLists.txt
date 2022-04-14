cmake_minimum_required(VERSION 3.14)

#####################################################################
project(blackscholes VERSION 1.0 LANGUAGES CXX)
#####################################################################

# ---- Get CPM ----

set(CPM_DOWNLOAD_VERSION 0.34.0)

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
       https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION}
  )
endif()

include(${CPM_DOWNLOAD_LOCATION})
# --- alternatively, use a locally packaged copy ---
#include(cmake/CPM.cmake)

CPMAddPackage("gh:harrism/Hemi@2.0.1")
include("${Hemi_SOURCE_DIR}/Hemi.cmake")
message(STATUS "Configured Hemi with GPU = ${HEMI_ENABLE_GPU} (${HEMI_GPU_LANG})")

# Define the source to compile:
add_executable(blackscholes blackscholes.cpp)
HemiLink(blackscholes)

install(TARGETS blackscholes DESTINATION bin)

# CMake code to detect CMake flags and set hemi's target accordingly
#
# sets: HEMI_ENABLE_GPU (TRUE / FALSE)
# if true, also sets: HEMI_GPU_LANG (CUDA / HIP)
#
# creates Hemi target using build properties for the setup above.
#
# Also provides HemiLink(<target>) function that will
# call target_link_libraries(<target> PUBLIC Hemi) and then mark every
# source file belonging to the target as compiled with GPU.

if(CMAKE_CUDA_ARCHITECTURES)
    set(HEMI_ENABLE_GPU TRUE CACHE BOOL "" FORCE)
    set(HEMI_GPU_LANG CUDA CACHE STRING "" FORCE)
    if(${CMAKE_VERSION} VERSION_LESS "3.17")
        message(FATAL_ERROR "Compilation for CUDA requires CMake 3.17 or later.")
    endif()
    enable_language(CUDA)
    message(STATUS "Setting up CUDA")
elseif(CMAKE_HIP_ARCHITECTURES)
    set(HEMI_ENABLE_GPU TRUE CACHE BOOL "" FORCE)
    set(HEMI_GPU_LANG HIP CACHE STRING "" FORCE)
    if(${CMAKE_VERSION} VERSION_LESS "3.21.3")
        message(FATAL_ERROR "Compilation for HIP requires CMake 3.21.3 or later.")
    endif()
    enable_language(HIP)
    message(STATUS "Setting up HIP using ROCM_ROOT = ${ROCM_ROOT}")
    set(CMAKE_MODULE_PATH "${ROCM_ROOT}/hip/cmake" ${CMAKE_MODULE_PATH})
else()
    set(HEMI_ENABLE_GPU FALSE CACHE BOOL "" FORCE)
endif()

add_library(Hemi INTERFACE IMPORTED)
if(HEMI_ENABLE_GPU)
    target_link_libraries(Hemi INTERFACE HemiCUDA)
else()
    target_link_libraries(Hemi INTERFACE HemiCPU)
endif()

function(HemiLink target_name)
  target_link_libraries( ${target_name} PUBLIC Hemi)
  if(HEMI_ENABLE_GPU)
    if(HEMI_GPU_LANG STREQUAL "HIP")
      set_target_properties( ${target_name} PROPERTIES LINKER_LANGUAGE "HIP")
      #set_target_properties( ${target_name}
      #  PROPERTIES HIP_SEPARABLE_COMPILATION ON)
      #set_target_properties( ${target_name}
      #  PROPERTIES HIP_RESOLVE_DEVICE_SYMBOLS ON)
      #target_link_libraries(${target_name} PRIVATE hip::device)
    endif()
    get_target_property(_srcs ${target_name} SOURCES)
    get_target_property(_src_dir ${target_name} SOURCE_DIR)
    # Mark all source files as GPU code.
    foreach(_src IN LISTS _srcs)
      set_source_files_properties(${_src} PROPERTIES LANGUAGE ${HEMI_GPU_LANG})
    endforeach()
  endif()
endfunction()

///////////////////////////////////////////////////////////////////////////////
// 
// "Hemi" CUDA Portable C/C++ Utilities
// 
// Copyright 2012-2014 NVIDIA Corporation
//
// License: BSD License, see LICENSE file in Hemi home directory
//
// The home for Hemi is https://github.com/harrism/hemi
//
///////////////////////////////////////////////////////////////////////////////
// Please see the file README.md (https://github.com/harrism/hemi/README.md) 
// for full documentation and discussion.
///////////////////////////////////////////////////////////////////////////////
#ifndef __HEMI_H__
#define __HEMI_H__

#include <stdio.h>
#include <assert.h>
#include "cuda_runtime_api.h"

/* HEMI_VERSION encodes the version number of the HEMI utilities.
 *
 *   HEMI_VERSION / 100000 is the major version.
 *   HEMI_VERSION / 100 % 1000 is the minor version.
 */
#define HEMI_VERSION 100000

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

// Convenience function for checking CUDA error state including 
// errors caused by asynchronous calls (like kernel launches). Note that
// this causes device synchronization, but is a no-op in release builds.
inline
cudaError_t checkCudaErrors()
{
  cudaError_t result = cudaSuccess;
  checkCuda(result = cudaGetLastError()); // runtime API errors
#if defined(DEBUG) || defined(_DEBUG)
  result = cudaDeviceSynchronize(); // async kernel launch errors
  if (result != cudaSuccess)
    fprintf(stderr, "CUDA Launch Error: %s\n", cudaGetErrorString(result));  
#endif
  return result;
}

#ifdef __CUDACC__ // CUDA compiler
  #define HEMI_CUDA_COMPILER              // to detect CUDACC compilation
  #define HEMI_LOC_STRING "Device"

  #ifdef __CUDA_ARCH__
    #define HEMI_DEV_CODE                 // to detect device compilation
  #endif
  
  #define HEMI_KERNEL(name)               __global__ void name ## _kernel
  #define HEMI_KERNEL_NAME(name)          name ## _kernel
  
  #if defined(DEBUG) || defined(_DEBUG)
    #define HEMI_KERNEL_LAUNCH(name, gridDim, blockDim, sharedBytes, streamId, ...) \
      do {                                                                     \
        name ## _kernel<<< (gridDim), (blockDim), (sharedBytes), (streamId) >>>\
            (__VA_ARGS__);                                                     \
        checkCudaErrors();                                                     \
      } while(0)
  #else
    #define HEMI_KERNEL_LAUNCH(name, gridDim, blockDim, sharedBytes, streamId, ...) \
      name ## _kernel<<< (gridDim) , (blockDim), (sharedBytes), (streamId) >>>(__VA_ARGS__)
  #endif

  #define HEMI_LAUNCHABLE                 __global__
  #define HEMI_LAMBDA                     __device__
  #define HEMI_DEV_CALLABLE               __host__ __device__
  #define HEMI_DEV_CALLABLE_INLINE        __host__ __device__ inline
  #define HEMI_DEV_CALLABLE_MEMBER        __host__ __device__
  #define HEMI_DEV_CALLABLE_INLINE_MEMBER __host__ __device__ inline

  // Memory specifiers
  #define HEMI_MEM_DEVICE                 __device__

  // Constants: declares both a device and a host copy of this constant
  // static and extern flavors can be used to declare static and extern
  // linkage as required.
  #define HEMI_DEFINE_CONSTANT(def, value) \
      __constant__ def ## _devconst = value; \
      def ## _hostconst = value
  #define HEMI_DEFINE_STATIC_CONSTANT(def, value) \
      static __constant__ def ## _devconst = value; \
      static def ## _hostconst = value
  #define HEMI_DEFINE_EXTERN_CONSTANT(def) \
      extern __constant__ def ## _devconst; \
      extern def ## _hostconst

  // use to access device constant explicitly
  #define HEMI_DEV_CONSTANT(name) name ## _devconst

  // use to access a constant defined with HEMI_DEFINE_*_CONSTANT
  // automatically chooses either host or device depending on compilation
  #ifdef HEMI_DEV_CODE
    #define HEMI_CONSTANT(name) name ## _devconst
  #else
    #define HEMI_CONSTANT(name) name ## _hostconst
  #endif

  #if !defined(HEMI_ALIGN)
    #define HEMI_ALIGN(n) __align__(n)
  #endif

#else             // host compiler
  #define HEMI_HOST_COMPILER              // to detect non-CUDACC compilation
  #define HEMI_LOC_STRING "Host"

  #define HEMI_KERNEL(name)               void name
  #define HEMI_KERNEL_NAME(name)          name
  #define HEMI_KERNEL_LAUNCH(name, gridDim, blockDim, sharedBytes, streamId, ...) name(__VA_ARGS__)

  #define HEMI_LAUNCHABLE
  #define HEMI_LAMBDA
  #define HEMI_DEV_CALLABLE               
  #define HEMI_DEV_CALLABLE_INLINE        inline
  #define HEMI_DEV_CALLABLE_MEMBER
  #define HEMI_DEV_CALLABLE_INLINE_MEMBER inline

  // memory specifiers
  #define HEMI_MEM_DEVICE

  #define HEMI_DEFINE_CONSTANT(def, value) def ## _hostconst = value
  #define HEMI_DEFINE_STATIC_CONSTANT(def, value) static def ## _hostconst = value
  #define HEMI_DEFINE_EXTERN_CONSTANT(def) extern def ## _hostconst

  #undef HEMI_DEV_CONSTANT // requires NVCC, so undefined here!
  #define HEMI_CONSTANT(name) name ## _hostconst      
  
  #if !defined(HEMI_ALIGN)

    #if defined(__GNUC__)
      #define HEMI_ALIGN(n) __attribute__((aligned(n)))
    #elif defined(_MSC_VER)
      #define HEMI_ALIGN(n) __declspec(align(n))
    #else
      #error "Please provide a definition of HEMI_ALIGN for your host compiler!"
    #endif
  
  #endif

#endif

// Helper macro for defining device functors that can be launched as kernels
#define HEMI_KERNEL_FUNCTION(name, ...)                \
  struct name {                                             \
      HEMI_DEV_CALLABLE_MEMBER void operator()(__VA_ARGS__) const;  \
  };                                                        \
  HEMI_DEV_CALLABLE_MEMBER void name::operator()(__VA_ARGS__) const
;

namespace hemi {

  inline cudaError_t deviceSynchronize() {
  #ifdef HEMI_CUDA_COMPILER
    return checkCuda(cudaDeviceSynchronize());
  #else
    return cudaSuccess;
  #endif
  }

} // namespace hemi

#endif // __HEMI_H__

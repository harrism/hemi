///////////////////////////////////////////////////////////////////////////////
// 
// "HEMI" CUDA Portable C/C++ Utilities
// 
// Copyright 2012 NVIDIA Corporation
//
// License: Apache License, v2.0 http://www.apache.org/licenses/LICENSE-2.0.html
//
// The home for HEMI is https://github.com/harrism/hemi
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
#define HEMI_VERSION 000100

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

  #define HEMI_DEV_CALLABLE               __host__ __device__
  #define HEMI_DEV_CALLABLE_INLINE        __host__ __device__ inline
  #define HEMI_DEV_CALLABLE_MEMBER        __host__ __device__
  #define HEMI_DEV_CALLABLE_INLINE_MEMBER __host__ __device__ inline

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

  #define HEMI_DEV_CALLABLE               
  #define HEMI_DEV_CALLABLE_INLINE        inline
  #define HEMI_DEV_CALLABLE_MEMBER
  #define HEMI_DEV_CALLABLE_INLINE_MEMBER inline

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

// Note: the following two functions demonstrate using the same code to process
// 1D arrays of data/computations either with a parallel grid of threads on the 
// CUDA device, or with a sequential loop on the host. For example, we might 
// use them like this.
//
//  int offset = hemiGetElementOffset();
//  int stride = hemiGetElementStride();
// 
//  for(int idx = offset; idx < N; idx += stride)
//    processElement(elementsOut, elementsIn, idx, anotherArgument);
//

// Returns the offset of the current thread's element within the grid for device
// code compiled with NVCC, or 0 for sequential host code.
HEMI_DEV_CALLABLE_INLINE
int hemiGetElementOffset() 
{
#ifdef HEMI_DEV_CODE
  return blockIdx.x * blockDim.x + threadIdx.x;
#else
  return 0;
#endif
}

// Returns the stride of the current grid (blockDim.x * gridDim.x) for device 
// code compiled with NVCC, or 1 for sequential host code.
HEMI_DEV_CALLABLE_INLINE
int hemiGetElementStride() 
{
#ifdef HEMI_DEV_CODE
  return blockDim.x * gridDim.x;
#else
  return 1;
#endif
}

#endif // __HEMI_H__

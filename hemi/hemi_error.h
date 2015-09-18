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

#pragma once

#include <stdio.h>
#include <assert.h>

namespace hemi {
  
    enum Error_t {
        success = 0,
        cudaError = 1
    };
}

#ifndef HEMI_CUDA_DISABLE

    #include "cuda_runtime_api.h"

    // Convenience function for checking CUDA runtime API results
    // can be wrapped around any runtime API call. No-op in release builds.
    inline cudaError_t checkCuda(cudaError_t result)
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
    inline cudaError_t checkCudaErrors()
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
#endif // HEMI_CUDA_DISABLE

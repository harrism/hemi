///////////////////////////////////////////////////////////////////////////////
// 
// "Hemi" CUDA Portable C/C++ Utilities
// 
// Copyright 2012-2015 NVIDIA Corporation
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

// Functions available inside "device" code (whether compiled for CUDA device
// or CPU.)

#include "hemi.h"

namespace hemi
{
    template <unsigned Dim = 0>
	HEMI_DEV_CALLABLE_INLINE
    unsigned int globalThreadIndex() {
    #ifdef HEMI_DEV_CODE
        if (Dim == 0)
    	return threadIdx.x + blockIdx.x * blockDim.x;
        else if (Dim == 1)
            return threadIdx.y + blockIdx.y * blockDim.y;
        else if (Dim == 2)
            return threadIdx.z + blockIdx.z * blockDim.z;
        else
            return 0;
    #else
    	return 0;
    #endif
    }

    template <unsigned Dim = 0>
    HEMI_DEV_CALLABLE_INLINE
    unsigned int globalThreadCount() {
    #ifdef HEMI_DEV_CODE
        if (Dim == 0)
    	return blockDim.x * gridDim.x;
        else if (Dim == 1)
    	    return blockDim.y * gridDim.y;
        else if (Dim == 2)
    	    return blockDim.z * gridDim.z;
        else
            return 0;
    #else
    	return 1;
    #endif
    }


    HEMI_DEV_CALLABLE_INLINE
    unsigned int globalBlockCount() {
    #ifdef HEMI_DEV_CODE
    	return gridDim.x;
    #else
    	return 1;
    #endif
    }


    HEMI_DEV_CALLABLE_INLINE
    unsigned int localThreadIndex() {
    #ifdef HEMI_DEV_CODE
    	return threadIdx.x;
    #else
    	return 0;
    #endif
    }


    HEMI_DEV_CALLABLE_INLINE
    unsigned int localThreadCount() {
    #ifdef HEMI_DEV_CODE
    	return blockDim.x;
    #else
    	return 1;
    #endif
    }


    HEMI_DEV_CALLABLE_INLINE
    unsigned int globalBlockIndex() {
    #ifdef HEMI_DEV_CODE
    	return blockIdx.x;
    #else
    	return 0;
    #endif
    }


    HEMI_DEV_CALLABLE_INLINE
    void synchronize() {
    #ifdef HEMI_DEV_CODE
        __syncthreads();
    #endif
    }
}

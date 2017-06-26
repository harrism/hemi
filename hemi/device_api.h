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
	///////Global Grid Gets//////////////////////////////////////////////////////
	HEMI_DEV_CALLABLE_INLINE
		int globalThreadIndex() {
#ifdef HEMI_DEV_CODE
		return (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) // get block idx
			* (blockDim.x * blockDim.y * blockDim.z) // multiply by num blocks
			+ threadIdx.x // add thread x component
			+ threadIdx.y * blockDim.x // add thread y component
			+ threadIdx.z * (blockDim.x * blockDim.y); // add thread z component
#else
		return 0;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int globalThreadCount() {
#ifdef HEMI_DEV_CODE
		return blockDim.x * gridDim.x * blockDim.y * gridDim.y * blockDim.z * gridDim.z;
#else
		return 1;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int globalBlockCount() {
#ifdef HEMI_DEV_CODE
		return gridDim.x * gridDim.y * gridDim.z;
#else
		return 1;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int localThreadIndex() {
#ifdef HEMI_DEV_CODE
		return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
#else
		return 0;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int localThreadCount() {
#ifdef HEMI_DEV_CODE
		return blockDim.x * blockDim.y * blockDim.z;
#else
		return 1;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int globalBlockIndex() {
#ifdef HEMI_DEV_CODE
		return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
#else
		return 0;
#endif
	}

	///////X-Demension Grid Gets//////////////////////////////////////////////////////
	HEMI_DEV_CALLABLE_INLINE
		int xGlobalThreadIndex() {
#ifdef HEMI_DEV_CODE
		return threadIdx.x + blockIdx.x * blockDim.x;
#else
		return 0;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int xGlobalThreadCount() {
#ifdef HEMI_DEV_CODE
		return blockDim.x * gridDim.x;
#else
		return 1;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int xGlobalBlockCount() {
#ifdef HEMI_DEV_CODE
		return gridDim.x;
#else
		return 1;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int xLocalThreadIndex() {
#ifdef HEMI_DEV_CODE
		return threadIdx.x;
#else
		return 0;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int xLocalThreadCount() {
#ifdef HEMI_DEV_CODE
		return blockDim.x;
#else
		return 1;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int xGlobalBlockIndex() {
#ifdef HEMI_DEV_CODE
		return blockIdx.x;
#else
		return 0;
#endif
	}

	///////Y-Demension Grid Gets//////////////////////////////////////////////////////
	HEMI_DEV_CALLABLE_INLINE
		int yGlobalThreadIndex() {
#ifdef HEMI_DEV_CODE
		return threadIdx.y + blockIdx.y * blockDim.y;
#else
		return 0;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int yGlobalThreadCount() {
#ifdef HEMI_DEV_CODE
		return blockDim.y * gridDim.y;
#else
		return 1;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int yGlobalBlockCount() {
#ifdef HEMI_DEV_CODE
		return gridDim.y;
#else
		return 1;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int yLocalThreadIndex() {
#ifdef HEMI_DEV_CODE
		return threadIdx.y;
#else
		return 0;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int yLocalThreadCount() {
#ifdef HEMI_DEV_CODE
		return blockDim.y;
#else
		return 1;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int yGlobalBlockIndex() {
#ifdef HEMI_DEV_CODE
		return blockIdx.y;
#else
		return 0;
#endif
	}

	///////Z-Demension Grid Gets//////////////////////////////////////////////////////
	HEMI_DEV_CALLABLE_INLINE
		int zGlobalThreadIndex() {
#ifdef HEMI_DEV_CODE
		return threadIdx.z + blockIdx.z * blockDim.z;
#else
		return 0;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int zGlobalThreadCount() {
#ifdef HEMI_DEV_CODE
		return blockDim.z * gridDim.z;
#else
		return 1;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int zGlobalBlockCount() {
#ifdef HEMI_DEV_CODE
		return gridDim.z;
#else
		return 1;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int zLocalThreadIndex() {
#ifdef HEMI_DEV_CODE
		return threadIdx.z;
#else
		return 0;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int zLocalThreadCount() {
#ifdef HEMI_DEV_CODE
		return blockDim.z;
#else
		return 1;
#endif
	}

	HEMI_DEV_CALLABLE_INLINE
		int zGlobalBlockIndex() {
#ifdef HEMI_DEV_CODE
		return blockIdx.z;
#else
		return 0;
#endif
	}

	///////Synchronize/////////////////////////////////////////////////////////
    HEMI_DEV_CALLABLE_INLINE
    void synchronize() {
    #ifdef HEMI_DEV_CODE
        __syncthreads();
    #endif
    }

	///////Split Execution/////////////////////////////////////////////////////
#ifdef HEMI_DEV_CODE
#define SPLIT_DATA(gpu, cpu) (gpu)
#define	GPU_ONLY(gpu_data) (gpu_data)
#define	CPU_ONLY(cpu_data)
#else
#define SPLIT_DATA(gpu, cpu) (cpu)
#define	GPU_ONLY(gpu_data)
#define	CPU_ONLY(cpu_data) (cpu_data)
#endif

	///////Indexing Functions//////////////////////////////////////////////////
	HEMI_DEV_CALLABLE_INLINE
	int unflatten_grid_x(int index, int size_x) {
		return index % size_x;
	}

	HEMI_DEV_CALLABLE_INLINE
	int unflatten_grid_y(int index, int size_x, int size_y) {
		return (index / size_x) % size_y;
	}

	HEMI_DEV_CALLABLE_INLINE
	int unflatten_grid_z(int index, int size_x, int size_y) {
		return index / (size_x * size_y);
	}

	HEMI_DEV_CALLABLE_INLINE
	int flatten_grid_2D(int index_x, int index_y, int size_x) {
		return index_x + size_x * index_y;
	}

	HEMI_DEV_CALLABLE_INLINE
	int flatten_grid_3D(int index_x, int index_y, int index_z, int size_x, int size_y) {
		return index_x + size_x * (index_y + size_y * index_z);
	}
}

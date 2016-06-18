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

#include "hemi/hemi.h"
#include <cstring>
#include <type_traits> 

#ifndef HEMI_ARRAY_DEFAULT_LOCATION
  #ifdef HEMI_CUDA_COMPILER
    #define HEMI_ARRAY_DEFAULT_LOCATION hemi::device
  #else
    #define HEMI_ARRAY_DEFAULT_LOCATION hemi::host
  #endif
#endif

namespace hemi {

	//template <typename T> class Table1D; // forward decl
	//template <typename T> class Table2D; // forward decl
	template <typename T> class Table3D; // forward decl

	// move these typedefs to hemi.h maybe
	// http://tipsandtricks.runicsoft.com/Cpp/MemberFunctionPointers.html
	typedef float ( hemi::Table3D::*table ) (float, float, float);

	enum Location {
        host   = 0,
        device = 1
    };

    template <typename T>
    class Table3D 
    {
    public:
    	Table3D(size_t nx, size_t ny, size_t nz) :
    	nSize(nx * ny * nz);
    	nSizeX(nx);
    	nSizeY(ny);
    	nSizeZ(nz);
    	{	
    	}

    	~Table3D()
    	{
    		deallocateDevice();
            deallocateHost();
    	}

    	table GetTable(T x, T y, T z)
    	{
    		// notice the typedef
    		table tbl;
#ifdef HEMI_DEV_CODE
    		tbl = &hemi::Table3D::dEval;
#else
    		tbl = &hemi::Table3D::hEval;
#endif
    		return tbl;
    	}

    private:
    	size_t nSize;
    	size_t nSizeX;
    	size_t nSizeY;
    	size_t nSizeZ;

    	mutable T *hPtr;
    	cudaArray *dPtr;
        
		cudaMemcpy3DParms copyParams;
		cudaExtent volumeSize;
		cudaTextureObject_t tex;

        mutable bool    isHostAlloced;
        mutable bool    isDeviceAlloced;        

        mutable bool    isHostValid;
        mutable bool    isDeviceValid;

    protected:
        void allocateHost() const
        {
            assert(!isHostAlloced);
            hPtr = new T[nSize];    
                
            isHostAlloced = true;
            isHostValid = false;
        }

    	void allocateDevice() const
    	{
#ifndef HEMI_CUDA_DISABLE
    		assert(!isDeviceAlloced);
    	    volumeSize = make_cudaExtent(nSizeX, nSizeY, nSizeZ);
  			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
  			cudaMalloc3DArray(dPtr, &channelDesc, volumeSize);
			copyParams.dstArray = dPtr;

    	    isDeviceAlloced = true;
            isDeviceValid = false;
#endif
    	}

        void deallocateHost()
        {
            if (isHostAlloced) {
                delete [] hPtr;
                nSize = 0;
                nSizeX = 0;
                nSizeY = 0;
                nSixeZ = 0;
                isHostAlloced = false;
                isHostValid   = false;
            }
        }

        void deallocateDevice()
        {
#ifndef HEMI_CUDA_DISABLE
            if (isDeviceAlloced) {
                checkCuda( cudaFreeArray(dPtr) );
                checkCuda( cudaDestroyTextureObject(tex) );
                isDeviceAlloced = false;
                isDeviceValid   = false;
            }
#endif
        }

        void copyHostToDevice() const
        {
#ifndef HEMI_CUDA_DISABLE
            assert(isHostAlloced);
            if (!isDeviceAlloced) allocateDevice();
            copyParams.srcPtr = make_cudaPitchedPtr((void*)hPtr, volumeSize.width * sizeof(float), volumeSize.width, volumeSize.height);                                                                                                                                        
  			copyParams.extent   = volumeSize;
  			copyParams.kind     = cudaMemcpyHostToDevice;
  			cudaMemcpy3D(&copyParams);

  			// bind to texture
  			struct cudaResourceDesc resDesc;
  			memset(&resDesc, 0, sizeof(resDesc));
  			resDesc.resType =  cudaResourceTypeArray;
  			resDesc.res.linear.devPtr = dPtr;
  			resDesc.res.linear.sizeInBytes = volumeSize.width * volumeSize.height * volumeSize.depth * sizeof(T);
			resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
 			resDesc.res.linear.desc.x = sizeof(T) * 8;
  			struct cudaTextureDesc texDesc;
 			memset(&texDesc, 0, sizeof(texDesc));
 			//texDesc.normalizedCoords = false;                                                                                                                                     
 			texDesc.normalizedCoords = true;
 			texDesc.readMode = cudaReadModeElementType;
  			//texDesc.filterMode = cudaFilterModePoint;                                                                                                                             
  			texDesc.filterMode = cudaFilterModeLinear;

  			texDesc.addressMode[0] = cudaAddressModeClamp;
 			texDesc.addressMode[1] = cudaAddressModeClamp;
 			texDesc.addressMode[2] = cudaAddressModeClamp;

 			cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

            isDeviceValid = true;
#endif
        }

        void copyDeviceToHost() const
        {
/*#ifndef HEMI_CUDA_DISABLE
            assert(isDeviceAlloced);
            if (!isHostAlloced) allocateHost();
            checkCuda( cudaMemcpy(hPtr, 
                                  dPtr, 
                                  nSize * sizeof(T), 
                                  cudaMemcpyDeviceToHost) );
            isHostValid = true;
#endif*/
        }

    };


}
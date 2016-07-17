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
#include <climits>
#include <type_traits> 

#ifndef HEMI_ARRAY_DEFAULT_LOCATION
  #ifdef HEMI_CUDA_COMPILER
    #define HEMI_ARRAY_DEFAULT_LOCATION hemi::device
  #else
    #define HEMI_ARRAY_DEFAULT_LOCATION hemi::host
  #endif
#endif

namespace hemi {

  inline int index(int x, int y, int z, int dx, int dy)
  {
    return x + dx * (y + dy*z);
  }

  
  //template <typename T> class Table1D; // forward decl
  //template <typename T> class Table2D; // forward decl
  template <typename T> class Table3D; // forward decl
  
  // EDIT: this idea wont work!
  // move these typedefs to hemi.h maybe.
  // http://tipsandtricks.runicsoft.com/Cpp/MemberFunctionPointers.html
  //typedef float ( hemi::Table3D::*table ) (float, float, float);
  
  template <typename T>
    struct table3D {
      
      HEMI_DEV_CALLABLE_INLINE_MEMBER T lookup(const T x, const T y, const T z) const
      {
#ifdef HEMI_DEV_CODE
	return tex3D<T>(texture, x, y, z);
#else
	T xd = x;//(x - xbound[0]) * inv_cell_size[0];
	T yd = y;// (y - ybound[0]) * inv_cell_size[1];
	T zd = z;//(z - zbound[0]) * inv_cell_size[2];

	int ubx = static_cast<int>(xd);
	int uby = static_cast<int>(yd);
	int ubz = static_cast<int>(zd);

	int obx = ubx + 1;
	int oby = uby + 1;
	int obz = ubz + 1;

	const T v[] = { hPtr[hemi::index(ubx, uby, ubz, sizeX, sizeY)], hPtr[hemi::index(ubx, uby, obz, sizeX, sizeY)],
			hPtr[hemi::index(ubx, oby, ubz, sizeX, sizeY)], hPtr[hemi::index(ubx, oby, obz, sizeX, sizeY)],
			hPtr[hemi::index(obx, uby, ubz, sizeX, sizeY)], hPtr[hemi::index(obx, uby, obz, sizeX, sizeY)],
			hPtr[hemi::index(obx, oby, ubz, sizeX, sizeY)], hPtr[hemi::index(obx, oby, obz, sizeX, sizeY)] };
	
	xd -= (float)ubx;
	yd -= (float)uby;
	zd -= (float)ubz;
  
	float i1 = v[0] * (1 - zd) + v[1] * zd;
	float i2 = v[2] * (1 - zd) + v[3] * zd;
	float j1 = v[4] * (1 - zd) + v[5] * zd;
	float j2 = v[6] * (1 - zd) + v[7] * zd;
  
	float w1 = i1 * (1 - yd) + i2 * yd;
	float w2 = j1 * (1 - yd) + j2 * yd;

	float result = w1 * (1 - xd) + w2 * xd;
	return result;
#endif
      };
      
#ifndef HEMI_CUDA_DISABLE
      cudaTextureObject_t texture;
#endif
      mutable T *hPtr;
      int sizeX;
      int sizeY; 
      int sizeZ;
    };
  
  // already declared in array.h
  /*enum Location {
    host   = 0,
    device = 1
    };*/
  
  template <typename T>
    class Table3D 
    {
    public:
    Table3D(size_t nx, size_t ny, size_t nz, T *data) :
      nSize(nx * ny * nz),
	nSizeX(nx),
	nSizeY(ny),
	nSizeZ(nz)
	{	
	  // this is unsafe!
	  _table.hPtr = data;
	  _table.sizeX = nx;
	  _table.sizeY = ny;
	  _table.sizeZ = nz;

	  isHostAlloced = true;
	}
      
      ~Table3D()
	{
	  deallocateDevice();
	  deallocateHost();
	}
      
      const table3D<T> readOnlyTable() const
	{
	  return _table;
	}
      
    private:
      size_t nSize;
      size_t nSizeX;
      size_t nSizeY;
      size_t nSizeZ;
      
#ifndef HEMI_CUDA_DISABLE
      cudaArray *dPtr;
      
      cudaMemcpy3DParms copyParams;
      cudaExtent volumeSize;
#endif
      table3D<T> _table;
      
      mutable bool    isHostAlloced;
      mutable bool    isDeviceAlloced;        
      
      mutable bool    isHostValid;
      mutable bool    isDeviceValid;
      
    protected:
      void allocateHost() const
      {
	assert(!isHostAlloced);
	_table.hPtr = new T[nSize];    
	
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
	  delete [] _table.hPtr;
	  nSize = 0;
	  nSizeX = 0;
	  nSizeY = 0;
	  nSizeZ = 0;
	  isHostAlloced = false;
	  isHostValid   = false;
	}
      }
      
      void deallocateDevice()
      {
#ifndef HEMI_CUDA_DISABLE
	if (isDeviceAlloced) {
	  checkCuda( cudaFreeArray(dPtr) );
	  checkCuda( cudaDestroyTextureObject(_table.texture) );
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
	copyParams.srcPtr = make_cudaPitchedPtr((void*)_table.hPtr, volumeSize.width * sizeof(float), volumeSize.width, volumeSize.height);
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
	resDesc.res.linear.desc.x = sizeof(T) * CHAR_BIT;
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.normalizedCoords = true;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.filterMode = cudaFilterModeLinear;
	
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaAddressModeClamp;
	
	cudaCreateTextureObject(&_table.texture, &resDesc, &texDesc, NULL);
	
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

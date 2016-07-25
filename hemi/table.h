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
#include <iostream>
#include <cstring>
#include <climits>
#include <type_traits> 

/*#ifndef HEMI_ARRAY_DEFAULT_LOCATION
  #ifdef HEMI_CUDA_COMPILER
    #define HEMI_ARRAY_DEFAULT_LOCATION hemi::device
  #else
    #define HEMI_ARRAY_DEFAULT_LOCATION hemi::host
  #endif
  #endif*/

namespace hemi {

  /* flatten a 2D array index into 1D */
  HEMI_DEV_CALLABLE_INLINE int index(const int x, const int y, const int dx) {
    return x + dx * y;
  }

  /* flatten a 3D array index into 1D */
  HEMI_DEV_CALLABLE_INLINE int index(const int x, const int y, const int z, const int dx, const int dy) {
    return x + dx * (y + dy * z);
  }

  //template <typename T> class Table1D; // forward decl
  //template <typename T> class Table2D; // forward decl
  template <typename T> class Table3D; // forward decl
  
  /* this is essentially a wrapper for cudaTextureObject, which should be passed by value to the kernel */
  template <typename T>
    struct table3 {      
      /* look up element value, i.e. no interpolation between cells is applied */
      HEMI_DEV_CALLABLE_INLINE_MEMBER T getElement(const int i, const int j, const int k) const
      {
#ifdef HEMI_DEV_CODE
        return tex3D<T>(texture, (float)i + 0.5f, (float)j + 0.5f, (float)k + 0.5f);
#else
	return hPtr[hemi::index(i, j, k, size[0], size[1])];
#endif
      };
      
      /* lookup function, linearly interpolates between cells */
      HEMI_DEV_CALLABLE_INLINE_MEMBER T lookup(const float x, const float y, const float z) const
      {
	float xd = (x - low_edge[0]) * reciprocal_cell_size[0];
        float yd = (y - low_edge[1]) * reciprocal_cell_size[1];
        float zd = (z - low_edge[2]) * reciprocal_cell_size[2];
	
#ifdef HEMI_DEV_CODE
	return tex3D<T>(texture, xd + 0.5f, yd + 0.5f, zd + 0.5f);
#else
	int ubx = (int)xd;
	int uby = (int)yd;
	int ubz = (int)zd;
	
	int obx = ubx + 1;
	int oby = uby + 1;
	int obz = ubz + 1;

	xd -= std::floor(ubx);
	yd -= std::floor(uby);
	zd -= std::floor(ubz);

	float i1 = hPtr[hemi::index(ubx, uby, ubz, size[0], size[1])] * (1.0f - zd) 
	  + hPtr[hemi::index(ubx, uby, obz, size[0], size[1])] * zd;
        float i2 = hPtr[hemi::index(ubx, oby, ubz, size[0], size[1])] * (1.0f - zd) 
	  + hPtr[hemi::index(ubx, oby, obz, size[0], size[1])] * zd;
        float j1 = hPtr[hemi::index(obx, uby, ubz, size[0], size[1])] * (1.0f - zd) 
	  + hPtr[hemi::index(obx, uby, obz, size[0], size[1])] * zd;
	float j2 = hPtr[hemi::index(obx, oby, ubz, size[0], size[1])] * (1.0f - zd) 
	  + hPtr[hemi::index(obx, oby, obz, size[0], size[1])] * zd;

	float w1 = i1 * (1.0f - yd) + i2 * yd;
	float w2 = j1 * (1.0f - yd) + j2 * yd;

	return w1 * (1.0f - xd) + w2 * xd;
#endif
      };
      float low_edge[3];
      float reciprocal_cell_size[3];
      int size[3];
#ifndef HEMI_CUDA_DISABLE
      cudaTextureObject_t texture;
#endif
      mutable T *hPtr;
    };
  
  template <typename T>
    class Table3D 
    {
    public:
      Table3D(size_t nx, size_t ny, size_t nz,
	      float low_edge_x, float low_edge_y, float low_edge_z,
	      float up_edge_x, float up_edge_y, float up_edge_z) 
	{
	  h_table.size[0] = nx;
	  h_table.size[1] = ny;
	  h_table.size[2] = nz;
	  
	  // set the table range                                                                                                                                        
          h_table.low_edge[0] = low_edge_x;
          h_table.low_edge[1] = low_edge_y;
          h_table.low_edge[2] = low_edge_z;

	  // width of each cell                                                                                                   
          h_table.reciprocal_cell_size[0] = (nx-1) / static_cast<float>(up_edge_x - low_edge_x);
          h_table.reciprocal_cell_size[1] = (ny-1) / static_cast<float>(up_edge_y - low_edge_y);
          h_table.reciprocal_cell_size[2] = (nz-1) / static_cast<float>(up_edge_z - low_edge_z);

	  normalized_coords = false;
	  isHostAlloced = false;
	  isDeviceAlloced = false;
	  isHostValid = false;
	  isDeviceValid = false;
	  isForeignHostPtr = false;

	  allocateHost();

#ifndef HEMI_CUDA_DISABLE
	  allocateDevice();
#endif
	}

    Table3D(T *data,
	    size_t nx, size_t ny, size_t nz,
	    float low_edge_x, float low_edge_y, float low_edge_z,
	    float up_edge_x, float up_edge_y, float up_edge_z) /*:
      nSize(nx * ny * nz),
	nSizeX(nx),
	nSizeY(ny),
	nSizeZ(nz)*/
	{	
	  // this is unsafe!
	  h_table.hPtr = data;
	  h_table.size[0] = nx;
	  h_table.size[1] = ny;
	  h_table.size[2] = nz;

	  // set the table range
	  h_table.low_edge[0] = low_edge_x;
	  h_table.low_edge[1] = low_edge_y;
	  h_table.low_edge[2] = low_edge_z;
	  /*_table.up_edge[0] = up_edge_x;
	  _table.up_edge[1] = up_edge_y;
          _table.up_edge[2] = up_edge_z;*/

	  // width of each cell
	  h_table.reciprocal_cell_size[0] = (nx-1) / static_cast<float>(up_edge_x - low_edge_x);
	  h_table.reciprocal_cell_size[1] = (ny-1) / static_cast<float>(up_edge_y - low_edge_y);
          h_table.reciprocal_cell_size[2] = (nz-1) / static_cast<float>(up_edge_z - low_edge_z);

	  isForeignHostPtr = true;
	  isHostAlloced = true;
	  isHostValid = true;
	  isDeviceAlloced = false;
	  isDeviceValid = false;

	  // no need for this as we are using a foreign pointer
	  //allocateHost(); 
#ifndef HEMI_CUDA_DISABLE
	  allocateDevice();
	  isDeviceAlloced = true;
	  copyHostToDevice();
#endif
	  normalized_coords = false;
	}
      
      ~Table3D()
	{
	  deallocateDevice();
	  deallocateHost();
	}
      
      const hemi::table3<T>* readOnlyDevicePtr()
      {
	/* try copying the struct to device too */
#ifndef HEMI_CUDA_DISABLE
	assert(isDeviceAlloced);
	if (!isDeviceValid)
	  copyHostToDevice();
#endif
	
	return d_table;
      }

      const hemi::table3<T>* readOnlyHostPtr()
      {
	return &h_table;
      }

      T* writeOnlyHostPtr()
      {
	if (!isHostAlloced) allocateHost();
	isDeviceValid = false;
	isHostValid   = true;
	return h_table.hPtr;
      }
      
    private:
      
#ifndef HEMI_CUDA_DISABLE
      cudaArray_t dPtr;
      cudaExtent volumeSize;
#endif
      table3<T> h_table;
      table3<T> *d_table;

      bool            isForeignHostPtr;

      mutable bool    isHostAlloced;
      mutable bool    isDeviceAlloced;        
      
      mutable bool    isHostValid;
      mutable bool    isDeviceValid;

      bool normalized_coords;
      
    protected:
      void allocateHost() const
      {
	assert(!isHostAlloced);
	
	h_table.hPtr = new T[h_table.size[0] * h_table.size[1] * h_table.size[2]];    
	
	isHostAlloced = true;
	isHostValid = false;
      }
      
      void allocateDevice()
      {
#ifndef HEMI_CUDA_DISABLE
	assert(!isDeviceAlloced);
	volumeSize = make_cudaExtent(h_table.size[0], h_table.size[1], h_table.size[2]);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
	checkCuda( cudaMalloc3DArray(&dPtr, &channelDesc, volumeSize) );

	// allocate a table struct
	checkCuda( cudaMalloc((void**)&d_table, sizeof(hemi::table3<T>)) );

	isDeviceAlloced = true;
	isDeviceValid = false;
#endif
      }
      
      void deallocateHost()
      {
	if (isHostAlloced) {
	  if (!isForeignHostPtr)
	    delete [] h_table.hPtr;

	  isHostAlloced = false;
	  isHostValid   = false;
	}
      }
      
      void deallocateDevice()
      {
#ifndef HEMI_CUDA_DISABLE
	if (isDeviceAlloced) {
	  checkCuda( cudaFreeArray(dPtr) );
	  checkCuda( cudaDestroyTextureObject(h_table.texture) );
	  checkCuda( cudaFree(d_table) );

	  isDeviceAlloced = false;
	  isDeviceValid   = false;
	}
#endif
      }
      
      void copyHostToDevice()
      {
#ifndef HEMI_CUDA_DISABLE
	assert(isHostAlloced);
	if (!isDeviceAlloced) allocateDevice();

	cudaMemcpy3DParms copyParams = {0};
        copyParams.dstArray = dPtr;
	copyParams.srcPtr   = make_cudaPitchedPtr((void*)h_table.hPtr, volumeSize.width * sizeof(T), volumeSize.width, volumeSize.height);
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyHostToDevice;
	checkCuda( cudaMemcpy3D(&copyParams) );
	
	// bind to texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.linear.devPtr = dPtr;
	resDesc.res.linear.sizeInBytes = volumeSize.width * volumeSize.height * volumeSize.depth * sizeof(T);
	
	// I hope there is a more elegant way to do this
	if (std::is_same<T, float>::value || std::is_same<T, double>::value)
	  resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	else if (std::is_same<T, int>::value)
	  resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
	else if (std::is_same<T, unsigned int>::value)
	  resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
	else
	  {
	    std::cerr << "Warning, template typename doesn't match any acceptable formats for cudaResourceDesc. Setting to \"cudaChannelFormatKindNone\"" << std::endl;
	    resDesc.res.linear.desc.f = cudaChannelFormatKindNone;
	  }

	resDesc.res.linear.desc.x = sizeof(T) * CHAR_BIT;
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.normalizedCoords = normalized_coords;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.filterMode = cudaFilterModeLinear;
	
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaAddressModeClamp;
	
	checkCuda( cudaCreateTextureObject(&h_table.texture, &resDesc, &texDesc, NULL) );

	isDeviceValid = true;
	
	// copy the struct to device
	checkCuda( cudaMemcpy(d_table, &h_table, sizeof(hemi::table3<T>), cudaMemcpyHostToDevice) );
#endif
      }
    };
}

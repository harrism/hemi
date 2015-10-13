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

#ifndef HEMI_CUDA_DISABLE
#include <cuda_runtime_api.h>
#else
struct dim3 {
    unsigned x;
    unsigned y;
    unsigned z;

    dim3(unsigned x = 0, unsigned y = 0, unsigned z = 0) :
        x{x},
        y{y},
        z{z}
    {}
};
#endif

#include "hemi/hemi.h"

bool equals(dim3 a, dim3 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

namespace hemi {

class ExecutionPolicy {
public:
    enum ConfigurationState {
        Automatic = 0,
        SharedMem = 1,
        BlockSize = 2,
        GridSize = 4,
        FullManual = GridSize | BlockSize | SharedMem
    };

    ExecutionPolicy(unsigned dimensions)
    : mDimensions(dimensions),
      mState(Automatic),
      mGrid(0, 0, 0),
      mBlock(0, 0, 0),
      mSharedMemBytes(0),
      mStream((hemiStream_t)0) {}
    
    ExecutionPolicy(dim3 grid, dim3 block, size_t sharedMemBytes)
    : mDimensions(computeDimensions(grid, block)),
      mState(0), mStream(0) {
      setGrid(grid);
      setBlock(block);
      setSharedMemBytes(sharedMemBytes);  
    }

    ExecutionPolicy(dim3 grid, dim3 block, size_t sharedMemBytes, hemiStream_t stream)
    : mDimensions(computeDimensions(grid, block)),
      mState(0) {
      setGrid(grid),
      setBlock(block),
      setSharedMemBytes(sharedMemBytes);
      setStream(stream);
    }
          
    ~ExecutionPolicy() {}

    int    getConfigState()    const { return mState;          }
    
    dim3   getDimensions()     const { return mDimensions;     }
    dim3   getGrid()           const { return mGrid;           }
    dim3   getBlock()          const { return mBlock;          }
    int    getGridSize()       const { return mGrid.x * mGrid.y * mGrid.z;    }
    int    getBlockSize()      const { return mBlock.x * mBlock.y * mBlock.z; }
    int    getMaxBlockSize()   const { return mMaxBlockSize;   }
    size_t getSharedMemBytes() const { return mSharedMemBytes; }
    hemiStream_t getStream()   const { return mStream; }
 
    void setGridSize(int arg) { 
        setGrid(dim3(arg, 1, 1));
    }
    void setBlockSize(int arg) {
        setBlock(dim3(arg, 1, 1));
    }
    void setGrid(dim3 arg) {
        mGrid = arg;
        if (!equals(mGrid, dim3(0, 0, 0))) mState |= GridSize;
        else mState &= (FullManual - GridSize);
    }   
    void setBlock(dim3 arg) {
        mBlock = arg;
        if (!equals(mBlock, dim3(0, 0, 0))) mState |= BlockSize;
        else mState &= (FullManual - BlockSize);
    }
    void setMaxBlockSize(int arg) {
    	mMaxBlockSize = arg;
    }
    void setSharedMemBytes(size_t arg) { 
        mSharedMemBytes = arg; 
        mState |= SharedMem; 
    }
    void setStream(hemiStream_t stream) {
        mStream = stream;
    }

private:
    static unsigned computeDimensions(dim3 gridSize, dim3 blockSize)
    {
        if (gridSize.z > 1 || blockSize.z > 1)
            return 3;
        if (gridSize.y > 1 || blockSize.y > 1)
            return 2;
        if (gridSize.x > 1 || blockSize.x > 1)
            return 1;
        return 0;
    }

    unsigned mDimensions;
    int    mState;
    dim3   mGrid;
    dim3   mBlock;
    int    mMaxBlockSize;
    size_t mSharedMemBytes;
    hemiStream_t mStream;
};

}

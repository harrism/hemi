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

    ExecutionPolicy() 
    : mState(Automatic), 
      mGridSize(0), 
      mBlockSize(0), 
      mSharedMemBytes(0),
      mStream((hemiStream_t)0) {}
    
    ExecutionPolicy(int gridSize, int blockSize, size_t sharedMemBytes)
    : mState(0), mStream(0) {
      setGridSize(gridSize);
      setBlockSize(blockSize);
      setSharedMemBytes(sharedMemBytes);  
    }

    ExecutionPolicy(int gridSize, int blockSize, size_t sharedMemBytes, hemiStream_t stream)
    : mState(0) {
      setGridSize(gridSize);
      setBlockSize(blockSize);
      setSharedMemBytes(sharedMemBytes);
      setStream(stream);
    }
          
    ~ExecutionPolicy() {}

    int    getConfigState()    const { return mState;          }
    
    int    getGridSize()       const { return mGridSize;       }
    int    getBlockSize()      const { return mBlockSize;      }
    int    getMaxBlockSize()   const { return mMaxBlockSize;   }
    size_t getSharedMemBytes() const { return mSharedMemBytes; }
    hemiStream_t getStream()   const { return mStream; }
 
    void setGridSize(int arg) { 
        mGridSize = arg;  
        if (mGridSize > 0) mState |= GridSize; 
        else mState &= (FullManual - GridSize);
    }   
    void setBlockSize(int arg) { mBlockSize = arg; 
        if (mBlockSize > 0) mState |= BlockSize; 
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
    int    mState;
    int    mGridSize;
    int    mBlockSize;
    int    mMaxBlockSize;
    size_t mSharedMemBytes;
    hemiStream_t mStream;
};

}

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
      mSharedMemBytes(0) {}
    
    ExecutionPolicy(int gridSize, int blockSize, size_t sharedMemBytes)
    : mState(0) {
      setGridSize(gridSize);
      setBlockSize(blockSize);
      setSharedMemBytes(sharedMemBytes);  
    }
          
    ~ExecutionPolicy() {}

    int    getConfigState()    const { return mState;          }
    
    int    getGridSize()       const { return mGridSize;       }
    int    getBlockSize()      const { return mBlockSize;      }
    int    getMaxBlockSize()   const { return mMaxBlockSize;   }
    size_t getSharedMemBytes() const { return mSharedMemBytes; }
 
    void setGridSize(int arg) { 
        mGridSize = arg;  
        if (mGridSize > 0) mState |= GridSize; 
    }   
    void setBlockSize(int arg) { mBlockSize = arg; 
        if (mBlockSize > 0) mState |= BlockSize; 
    }
    void setMaxBlockSize(int arg) {
    	mMaxBlockSize = arg;
    }
    void setSharedMemBytes(size_t arg) { 
        mSharedMemBytes = arg; 
        mState |= SharedMem; 
    }

private:
    int    mState;
    int    mGridSize;
    int    mBlockSize;
    int    mMaxBlockSize;
    size_t mSharedMemBytes;
};

}

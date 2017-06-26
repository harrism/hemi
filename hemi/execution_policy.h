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

struct ExecutionGrid
{
	int x;
	int y;
	int z;

	ExecutionGrid()
		: x(1),
		y(1),
		z(1) {}

	ExecutionGrid(int inX)
	{
		x = inX > 0 ? inX : 1;
		y = 1;
		z = 1;
	}

	ExecutionGrid(int inX, int inY)
	{
		x = inX > 0 ? inX : 1;
		y = inY > 0 ? inY : 1;
		z = 1;
	}

	ExecutionGrid(int inX, int inY, int inZ)
	{
		x = inX > 0 ? inX : 1;
		y = inY > 0 ? inY : 1;
		z = inZ > 0 ? inZ : 1;
	}

#ifdef HEMI_CUDA_COMPILER
	dim3 toDim3() const
	{
		return dim3(x, y, z);
	}
#endif
};

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
	  mTarget(device),
      mStream((hemiStream_t)0) {}

    ExecutionPolicy(int gridSize, int blockSize, size_t sharedMemBytes)
    : mState(0), mStream(0), mTarget(device) {
      setGridSize(gridSize);
      setBlockSize(blockSize);
      setSharedMemBytes(sharedMemBytes);
    }

    ExecutionPolicy(int gridSize, int blockSize, size_t sharedMemBytes, hemiStream_t stream)
    : mState(0), mTarget(device) {
      setGridSize(gridSize);
      setBlockSize(blockSize);
      setSharedMemBytes(sharedMemBytes);
      setStream(stream);
    }

	ExecutionPolicy(Location loc)
		: mState(Automatic),
		mSharedMemBytes(0),
		mStream((hemiStream_t)0) {
		setLocation(loc);
	}

	ExecutionPolicy(Location loc, ExecutionGrid gridSize, ExecutionGrid blockSize, size_t sharedMemBytes)
		: mState(0), mStream(0) {
		setLocation(loc);
		setGridSize(gridSize);
		setBlockSize(blockSize);
		setSharedMemBytes(sharedMemBytes);
	}

	ExecutionPolicy(Location loc, ExecutionGrid gridSize, ExecutionGrid blockSize, size_t sharedMemBytes, hemiStream_t stream)
		: mState(0) {
		setLocation(loc);
		setGridSize(gridSize);
		setBlockSize(blockSize);
		setSharedMemBytes(sharedMemBytes);
		setStream(stream);
	}

	ExecutionPolicy(Location loc, ExecutionGrid gridSize, ExecutionGrid blockSize, size_t sharedMemBytes, hemiStream_t stream, int numSessions)
		: mState(0) {
		setLocation(loc);
		setGridSize(gridSize);
		setBlockSize(blockSize);
		setSharedMemBytes(sharedMemBytes);
		setStream(stream);
	}

    ~ExecutionPolicy() {}

    int    getConfigState()    const { return mState;          }

	Location    getLocation()  const { return mTarget; }
	int getGridSize()          const { return mGridSize.x * mGridSize.y * mGridSize.z; }
	int getBlockSize()         const { return mBlockSize.x * mBlockSize.y * mBlockSize.z; }
    int    getMaxBlockSize()   const { return mMaxBlockSize;   }
    size_t getSharedMemBytes() const { return mSharedMemBytes; }
    hemiStream_t getStream()   const { return mStream; }
#ifdef HEMI_CUDA_COMPILER
	dim3 getExecutionBlock()   const { return mBlockSize.toDim3(); }
	dim3 getExecutionGrid()    const { return mGridSize.toDim3(); }
#endif

    void setGridSize(int arg) {
		mGridSize = ExecutionGrid(arg);
        if (arg > 0) mState |= GridSize;
        else mState &= (FullManual - GridSize);
    }
    void setBlockSize(int arg) {
		mBlockSize = ExecutionGrid(arg);
        if (arg > 0) mState |= BlockSize;
        else mState &= (FullManual - BlockSize);
    }
	void setGridSize(ExecutionGrid arg) {
		mGridSize = arg;
		mState |= GridSize;
	}
	void setBlockSize(ExecutionGrid arg) {
		mBlockSize = arg;
		mState |= BlockSize;
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
	void setLocation(Location NewTarget) {
		mTarget = NewTarget;
	}

private:
    int				 mState;
	Location		 mTarget;
	ExecutionGrid    mGridSize;
	ExecutionGrid    mBlockSize;
    int				 mMaxBlockSize;
    size_t			 mSharedMemBytes;
    hemiStream_t	 mStream;
};

}

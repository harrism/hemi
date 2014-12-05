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
#include "kernel.h"

#ifdef HEMI_CUDA_COMPILER
#include "configure.h"
#endif


namespace hemi {

// Automatic Parallel Launch
    template <typename Function, typename... Arguments>
    void Launch(Function f, Arguments... args)
    {
#ifdef HEMI_CUDA_COMPILER
        ExecutionPolicy p;
        checkCuda(configureGrid(p, Kernel<Function, Arguments...>));
        Kernel <<<p.getGridSize(), p.getBlockSize(), p.getSharedMemBytes() >>>(f, args...);
#else
        Kernel(f, args...);
#endif
    }


// Launch with an explicit execution policy / configuration
//template <typename ExecutionPolicy, typename Function, typename... Arguments>
//void Launch(const ExecutionPolicy &p, Function f, Arguments... args);

}

//#include "launch.inl"
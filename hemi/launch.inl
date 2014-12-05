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

#include "kernel.h"

#ifdef HEMI_CUDA_COMPILER
#include "configure.h"
#endif

namespace hemi {
//
// Automatic Launch functions for closures (functor or lambda)
//
template <typename Function, typename... Arguments>
void Launch(Function f, Arguments... args)
{
#ifdef HEMI_CUDA_COMPILER
    ExecutionPolicy p;
    checkCuda(configureGrid(Kernel<Function, Arguments...>, p));
    Kernel<<<p.getGridSize(), p.getBlockSize(), p.getSharedMemBytes()>>>(f, args...);
#else
    Kernel(f, args...);
#endif
}

//
// Launch with explicit configuration
//
/*template <typename Function, typename... Arguments>
void Launch(const ExecutionPolicy &policy, Function f, Arguments... args)
{
#ifdef HEMI_CUDA_COMPILER
    ExecutionPolicy p = policy;
    configureGrid(Kernel<Function>, policy);
    Kernel<<<p.getGridSize(), p.getBlockSize(), p.getSharedMemBytes()>>>(f, args...);
#else
    Kernel(f, args...);
#endif
}

//
// Automatic launch functions for __global__ kernel function pointers: CUDA only
//
#ifdef HEMI_CUDA_COMPILER

template <typename... Arguments>
void Launch(void (*f)(Arguments...), Arguments... args)
{
    ExecutionPolicy p;
    configureGrid(f, p);
    f<<<p.getGridSize(), p.getBlockSize(), p.getSharedMemBytes()>>>(args...);
}

//
// Launch __global__ kernel function with explicit configuration
//
/*template <typename... Arguments>
void Launch(const ExecutionPolicy &policy, void (*f)(Arguments...), Arguments... args)
{
    ExecutionPolicy p = policy;
    configureGrid(f, p);
    f<<<p.getGridSize(), p.getBlockSize(), p.getSharedMemBytes()>>>(args...);
}

#endif // HEMI_CUDA_COMPILER
*/
} // namespace hemi

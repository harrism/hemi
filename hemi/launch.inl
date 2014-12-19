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
void launch(Function f, Arguments... args)
{
#ifdef HEMI_CUDA_COMPILER
    ExecutionPolicy p;
    launch(p, f, args...);
#else
    Kernel(f, args...);
#endif
}

//
// Launch with explicit (or partial) configuration
//
template <typename Function, typename... Arguments>
void launch(const ExecutionPolicy &policy, Function f, Arguments... args)
{
#ifdef HEMI_CUDA_COMPILER
    ExecutionPolicy p = policy;
    checkCuda(configureGrid(policy, Kernel<Function>));
    Kernel<<<p.getGridSize(), p.getBlockSize(), p.getSharedMemBytes()>>>(f, args...);
#else
    Kernel(f, args...);
#endif
}

//
// Automatic launch functions for __global__ kernel function pointers: CUDA only
//

template <typename... Arguments>
void cudaLaunch(void(*f)(Arguments... args), Arguments... args)
{
#ifdef HEMI_CUDA_COMPILER
    ExecutionPolicy p;
    cudaLaunch(p, f, args...);
#else
    f(args...);
#endif
}

//
// Launch __global__ kernel function with explicit configuration
//
template <typename... Arguments>
void cudaLaunch(const ExecutionPolicy &policy, void (*f)(Arguments...), Arguments... args)
{
#ifdef HEMI_CUDA_COMPILER
    ExecutionPolicy p = policy;
    checkCuda(configureGrid(p, f));
    f<<<p.getGridSize(), p.getBlockSize(), p.getSharedMemBytes()>>>(args...);
#else
    f(args...);
#endif
}

} // namespace hemi

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

#include "kernel.h"
#include "device_runtime.h"
#include "execution_policy.h"

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
// Automatic parallel launch
//
template <typename Function, typename... Arguments>
void launch(Arguments... args)
{
	Function f;
	launch(f, args...);
}

//
// Launch with explicit (or partial) configuration
//
template <typename Function, typename... Arguments>
#ifdef HEMI_CUDA_COMPILER
void launch(const ExecutionPolicy &policy, Function f, Arguments... args)
{
	ExecutionPolicy p = policy;
	if (p.getLocation() == hemi::device && queryForDevice())
	{
		checkCuda(configureGrid(p, Kernel<Function, Arguments...>));
		Kernel << <p.getExecutionGrid(),
			p.getExecutionBlock(),
			p.getSharedMemBytes(),
			p.getStream() >> >(f, args...);
	}
	else
	{
		f(args...);
	}
}
#else
void launch(const ExecutionPolicy&, Function f, Arguments... args)
{
    Kernel(f, args...);
}
#endif

//
// Launch with explicit (or partial) configuration and one launch bounds
//
template <int MaxThreadsPerBlock, typename Function, typename... Arguments>
#ifdef HEMI_CUDA_COMPILER
void launch(const ExecutionPolicy &policy, Function f, Arguments... args)
{
	ExecutionPolicy p = policy;
	if (p.getLocation() == hemi::device && queryForDevice())
	{
		checkCuda(configureGrid(p, Kernel<MaxThreadsPerBlock, Function, Arguments...>));
		Kernel<MaxThreadsPerBlock> << <p.getExecutionGrid(),
			p.getExecutionBlock(),
			p.getSharedMemBytes(),
			p.getStream() >> >(f, args...);
	}
	else
	{
		f(args...);
	}
}
#else
void launch(const ExecutionPolicy&, Function f, Arguments... args)
{
    Kernel(f, args...);
}
#endif

//
// Launch with explicit (or partial) configuration and two launch bounds
//
template <int MaxThreadsPerBlock, int MinBlocksPerMultiprocessor, typename Function, typename... Arguments>
#ifdef HEMI_CUDA_COMPILER
void launch(const ExecutionPolicy &policy, Function f, Arguments... args)
{
	ExecutionPolicy p = policy;
	if (p.getLocation() == hemi::device && queryForDevice())
	{
		checkCuda(configureGrid(p, Kernel<MaxThreadsPerBlock, MinBlocksPerMultiprocessor, Function, Arguments...>));
		Kernel<MaxThreadsPerBlock, MinBlocksPerMultiprocessor> << <p.getExecutionGrid(),
			p.getExecutionBlock(),
			p.getSharedMemBytes(),
			p.getStream() >> >(f, args...);
	}
	else
	{
		f(args...);
	}
}
#else
void launch(const ExecutionPolicy&, Function f, Arguments... args)
{
    Kernel(f, args...);
}
#endif

//
// Launch function with an explicit execution policy / configuration with one launch bounds
//
template <int MaxThreadsPerBlock, typename Function, typename... Arguments>
void launch(const ExecutionPolicy &p, Arguments... args)
{
	Function f;
	launch<MaxThreadsPerBlock>(p, f, args...);
}

//
// Launch function with an explicit execution policy / configuration with two launch bounds
//
template <int MaxThreadsPerBlock, int MinBlocksPerMultiprocessor, typename Function, typename... Arguments>
void launch(const ExecutionPolicy &p, Arguments... args)
{
	Function f;
  launch<MaxThreadsPerBlock, MinBlocksPerMultiprocessor>(p, f, args...);
}

//
// Launch function with an explicit execution policy / configuration
//
template <typename Function, typename... Arguments>
void launch(const ExecutionPolicy &p, Arguments... args)
{
	Function f;
	launch(p, f, args...);
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
#ifdef HEMI_CUDA_COMPILER
void cudaLaunch(const ExecutionPolicy &policy, void (*f)(Arguments...), Arguments... args)
{
    ExecutionPolicy p = policy;
    checkCuda(configureGrid(p, f));
    f<<<p.getGridSize(),
        p.getBlockSize(),
        p.getSharedMemBytes(),
        p.getStream()>>>(args...);
}
#else
void cudaLaunch(const ExecutionPolicy&, void (*f)(Arguments...), Arguments... args)
{
    f(args...);
}
#endif

} // namespace hemi

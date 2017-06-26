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

namespace hemi {

    class ExecutionPolicy; // forward decl

    // Automatic parallel launch for function object
    template <typename Function, typename... Arguments>
    void launch(Function f, Arguments... args);

  	// Automatic parallel launch
  	template <typename Function, typename... Arguments>
  	void launch(Arguments... args);

    // Launch function object with an explicit execution policy / configuration
    template <typename Function, typename... Arguments>
    void launch(const ExecutionPolicy &p, Function f, Arguments... args);

  	// Launch function with an explicit execution policy / configuration
  	template <typename Function, typename... Arguments>
  	void launch(const ExecutionPolicy &p, Arguments... args);

    // Launch function object with an explicit execution policy / configuration and one launch bounds
    template <int MaxThreadsPerBlock, typename Function, typename... Arguments>
    void launch(const ExecutionPolicy &p, Function f, Arguments... args);

    // Launch function object with an explicit execution policy / configuration and two launch bounds
    template <int MaxThreadsPerBlock, int MinBlocksPerMultiprocessor, typename Function, typename... Arguments>
    void launch(const ExecutionPolicy &p, Function f, Arguments... args);

    // Launch function with an explicit execution policy / configuration and one lanuch bounds
  	template <int MaxThreadsPerBlock, typename Function, typename... Arguments>
  	void launch(const ExecutionPolicy &p, Arguments... args);

    // Launch function with an explicit execution policy / configuration and two lanuch bounds
  	template <int MaxThreadsPerBlock, int MinBlocksPerMultiprocessor, typename Function, typename... Arguments>
  	void launch(const ExecutionPolicy &p, Arguments... args);

    // Automatic parallel launch for CUDA __global__ functions
    template <typename... Arguments>
    void cudaLaunch(void(*f)(Arguments...), Arguments... args);

    // Launch __global__ function with an explicit execution policy / configuration
    template <typename... Arguments>
    void cudaLaunch(const ExecutionPolicy &p, void(*f)(Arguments...), Arguments... args);
}

#include "launch.inl"

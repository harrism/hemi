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

#include <assert.h>
#include <stdlib.h>
#include "hemi.h"

namespace hemi {

template <typename Function, typename... Arguments>
HEMI_LAUNCHABLE
void Kernel(Function f, Arguments... args)
{
    f(args...);
}

}
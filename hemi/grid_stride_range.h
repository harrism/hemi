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

/////////////////////////////////////////////////////////////////
// Some utility code to define grid_stride_range
#include "range/range.hpp"
#include "hemi.h"
#include "device_api.h"

using namespace util::lang;

// type alias to simplify typing...
using step_range = typename range_proxy<int>::step_range_proxy;

namespace hemi {

	HEMI_DEV_CALLABLE_INLINE
	step_range grid_stride_range(int begin, int end) {
	    begin += hemi::globalThreadIndex();
	    return range(begin, end).step(hemi::globalThreadCount());
	}

	HEMI_DEV_CALLABLE_INLINE
	step_range block_stride_range(int begin, int end) {
	    begin += hemi::localThreadIndex();
	    return range(begin, end).step(hemi::localThreadCount());
	}	
	
}
/////////////////////////////////////////////////////////////////

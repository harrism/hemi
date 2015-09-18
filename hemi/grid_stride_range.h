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
template<typename T>
using step_range = typename range_proxy<T>::step_range_proxy;

namespace hemi {

	template <typename T>
	HEMI_DEV_CALLABLE_INLINE
	step_range<T> grid_stride_range(T begin, T end) {
	    begin += hemi::globalThreadIndex();
	    return range(begin, end).step(hemi::globalThreadCount());
	}
	
}
/////////////////////////////////////////////////////////////////
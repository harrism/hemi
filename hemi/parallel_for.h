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

#include "hemi.h"
#include "launch.h"
#include "configure.h"
#include "grid_stride_range.h"

// TODO, add range-based version
// TODO, possibly add custom termination condition?

namespace hemi 
{
	class ExecutionPolicy; // forward decl

	template <typename index_type, typename F>
	ExecutionPolicy parallel_for(index_type first, index_type last, F function) {
		ExecutionPolicy p;
		return parallel_for(p, first, last, function);
	}

	template <typename index_type, typename F>
	ExecutionPolicy parallel_for(const ExecutionPolicy &p, index_type first, index_type last, F function) {
		return hemi::launch(p, [=] HEMI_LAMBDA () {
			for (auto idx : grid_stride_range(first, last)) function(idx);
		});
	}

	template <typename F>
	ExecutionPolicy parallel_for(size_t first, size_t last, F function) {
		ExecutionPolicy p;
		return parallel_for(p, first, last, function);
	}

	template <typename F>
	ExecutionPolicy parallel_for(const ExecutionPolicy &p, size_t first, size_t last, F function) {
		return hemi::launch(p, [=] HEMI_LAMBDA () {
			for (auto idx : grid_stride_range(first, last)) function(idx);
		});
	}
}

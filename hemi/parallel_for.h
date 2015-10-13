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

// TODO, add versions with execution policy
// TODO, add range-based version
// TODO, possibly add custom termination condition?

namespace hemi 
{
	template <unsigned Dims, typename T = int>
	struct index {
		T values[Dims];

		index(std::initializer_list<T> values_)
		{
			std::copy(values_.begin(), values_.end(), std::begin(values));
		}

		bool operator==(index const& other) const
		{
		    return std::equal(std::begin(values), std::end(values), std::begin(other.values));
		}

		bool operator!=(index const& other) const
		{
		    return !(*this == other);
		}
	};

	template <typename T>
	constexpr unsigned get_index_dims(const T &idx)
	{
		return 1;
	}

	template <unsigned Dims, typename T>
	constexpr unsigned get_index_dims(const index<Dims, T> &idx)
	{
		return Dims;
	}

	class ExecutionPolicy; // forward decl

	using index1d = index<1>;
	using index2d = index<2>;
	using index3d = index<3>;

	template <typename index_type, typename F>
	void parallel_for(const index_type &first, const index_type &last, F function) {
		ExecutionPolicy p(get_index_dims(first));
		parallel_for(p, first, last, function);
	}

	template <typename index_type, typename F>
	void parallel_for(const ExecutionPolicy &p, index_type first, index_type last, F function) {
		hemi::launch(p, [=] HEMI_LAMBDA () {
			for (auto idx : grid_stride_range<0>(first, last)) function(idx);
		});
	}

	template <typename index_type, typename F>
	void parallel_for(const ExecutionPolicy &p,
					  const index<3, index_type> &first,
					  const index<3, index_type> &last,
					  F function) {
		hemi::launch(p, [=] HEMI_LAMBDA () {
#ifdef HEMI_DEBUG
			printf("{%d, %d, %d} -> {%d, %d, %d}\n", first.values[2], first.values[1], first.values[0], last.values[2], last.values[1], last.values[0]);
#endif
			for (auto idx_i : grid_stride_range<2>(first.values[2], last.values[2])) {
				for (auto idx_j : grid_stride_range<1>(first.values[1], last.values[1])) {
					for (auto idx_k : grid_stride_range<0>(first.values[0], last.values[0])) {
						function(idx_i, idx_j, idx_k);
					}
				}
			}
		});
	}

    template <typename index_type, typename F>
	void parallel_for(const ExecutionPolicy &p,
					  const index<2, index_type> &first,
					  const index<2, index_type> &last,
					  F function) {
		hemi::launch(p, [=] HEMI_LAMBDA () {
#ifdef HEMI_DEBUG
			printf("{%d, %d} -> {%d, %d}\n", first.values[1], first.values[0], last.values[1], last.values[0]);
#endif
			for (auto idx_i : grid_stride_range<1>(first.values[1], last.values[1])) {
				for (auto idx_j : grid_stride_range<0>(first.values[0], last.values[0])) {
					function(idx_i, idx_j);
				}
			}
		});
	}

    template <typename index_type, typename F>
	void parallel_for(const ExecutionPolicy &p,
					  const index<1, index_type> &first,
					  const index<1, index_type> &last,
					  F function) {
		hemi::launch(p, [=] HEMI_LAMBDA () {
#ifdef HEMI_DEBUG
			printf("{%d} -> {%d}\n", first.values[0], last.values[0]);
#endif
			for (auto idx_i : grid_stride_range<0>(first.values[0], last.values[0])) {
				function(idx_i);
			}
		});
	}

}

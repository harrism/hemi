#ifndef __HEMI_PARALLEL_FOR_H__
#define __HEMI_PARALLEL_FOR_H__

#include "hemi.h"
#include "launch.h"
#include "grid_stride_range.h"

// TODO, add versions with execution policy
// TODO, add range-based version
// TODO, possibly add custom termination condition?

namespace hemi 
{
	template <typename index_type, typename F>
	void parallel_for(index_type first, index_type last, F function) {
		hemi::launch([=] HEMI_LAMBDA () {
			for (auto idx : grid_stride_range(first, last)) function(idx);
		});
	}
}

#endif // __HEMI_PARALLEL_FOR_H__
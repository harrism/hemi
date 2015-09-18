#include "gtest/gtest.h"
#include "hemi/launch.h"

HEMI_MEM_DEVICE int result;
HEMI_MEM_DEVICE int rGDim;
HEMI_MEM_DEVICE int rBDim;

template <typename T, typename... Arguments>
HEMI_DEV_CALLABLE
T first(T f, Arguments...) {
	return f;
}

template <typename... Arguments>
struct k {
	HEMI_DEV_CALLABLE_MEMBER void operator()(Arguments... args) {
		result = first(args...); //sizeof...(args); 
		rGDim = 1;//gridDim.x;
		rBDim = 1;//blockDim.x;			
	}
};

TEST(PortableLaunchTest, KernelFunction_AutoConfig) {
	k<int> kernel;
	hemi::launch(kernel, 1);
}

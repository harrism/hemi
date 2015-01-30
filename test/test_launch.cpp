#include "gtest/gtest.h"
#include "hemi/launch.h"

HEMI_MEM_DEVICE int result;
HEMI_MEM_DEVICE int rGDim;
HEMI_MEM_DEVICE int rBDim;

template <typename... Arguments>
struct k {
	HEMI_DEV_CALLABLE_MEMBER void operator()(Arguments... args) {
		result = sizeof...(args); 
		rGDim = 1;//gridDim.x;
		rBDim = 1;//blockDim.x;			
	}
};

TEST(LaunchTest, KernelFunction_AutoConfig) {
	k<int> kernel;
	hemi::launch(kernel, 1);
}

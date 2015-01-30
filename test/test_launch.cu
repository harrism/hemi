#include "gtest/gtest.h"
#include "hemi/launch.h"

struct k1 {
	template <typename... Arguments>
	HEMI_DEV_CALLABLE_MEMBER void operator()(int *count, Arguments... args) {
		*count = sizeof...(args); 
	}
};

TEST(LaunchTest, CorrectVariadicParams) {
	int *dCount;
	int count;
	cudaMalloc(&dCount, sizeof(int));

	k1 kern;
	hemi::launch(kern, dCount, 1);
	cudaMemcpy(&count, dCount, sizeof(int), cudaMemcpyDefault);
	ASSERT_EQ(count, 1);

	hemi::launch(kern, dCount, 1, 2);
	cudaMemcpy(&count, dCount, sizeof(int), cudaMemcpyDefault);
	ASSERT_EQ(count, 2);

	hemi::launch(kern, dCount, 1, 2, 'a', 4.0, "hello");
	cudaMemcpy(&count, dCount, sizeof(int), cudaMemcpyDefault);
	ASSERT_EQ(count, 5);
	
}

struct k2 {
	template <typename... Arguments>
	HEMI_DEV_CALLABLE_MEMBER void operator()(int *bdim, int *gdim, Arguments... args) {
		*bdim = blockDim.x;
		*gdim = gridDim.x;
	}
};


TEST(LaunchTest, AutoConfigMaximalLaunch) {
	int *dBdim, *dGdim;
	int bdim, gdim;
	cudaMalloc(&dBdim, sizeof(int));
	cudaMalloc(&dGdim, sizeof(int));

	k2 kern;
	hemi::launch(kern, dBdim, dGdim);
	cudaMemcpy(&bdim, dBdim, sizeof(int), cudaMemcpyDefault);
	cudaMemcpy(&gdim, dGdim, sizeof(int), cudaMemcpyDefault);

	int devId;
	cudaGetDevice(&devId);
	int smCount;
	cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, devId);
	ASSERT_GE(gdim, smCount);
	ASSERT_EQ(gdim%smCount, 0);
	ASSERT_GE(bdim, 1);
}

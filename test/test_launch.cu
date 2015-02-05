#include "gtest/gtest.h"
#include "hemi/launch.h"

inline void ASSERT_SUCCESS(cudaError_t res) {
	ASSERT_EQ(cudaSuccess, res);
}

struct Kernel {
	template <typename... Arguments>
	HEMI_DEV_CALLABLE_MEMBER void operator()(int *count, int *bdim, int *gdim, Arguments... args) {
		*count = sizeof...(args); 
		*bdim = blockDim.x;
		*gdim = gridDim.x;
	}
};

class LaunchTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
  	cudaMalloc(&dCount, sizeof(int));
    cudaMalloc(&dBdim, sizeof(int));
	cudaMalloc(&dGdim, sizeof(int));

	int devId;
	cudaGetDevice(&devId);
	cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, devId);
  }

  virtual void TearDown() {
  	cudaFree(dCount);
  	cudaFree(dBdim);
  	cudaFree(dGdim);
  }	

  Kernel kernel;
  int smCount;
  
  int *dCount;

  int *dBdim;
  int *dGdim;
  
  int count;
  
  int bdim;
  int gdim;
};


TEST_F(LaunchTest, CorrectVariadicParams) {
	hemi::launch(kernel, dCount, dBdim, dGdim, 1);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&count, dCount, sizeof(int), cudaMemcpyDefault));
	ASSERT_EQ(count, 1);

	hemi::launch(kernel, dCount, dBdim, dGdim, 1, 2);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&count, dCount, sizeof(int), cudaMemcpyDefault));
	ASSERT_EQ(count, 2);

	hemi::launch(kernel, dCount, dBdim, dGdim, 1, 2, 'a', 4.0, "hello");
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&count, dCount, sizeof(int), cudaMemcpyDefault));
	ASSERT_EQ(count, 5);
}

TEST_F(LaunchTest, AutoConfigMaximalLaunch) {
	hemi::launch(kernel, dCount, dBdim, dGdim);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&bdim, dBdim, sizeof(int), cudaMemcpyDefault));
	ASSERT_SUCCESS(cudaMemcpy(&gdim, dGdim, sizeof(int), cudaMemcpyDefault));

	ASSERT_GE(gdim, smCount);
	ASSERT_EQ(gdim%smCount, 0);
	ASSERT_GE(bdim, 1);
}

TEST_F(LaunchTest, ExplicitBlockSize) 
{
	hemi::ExecutionPolicy ep;
	ep.setBlockSize(128);
	hemi::launch(ep, kernel, dCount, dBdim, dGdim);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&bdim, dBdim, sizeof(int), cudaMemcpyDefault));
	ASSERT_SUCCESS(cudaMemcpy(&gdim, dGdim, sizeof(int), cudaMemcpyDefault));

	ASSERT_GE(gdim, smCount);
	ASSERT_EQ(gdim%smCount, 0);
	ASSERT_EQ(bdim, 128);	
}


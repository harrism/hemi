#include "hemi_test.h"
#include "hemi/launch.h"

#define ASSERT_SUCCESS(res) ASSERT_EQ(cudaSuccess, (res));
#define ASSERT_FAILURE(res) ASSERT_NE(cudaSuccess, (res));

// for testing hemi::launch()
struct Kernel {
	template <typename... Arguments>
	HEMI_DEV_CALLABLE_MEMBER void operator()(int *count, int *bdim, int *gdim, Arguments... args) {
		*count = sizeof...(args); 
		*bdim = blockDim.x;
		*gdim = gridDim.x;
	}
};

// for testing hemi::cudaLaunch()
template <typename... Arguments>
HEMI_LAUNCHABLE void KernelFunc(int *count, int *bdim, int *gdim, Arguments... args) {
	Kernel k;
	k(count, bdim, gdim, args...);
}

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
	ASSERT_GE(bdim, 32);
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

TEST_F(LaunchTest, ExplicitGridSize) 
{
	hemi::ExecutionPolicy ep;
	ep.setGridSize(100);
	hemi::launch(ep, kernel, dCount, dBdim, dGdim);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&bdim, dBdim, sizeof(int), cudaMemcpyDefault));
	ASSERT_SUCCESS(cudaMemcpy(&gdim, dGdim, sizeof(int), cudaMemcpyDefault));

	ASSERT_EQ(gdim, 100);
	ASSERT_GE(bdim, 32);	
}

TEST_F(LaunchTest, InvalidConfigShouldFail)
{
	// Fail due to block size too large
	hemi::ExecutionPolicy ep;
	ep.setBlockSize(10000);
	hemi::launch(ep, kernel, dCount, dBdim, dGdim);
	ASSERT_FAILURE(checkCudaErrors());

	// Fail due to excessive shared memory size
	ep.setBlockSize(0);
	ep.setGridSize(0);
	ep.setSharedMemBytes(1000000);
	hemi::launch(ep, kernel, dCount, dBdim, dGdim);
	ASSERT_FAILURE(checkCudaErrors());
}

TEST_F(LaunchTest, CorrectVariadicParams_cudaLaunch) {
	hemi::cudaLaunch(KernelFunc, dCount, dBdim, dGdim, 1);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&count, dCount, sizeof(int), cudaMemcpyDefault));
	ASSERT_EQ(count, 1);

	hemi::cudaLaunch(KernelFunc, dCount, dBdim, dGdim, 1, 2);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&count, dCount, sizeof(int), cudaMemcpyDefault));
	ASSERT_EQ(count, 2);

	hemi::cudaLaunch(KernelFunc, dCount, dBdim, dGdim, 1, 2, 'a', 4.0, "hello");
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&count, dCount, sizeof(int), cudaMemcpyDefault));
	ASSERT_EQ(count, 5);
}

TEST_F(LaunchTest, AutoConfigMaximalLaunch_cudaLaunch) {
	hemi::cudaLaunch(KernelFunc, dCount, dBdim, dGdim);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&bdim, dBdim, sizeof(int), cudaMemcpyDefault));
	ASSERT_SUCCESS(cudaMemcpy(&gdim, dGdim, sizeof(int), cudaMemcpyDefault));

	ASSERT_GE(gdim, smCount);
	ASSERT_EQ(gdim%smCount, 0);
	ASSERT_GE(bdim, 32);
}

TEST_F(LaunchTest, ExplicitBlockSize_cudaLaunch) 
{
	hemi::ExecutionPolicy ep;
	ep.setBlockSize(128);
	hemi::cudaLaunch(ep, KernelFunc, dCount, dBdim, dGdim);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&bdim, dBdim, sizeof(int), cudaMemcpyDefault));
	ASSERT_SUCCESS(cudaMemcpy(&gdim, dGdim, sizeof(int), cudaMemcpyDefault));

	ASSERT_GE(gdim, smCount);
	ASSERT_EQ(gdim%smCount, 0);
	ASSERT_EQ(bdim, 128);	
}

TEST_F(LaunchTest, ExplicitGridSize_cudaLaunch) 
{
	hemi::ExecutionPolicy ep;
	ep.setGridSize(100);
	hemi::cudaLaunch(ep, KernelFunc, dCount, dBdim, dGdim);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&bdim, dBdim, sizeof(int), cudaMemcpyDefault));
	ASSERT_SUCCESS(cudaMemcpy(&gdim, dGdim, sizeof(int), cudaMemcpyDefault));

	ASSERT_EQ(gdim, 100);
	ASSERT_GE(bdim, 32);	
}

TEST_F(LaunchTest, InvalidConfigShouldFail_cudaLaunch)
{
	// Fail due to block size too large
	hemi::ExecutionPolicy ep;
	ep.setBlockSize(10000);
	hemi::cudaLaunch(ep, KernelFunc, dCount, dBdim, dGdim);
	ASSERT_FAILURE(checkCudaErrors());

	// Fail due to excessive shared memory size
	ep.setBlockSize(0);
	ep.setGridSize(0);
	ep.setSharedMemBytes(1000000);
	hemi::cudaLaunch(ep, KernelFunc, dCount, dBdim, dGdim);
	ASSERT_FAILURE(checkCudaErrors());
}
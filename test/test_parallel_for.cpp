#include "hemi_test.h"
#include "hemi/parallel_for.h"
#include <cuda_runtime_api.h>

// Separate function because __device__ lambda can't be declared
// inside a private member function, and TEST() defines TestBody()
// private to the test class
void runParallelFor(int *count, int *gdim, int *bdim) 
{
	hemi::parallel_for(0, 100, [=] HEMI_LAMBDA (int) {
		*gdim = hemi::globalBlockCount();
		*bdim = hemi::localThreadCount();

#ifdef HEMI_DEV_CODE
        atomicAdd(count, 1);
#else
        (*count)++;
#endif
	});
}

void runParallelForEP(const hemi::ExecutionPolicy &ep, int *count, int *gdim, int *bdim) 
{
	hemi::parallel_for(ep, 0, 100, [=] HEMI_LAMBDA (int) {
		*gdim = hemi::globalBlockCount();
		*bdim = hemi::localThreadCount();

#ifdef HEMI_DEV_CODE
        atomicAdd(count, 1);
#else
        (*count)++;
#endif
	});
}

class ParallelForTest : public ::testing::Test {
protected:
  virtual void SetUp() {
#ifdef HEMI_CUDA_COMPILER
  	ASSERT_SUCCESS(cudaMalloc(&dCount, sizeof(int)));
    ASSERT_SUCCESS(cudaMalloc(&dBdim, sizeof(int)));
	ASSERT_SUCCESS(cudaMalloc(&dGdim, sizeof(int)));

	int devId;
	ASSERT_SUCCESS(cudaGetDevice(&devId));
	ASSERT_SUCCESS(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, devId));
#else
	dCount = new int;
	dBdim = new int;
	dGdim = new int;

	smCount = 1;
#endif
  }

  virtual void TearDown() {
#ifdef HEMI_CUDA_COMPILER
  	ASSERT_SUCCESS(cudaFree(dCount));
  	ASSERT_SUCCESS(cudaFree(dBdim));
  	ASSERT_SUCCESS(cudaFree(dGdim));
#else
  	delete dCount;
  	delete dBdim;
  	delete dGdim;
#endif
  }	

  void Zero() {
#ifdef HEMI_CUDA_COMPILER
    ASSERT_SUCCESS(cudaMemset(dCount, 0, sizeof(int)));
    ASSERT_SUCCESS(cudaMemset(dBdim, 0, sizeof(int)));
    ASSERT_SUCCESS(cudaMemset(dGdim, 0, sizeof(int)));
#else
    *dCount = 0;
    *dBdim = 0;
    *dGdim = 0;
#endif
    count = 0;
    bdim = 0;
    gdim = 0;
  }

  void CopyBack() {
#ifdef HEMI_CUDA_COMPILER
    ASSERT_SUCCESS(cudaMemcpy(&count, dCount, sizeof(int), cudaMemcpyDefault));
	ASSERT_SUCCESS(cudaMemcpy(&bdim, dBdim, sizeof(int), cudaMemcpyDefault));
	ASSERT_SUCCESS(cudaMemcpy(&gdim, dGdim, sizeof(int), cudaMemcpyDefault));
#else
    count = *dCount;
    bdim = *dBdim;
    gdim = *dGdim;
#endif
  }

  int smCount;
  
  int *dCount;
  int *dBdim;
  int *dGdim;
  
  int count; 
  int bdim;
  int gdim;
};

TEST_F(ParallelForTest, ComputesCorrectSum) {
    Zero();
    runParallelFor(dCount, dGdim, dBdim);
    CopyBack();

    ASSERT_EQ(count, 100);
}


TEST_F(ParallelForTest, AutoConfigMaximalLaunch) {
    Zero();
	runParallelFor(dCount, dGdim, dBdim);
	ASSERT_SUCCESS(cudaDeviceSynchronize());

    CopyBack();

	ASSERT_GE(gdim, smCount);
	ASSERT_EQ(gdim%smCount, 0);
#ifdef HEMI_CUDA_COMPILER
	ASSERT_GE(bdim, 32);
#else
    ASSERT_EQ(bdim, 1);
#endif
}

TEST_F(ParallelForTest, ExplicitBlockSize) 
{
    Zero();
	hemi::ExecutionPolicy ep;
	ep.setBlockSize(128);
	runParallelForEP(ep, dCount, dGdim, dBdim);
	ASSERT_SUCCESS(hemi::deviceSynchronize());

    CopyBack();

	ASSERT_GE(gdim, smCount);
	ASSERT_EQ(gdim%smCount, 0);
#ifdef HEMI_CUDA_COMPILER
	ASSERT_EQ(bdim, 128);	
#else
    ASSERT_EQ(bdim, 1);
#endif
}

TEST_F(ParallelForTest, ExplicitGridSize) 
{
    Zero();
	hemi::ExecutionPolicy ep;
	ep.setGridSize(100);
	runParallelForEP(ep, dCount, dGdim, dBdim);
	ASSERT_SUCCESS(cudaDeviceSynchronize());

    CopyBack();

#ifdef HEMI_CUDA_COMPILER
	ASSERT_EQ(gdim, 100);
	ASSERT_GE(bdim, 32);
#else
    ASSERT_EQ(gdim, 1);
    ASSERT_EQ(bdim, 1);
#endif	
}

TEST_F(ParallelForTest, InvalidConfigShouldFail)
{
    Zero();
	// Fail due to block size too large
	hemi::ExecutionPolicy ep;
	ep.setBlockSize(10000);
	runParallelForEP(ep, dCount, dGdim, dBdim);
#ifdef HEMI_CUDA_COMPILER
	ASSERT_FAILURE(checkCudaErrors());
#endif

	// Fail due to excessive shared memory size
	ep.setBlockSize(0);
	ep.setGridSize(0);
	ep.setSharedMemBytes(1000000);
	runParallelForEP(ep, dCount, dGdim, dBdim);
#ifdef HEMI_CUDA_COMPILER
	ASSERT_FAILURE(checkCudaErrors());
#endif
}
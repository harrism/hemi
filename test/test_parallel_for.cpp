#include "hemi_test.h"
#include "hemi/parallel_for.h"
#include <cuda_runtime_api.h>

// Separate function because __device__ lambda can't be declared
// inside a private member function, and TEST() defines TestBody()
// private to the test class
void runParallelFor(int *result) 
{
	hemi::parallel_for(0, 100, [=] HEMI_LAMBDA (int) {
#ifdef HEMI_DEV_CODE
		atomicAdd(result, 1);
#else
		(*result)++;
#endif
	});
}

TEST(ParallelForTest, ComputesCorrectSum) {

	int hResult = 0;
	int *dResult;
#ifdef HEMI_CUDA_COMPILER
	ASSERT_SUCCESS(cudaMalloc((void**)&dResult, sizeof(int)));
#else
	dResult = new int;
#endif

	ASSERT_SUCCESS(cudaMemcpy(dResult, &hResult, sizeof(int), cudaMemcpyDefault));


	runParallelFor(dResult);

	ASSERT_SUCCESS(cudaMemcpy(&hResult, dResult, sizeof(int), cudaMemcpyDefault));
	ASSERT_EQ(hResult, 100);

#ifdef HEMI_CUDA_COMPILER
	ASSERT_SUCCESS(cudaFree(dResult));
#else
	delete dResult;
#endif
}
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <chrono>

#include "hemi/parallel_for.h"
#include "hemi/array.h"

hemi::ExecutionPolicy saxpy(const hemi::ExecutionPolicy &p, 
	       int n, float a, const float * __restrict__ x, float * __restrict__ y)
{
	return hemi::parallel_for(p, 0, n, [=] HEMI_LAMBDA (int i) {
    	y[i] = a * x[i] + y[i];
    });	
}

int main(int argc, char **argv) {
	int n = 1 << 20; // 4MB
	int numReps = 10;
	if (argc > 1) n = 1 << atoi(argv[1]);
	if (argc > 2) numReps = atoi(argv[2]);
		
	cudaDeviceProp *props = &hemi::DevicePropertiesCache::get();
    
    const int numSMs = props->multiProcessorCount;
    const int maxBlocksPerSM = 16;
    const int maxBlocks = maxBlocksPerSM * numSMs;
    
	const float a = 2.0f;
	
	hemi::Array<float> x(n);
	hemi::Array<float> y(n);

	float *d_x = x.writeOnlyPtr();
	float *d_y = y.writeOnlyPtr();
	
	hemi::parallel_for(0, n, [=] HEMI_LAMBDA (int i) {
		d_x[i] = 1.0f;
		d_y[i] = i / (float)n;
	});

	std::cout << props->name << ": " << numSMs << "x SM " 
		      << props->major << "." << props->minor << " " << n << std::endl;

	for (int numBlocks = 1; numBlocks <= maxBlocks; numBlocks++) {
		hemi::ExecutionPolicy p(numBlocks, 0);
		hemi::ExecutionPolicy pConfigured;
		auto start = std::chrono::high_resolution_clock::now();		
		for (int i = 0; i < numReps; i++)
			pConfigured = saxpy(p, n, a, x.readOnlyPtr(), y.ptr());
		hemi::deviceSynchronize();
		std::chrono::duration<double> gpuTime 
        	= std::chrono::high_resolution_clock::now() - start;
        gpuTime /= numReps;

        std::cout << pConfigured.getGridSize() << " " 
                  << pConfigured.getBlockSize() << " " 
                  << gpuTime.count() << " "
        	      << 3 * n * sizeof(float) * 1e-9f / gpuTime.count() << std::endl;
	}
	   
    return 0;
}
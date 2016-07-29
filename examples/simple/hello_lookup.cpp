#include <stdio.h>

#include "hemi/hemi.h"
#include "hemi/array.h"
#include "hemi/table.h"
#include "hemi/parallel_for.h"

void lookup(const int n, float *val, const hemi::table3<float> *lookupTable, const float x, const float y, const float z)
{
  hemi::parallel_for(0, n, [=] HEMI_LAMBDA (int i) {
      val[i] = lookupTable->lookup(x, y, z);
    });
}

int main(void) 
{
  // init random generator
  std::srand(0);

  // number of lookups to perform
  const int n = 1;

  // lookup table dimensions
  const int nx = 32, ny = 32, nz = 32;

  hemi::Array<float> output(n);
  hemi::Table3D<float> lookup_table(nx, ny, nz,     // dimensions
				    0.0, 0.0, 0.0,  // lower edge
				    1.0, 1.0, 1.0); // upper edge
  
  // fill the table with random numbers between 0 and 1
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x)
        {
	  // host pointer is 1D, but we are storing 3D data there, so use hemi::index to flatten the array indices
	  lookup_table.writeOnlyHostPtr()[hemi::index(x, y, z, nx, ny)] = std::rand() / (float)RAND_MAX;
	}

#ifndef HEMI_CUDA_DISABLE
  // if compiled with CUDA, run on device        
  float *output_array = output.devicePtr();
  const hemi::table3<float> *lookup_table_ptr = lookup_table.readOnlyDevicePtr();
#else
  // else, lets run on CPU
  float *output_array = output.hostPtr();
  const hemi::table3<float> *lookup_table_ptr = lookup_table.readOnlyHostPtr();
#endif
  
  // perform the lookup
  lookup(n, output_array, lookup_table_ptr, 0.5, 0.5, 0.5);
  
  // print out the result
  for (int i = 0; i < n; ++i) {
    printf("%f\n", output.hostPtr()[i]);
  }
  
  return 0;
}

#include <stdio.h>

#include "hemi/hemi.h"
#include "hemi/array.h"
#include "hemi/table.h"
//#include "hemi/launch.h"
#include "hemi/grid_stride_range.h"
#include "hemi/parallel_for.h"
#include "hemi/device_api.h"

void lookup(const int n, float *val, const hemi::table3<float> lookupTable, const float x, const float y, const float z)
{
  hemi::parallel_for(0, n, [=] HEMI_LAMBDA (int i) {
      val[i] = lookupTable.lookup(x, y, z);
    });
}

HEMI_LAUNCHABLE void lookup_kernel(const int n, float *output, const hemi::table3<float> lookupTable, const float *x, const float *y, const float *z)
{
  for (auto i : hemi::grid_stride_range(0, n)) {
    output[i] = lookupTable.lookup(x[i], y[i], z[i]);
  }
}

float lookup_validate(const int n, const hemi::table3<float> lookupTable) 
{
  hemi::Array<float> output_gpu(n);
  hemi::Array<float> output_cpu(n);

  hemi::Array<float> x_val(n);
  hemi::Array<float> y_val(n);
  hemi::Array<float> z_val(n);
  
  for (int i = 0; i < n; ++i) {
    x_val.writeOnlyHostPtr()[i] = (std::rand() / (float)RAND_MAX) * ( (lookupTable.size[0]-1) / (float)(lookupTable.reciprocal_cell_size[0]));
    y_val.writeOnlyHostPtr()[i] = (std::rand() / (float)RAND_MAX) * ( (lookupTable.size[1]-1) / (float)(lookupTable.reciprocal_cell_size[1]));
    z_val.writeOnlyHostPtr()[i] = (std::rand() / (float)RAND_MAX) * ( (lookupTable.size[2]-1) / (float)(lookupTable.reciprocal_cell_size[2]));
  }

  float *dev_ptr = output_gpu.writeOnlyDevicePtr();
  const float *x_ptr = x_val.readOnlyDevicePtr();
  const float *y_ptr = y_val.readOnlyDevicePtr();
  const float *z_ptr = z_val.readOnlyDevicePtr();

  // calculate GPU interpolations
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  /*hemi::parallel_for(0, n, [lookupTable, x_ptr, y_ptr, z_ptr, dev_ptr] HEMI_LAMBDA (int i) {
      dev_ptr[i] = lookupTable.lookup(x_ptr[i], y_ptr[i], z_ptr[i]);
      });*/
  hemi::cudaLaunch(lookup_kernel, n, output_gpu.writeOnlyDevicePtr(), lookupTable, x_ptr, y_ptr, z_ptr);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time for %i lookups on GPU was %f ms\n", n, milliseconds);

  
  // compare to CPU result
  cudaEvent_t start2, stop2;
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);
  cudaEventRecord(start2);
  
  for (int i = 0; i < n; ++i) {
    output_cpu.hostPtr()[i] = lookupTable.lookup(x_val.readOnlyHostPtr()[i],
						 y_val.readOnlyHostPtr()[i],
						 z_val.readOnlyHostPtr()[i]);
  }

  cudaEventRecord(stop2);
  cudaEventSynchronize(stop2);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start2, stop2);
  printf("Time for %i lookups on CPU was %f ms\n", n, milliseconds);
  
  float average = 0;
  for (int i = 0; i < n; ++i) {
    average += (output_gpu.hostPtr()[i] - output_cpu.hostPtr()[i]) / output_cpu.hostPtr()[i];
  }

  average /= (float)n;
  return average;
}

int main(void) {
  std::srand(0);

  const int n = 1000000;
  const int nx = 100, ny = 100, nz = 100;

  hemi::Array<float> output(n);
  hemi::Table3D<float> lookup_table(nx, ny, nz, 
				    0.0, 0.0, 0.0,
				    100.0, 100.0, 100.0);
  
  // lets make some test data                                                                                              
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x)
        lookup_table.writeOnlyHostPtr()[hemi::index(x, y, z, nx, ny)] = std::rand() / (float)RAND_MAX;

  /*  float *pointer;
#ifndef HEMI_CUDA_DISABLE
  pointer = output.devicePtr();
#else
  pointer = output.hostPtr();
  #endif*/
  
  //lookup(n, pointer, lookup_table.readOnlyTable(), 1530.35, 1000.23, 5010.0);
  
  //printf("element 0 = %f\n", output.hostPtr()[0]);

  float precision = lookup_validate(n, lookup_table.readOnlyTable());
  printf("precision %f\n", precision);
  //delete [] table_array;

  return 0;
}

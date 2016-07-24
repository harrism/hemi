#include <stdio.h>

#include "hemi/hemi.h"
#include "hemi/array.h"
#include "hemi/table.h"
//#include "hemi/launch.h"
#include "hemi/parallel_for.h"
#include "hemi/device_api.h"

void lookup(const int n, float *val, const hemi::table3<float> lookupTable, const float x, const float y, const float z)
{
  hemi::parallel_for(0, n, [=] HEMI_LAMBDA (int i) {
      val[i] = lookupTable.lookup(x, y, z);
    });
}

float lookup_validate(const int n, hemi::Array<float> *output,
		      const hemi::table3<float> lookupTable) 
{
  hemi::Array<float> x_val(n);
  hemi::Array<float> y_val(n);
  hemi::Array<float> z_val(n);
  
  //std::srand(0);
  
  for (int i = 0; i < n; ++i) {
    x_val.writeOnlyHostPtr()[i] = (std::rand() / (float)RAND_MAX) * ( (lookupTable.size[0]-1) / (float)(lookupTable.inv_cell_size[0]));
    y_val.writeOnlyHostPtr()[i] = (std::rand() / (float)RAND_MAX) * ( (lookupTable.size[1]-1) / (float)(lookupTable.inv_cell_size[1]));
    z_val.writeOnlyHostPtr()[i] = (std::rand() / (float)RAND_MAX) * ( (lookupTable.size[2]-1) / (float)(lookupTable.inv_cell_size[2]));
    //printf("random sample %i = [%f, %f, %f]\n", i, x_val.readOnlyHostPtr()[i], y_val.readOnlyHostPtr()[i], z_val.readOnlyHostPtr()[i]);
  }

  float *dev_ptr = output->writeOnlyDevicePtr();
  const float *x_ptr = x_val.readOnlyDevicePtr();
  const float *y_ptr = y_val.readOnlyDevicePtr();
  const float *z_ptr = z_val.readOnlyDevicePtr();

  hemi::parallel_for(0, n, [=] HEMI_LAMBDA (int i) {
      dev_ptr[i] = lookupTable.lookup(x_ptr[i], y_ptr[i], z_ptr[i]);
      //if (i == 5000) {
      //printf("GPU %i %f\n", i, dev_ptr[i]);
	//printf("GPU element %i %f\n", 0, lookupTable.getElement(1,0,0));
	// }
    });
  
  for (int i = 0; i < n; ++i) {
    float value = lookupTable.lookup(x_val.readOnlyHostPtr()[i],
				     y_val.readOnlyHostPtr()[i],
				     z_val.readOnlyHostPtr()[i]);
    // if (i == 5000) {
    //printf("CPU %i %f [%f,%f,%f]\n", i, value, x_val.readOnlyHostPtr()[i], y_val.readOnlyHostPtr()[i], z_val.readOnlyHostPtr()[i]);
      //printf("CPU element %i %f\n", 0, lookupTable.getElement(2,5,1));
      // }
    output->hostPtr()[i] -= value;
    output->hostPtr()[i] /= value;
  }

  float average = 0.0f;
  for (int i = 0; i < n; ++i) {
    average += output->hostPtr()[i];
  }

  average /= (float)n;
  return average;
}

int main(void) {
  std::srand(0);

  const int n = 100;
  const int nx = 25, ny = 25, nz = 25;

  float *table_array = new float[nx*ny*nz];
  
  // lets make some test data
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x)
	table_array[hemi::index(x, y, z, nx, ny)] = x;// + y + z;// (std::rand() * 1.0) / (float)RAND_MAX;

  hemi::Array<float> output(n);
  hemi::Table3D<float> lookup_table(table_array, 
				    nx, ny, nz, 
				    0.0, 0.0, 0.0,
				    100.0, 100.0, 100.0);
  
  /*  float *pointer;
#ifndef HEMI_CUDA_DISABLE
  pointer = output.devicePtr();
#else
  pointer = output.hostPtr();
  #endif*/
  
  //lookup(n, pointer, lookup_table.readOnlyTable(), 1530.35, 1000.23, 5010.0);
  
  //printf("element 0 = %f\n", output.hostPtr()[0]);

  float precision = lookup_validate(n, &output, lookup_table.readOnlyTable());
  printf("precision %f\n", precision);
  //delete [] table_array;

  return 0;
}

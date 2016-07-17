#include <stdio.h>
#include "hemi/hemi.h"
#include "hemi/array.h"
#include "hemi/table.h"
#include "hemi/launch.h"
#include "hemi/parallel_for.h"
#include "hemi/device_api.h"

void lookup(const int n, float *val, const hemi::table3D<float> lookupTable, const float x, const float y, const float z)
{
  hemi::parallel_for(0, n, [&] HEMI_LAMBDA (int i) {
      val[i] = lookupTable.lookup(1.0, 1.0, 1.0);
    });
}

inline int index(int x, int y, int z, int dx, int dy)
{
  return x + dx * (y + dy*z);
}

int main(void) {

  const int n = 1000;
  const int nx = 5, ny = 5, nz = 5;

  float *table_array = new float[nx*ny*nz];

  // lets make some test data
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x)
	table_array[hemi::index(x,y,z,nx,ny)] = 0.5f;

  hemi::Array<float> output(n);
  hemi::Table3D<float> lookup_table(nx, ny, nz, table_array);
  
  //hemi::table3D<float> honk;
  //printf("%f", honk.lookup(1,1,1));

  //const hemi::table3D<float> tbl = lookup_table.readOnlyTable();
  //printf("%f/n", tbl.lookup(1.0, 1.0, 1.0));
  
  //printf("%f", lookup_table.readOnlyTable().lookup(1,1,1));
  
  lookup(n, output.hostPtr(), lookup_table.readOnlyTable(), 2.5, 2.5, 2.5);
  printf("element 0 = %f\n", output.hostPtr()[0]);

  //delete [] table_array;

  return 0;
}

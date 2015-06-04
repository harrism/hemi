#include <algorithm>
#include <iostream>
#include <cstdlib>

#include "hemi/parallel_for.h"
#include "hemi/array.h"

void saxpy(int n, float a, const float *x, float *y)
{
	hemi::parallel_for(0, n, [=] HEMI_LAMBDA (int i) {
    	y[i] = a * x[i] + y[i];
    });	
}

int main(void) {
	const int n = 100;

	const float a = 2.0f;
	
	hemi::Array<float> x(n);
	hemi::Array<float> y(n);

	float *d_x = x.writeOnlyPtr();
	float *d_y = y.writeOnlyPtr();
	
	hemi::parallel_for(0, n, [=] HEMI_LAMBDA (int i) {
		d_x[i] = 1.0f;
		d_y[i] = i / (float)n;
	});

    saxpy(n, a, x.readOnlyPtr(), y.ptr());

    const float *h_y = y.readOnlyPtr(hemi::host);
    std::for_each(h_y, h_y+n, [](float v) { std::cout << v << std::endl; });
    
    hemi::deviceSynchronize();
    return 0;
}
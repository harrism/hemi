#include <stdio.h>
#include "hemi/parallel_for.h"

using namespace hemi;

int main(void)
{
	parallel_for(0, 100, [] HEMI_LAMBDA (int i) { 
		printf("%d\n", i); 
	});

	parallel_for(index2d{0, 0}, index2d{10, 10}, [] HEMI_LAMBDA (int i, int j) {
		printf("%d, %d\n", i, j);
	});

	deviceSynchronize();

	return 0;
}


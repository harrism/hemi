#include <stdio.h>
#include "hemi/parallel_for.h"

using namespace hemi;

int main(void)
{
	parallel_for(0, 100, [] HEMI_LAMBDA (int i) { 
		printf("%d\n", i); 
	});

	deviceSynchronize();

	return 0;
}


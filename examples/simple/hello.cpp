#include <stdio.h>
#include "hemi/hemi.h"
#include "hemi/launch.h"

HEMI_KERNEL_FUNCTION(hello) {
    printf("Hello World from thread %d of %d\n",
           hemiGetElementOffset(),
           hemiGetElementStride());
}

int main(void) {

    hello hi;

    hemi::launch(hi);

    checkCuda(cudaDeviceReset());

    hi(); // call on CPU

    return 0;
}
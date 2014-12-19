#include <stdio.h>
#include "hemi/hemi.h"
#include "hemi/launch.h"

__global__ void hello() { 
    printf("Hello World from thread %d of %d\n", 
           hemiGetElementOffset(),
           hemiGetElementStride());
}

int main(void) {
    hemi::cudaLaunch(hello);

    checkCuda(cudaDeviceReset());

    return 0;
}
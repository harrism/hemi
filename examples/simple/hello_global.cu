#include <stdio.h>
#include "hemi/hemi.h"
#include "hemi/launch.h"
#include "hemi/device_api.h"

HEMI_LAUNCHABLE
void hello() { 
    printf("Hello World from thread %d of %d\n", 
           hemi::globalThreadIndex(),
           hemi::globalThreadCount());
}

int main(void) {
    hemi::cudaLaunch(hello);

    hemi::deviceSynchronize(); // make sure print flushes before exit

    return 0;
}
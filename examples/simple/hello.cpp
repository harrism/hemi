#include <stdio.h>
#include "hemi/hemi.h"
#include "hemi/launch.h"
#include "hemi/device_api.h"

HEMI_KERNEL_FUNCTION(hello) {
    printf("Hello World from thread %d of %d\n",
           hemi::globalThreadIndex(),
           hemi::globalThreadCount());
}

int main(void) {

    hello hi;

    hemi::launch(hi);

    hemi::deviceSynchronize(); // make sure print flushes before exit

    hi(); // call on CPU

    return 0;
}
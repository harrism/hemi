#include <stdio.h>
#include "hemi/launch.h"
#include "hemi/device_api.h"

int main(void) {

    hemi::launch([=] HEMI_LAMBDA() {
        printf("Hello World from Lambda in thread %d of %d\n",
            hemi::globalThreadIndex(),
            hemi::globalThreadCount());
    });

    hemi::deviceSynchronize(); // make sure print flushes before exit

    return 0;
}
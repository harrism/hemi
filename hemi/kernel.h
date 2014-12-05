#pragma once 

#include <assert.h>
#include <stdlib.h>
#include "hemi.h"

namespace hemi {

template <typename Function, typename... Arguments>
HEMI_LAUNCHABLE
void Kernel(Function f, Arguments... args)
{
    f(args...);
}

}
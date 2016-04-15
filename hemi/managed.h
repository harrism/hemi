#pragma once
#include "hemi/hemi.h"
#include <cstdlib>
#include <cassert>

namespace hemi {

template <typename T>
T * mallocManaged(size_t len)
{
    T * ptr;
#ifndef HEMI_CUDA_DISABLE
    checkCuda( cudaMallocManaged(&ptr, len * sizeof(T), cudaMemAttachGlobal) );
#else
    ptr = (T *) std::malloc(len * sizeof(T));
#endif
    // It has been reported that after 65536 small allocations, cudaMallocManaged returns
    // cudaSuccess and sets the pointer to NULL
    assert(ptr != nullptr);
    return ptr;
}


void freeManaged(void * ptr)
{
#ifndef HEMI_CUDA_DISABLE
    checkCuda( cudaFree(ptr) );
#else
    std::free(ptr);
#endif    
}


// Allocator to use with any member of the Standard Library.
// e.g. std::vector<float, hemi::ManagedAllocator<float>>
// Note however that while the data in the vector in the above example will be accessible
// from device space, everything else (e.g. std::vector::size()) won't be usable.
// You must extract the pointer to the first element and length while in host code.
template<class T>
class ManagedAllocator
{
public:
    using value_type = T;

    value_type * allocate(size_t n)
    {
        return mallocManaged<T>(n);
    }
  
    void deallocate(value_type * ptr, size_t)
    {
        freeManaged(ptr);
    }
};

}

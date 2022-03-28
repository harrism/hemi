#pragma once

#include <cstdlib>
#include <memory>
#include <hemi/hemi.h>

// Matrices are stored in row-major order (C-layout):
//   M(row, col) = M.host[row * M.stride + col]
template <typename T>
struct Matrix {
    using value_type = T;

    const int rows;
    const int cols;
    const int stride; 
    std::shared_ptr<T[]> host;
    //#ifndef HEMI_CUDA_DISABLE
    //std::shared_ptr<T[], cudaError_t(*)(void*)> device;
    //#else
    std::shared_ptr<T[]> device;
    //#endif

    Matrix(int rows_, int cols_)
        : rows(rows_)
        , cols(cols_)
        , stride( ((cols_+3)/4) * 4 )
    #ifdef HEMI_CUDA_DISABLE
        , host( (T *)aligned_alloc(64, rows*stride*sizeof(T)) )
        , device( host ) { }
    #else
        {
            size_t size = rows*stride*sizeof(T);
            T *x;
            checkCuda( cudaHostAlloc((void**)&x, size, 0) );
            host.reset(x, cudaFreeHost);
            checkCuda( cudaMalloc((void **)&x, size) );
            device.reset(x, cudaFree);
        }
    #endif

    void to_device(size_t nrows = ~0) {
        #ifndef HEMI_CUDA_DISABLE
        if(nrows > rows) {
            nrows = rows;
        }
        checkCuda( cudaMemcpy(device.get(), host.get(), nrows*stride*sizeof(T),
                              cudaMemcpyHostToDevice) );
        #endif
    }
    void to_host(size_t nrows = ~0) {
        #ifndef HEMI_CUDA_DISABLE
        if(nrows > rows) {
            nrows = rows;
        }
        checkCuda( cudaMemcpy(host.get(), device.get(), nrows*stride*sizeof(T),
                              cudaMemcpyDeviceToHost) );
        #endif
    }
};

enum class Place {
    Host, Device
};

template <typename T>
struct KernelMat {
    using value_type = T;

    const int rows;
    const int cols;
    const int stride; 
    const Place location;
    T *const data;

    KernelMat(const Matrix<T> &m, const Place loc_ = Place::Device)
        : rows(m.rows), cols(m.cols), stride(m.stride)
        , location(loc_)
        , data(loc_ == Place::Host ? m.host.get() : m.device.get()) { }

    HEMI_DEV_CALLABLE_INLINE
    KernelMat(int rows_, int cols_, int stride_, T *const x, const Place loc_)
        : rows(rows_), cols(cols_), stride(stride_)
        , location(loc_), data(x) {}

    HEMI_DEV_CALLABLE_INLINE
    T operator()(int row, int col) const {
        return data[row * stride + col];
    }

    HEMI_DEV_CALLABLE_INLINE
    T &operator()(int row, int col) {
        return data[row * stride + col];
    }

    // Get the BLOCK_SIZExBLOCK_SIZE sub-matrix that is
    // located col sub-matrices to the right and row sub-matrices down
    HEMI_DEV_CALLABLE_INLINE
    KernelMat<T> submat(int row, int col, int BLOCK_SIZE) {
        int i = BLOCK_SIZE * row;
        int j = BLOCK_SIZE * col;
        return KernelMat<T>{ i+BLOCK_SIZE <= rows ? BLOCK_SIZE : rows-i
                        , j+BLOCK_SIZE <= cols ? BLOCK_SIZE : cols-j
                        , stride
                        , &data[stride * i + j]
                        , location
                        };
    }
};

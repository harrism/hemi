#pragma once

#include <cstdlib>
#include <memory>
#include <vector>
#include <hemi/hemi.h>

// Matrices are stored in row-major order (C-layout):
//   M(row, col) = M.data[row * M.stride + col]
template <typename T>
struct Matrix {
    using value_type = T;

    const int rows;
    const int cols;
    const int stride; 
    std::shared_ptr<T[]> data;
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
        , data( (T *)aligned_alloc(64, rows*stride*sizeof(T)) )
        , device( data ) { }
    #else
        {
            size_t size = rows*stride*sizeof(T);
            T *x;
            checkCuda( cudaHostAlloc((void**)&x, size, 0) );
            data.reset(x, cudaFreeHost);
            checkCuda( cudaMalloc((void **)&x, size) );
            device.reset(x, cudaFree);
        }
    #endif

    void to_device(size_t nrows = ~0) {
        #ifndef HEMI_CUDA_DISABLE
        if(nrows > rows) {
            nrows = rows;
        }
        checkCuda( cudaMemcpy(device.get(), data.get(), nrows*stride*sizeof(T),
                              cudaMemcpyHostToDevice) );
        #endif
    }
    void to_host(size_t nrows = !0) {
        #ifndef HEMI_CUDA_DISABLE
        if(nrows > rows) {
            nrows = rows;
        }
        checkCuda( cudaMemcpy(data.get(), device.get(), nrows*stride*sizeof(T),
                              cudaMemcpyDeviceToHost) );
        #endif
    }
};

template <typename T>
struct GPUMat {
    using value_type = T;

    const int rows;
    const int cols;
    const int stride; 
    T *const data;

    GPUMat(const Matrix<T> &m) : rows(m.rows), cols(m.cols), stride(m.stride),
                                 data(m.device.get()) { }

    HEMI_DEV_CALLABLE_INLINE
    GPUMat(int rows_, int cols_, int stride_, T *const x)
        : rows(rows_), cols(cols_), stride(stride_), data(x) {}

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
    GPUMat<T> submat(int row, int col, int BLOCK_SIZE) {
        int i = BLOCK_SIZE * row;
        int j = BLOCK_SIZE * col;
        return GPUMat<T>{ i+BLOCK_SIZE <= rows ? BLOCK_SIZE : rows-i
                        , j+BLOCK_SIZE <= cols ? BLOCK_SIZE : cols-j
                        , stride
                        , &data[stride * i + j]
                        };
    }
};

#include <cstdlib>

#include <hemi/launch.h>
#include <hemi/device_api.h>
#include <hemi/execution_policy.h>

#include "mat.hh"

#define BLOCK_SIZE 16

namespace hemi {
    HEMI_DEV_CALLABLE_INLINE
    constexpr unsigned int linSize() {
    #ifdef HEMI_DEV_CODE
        return 1;
    #else
        return BLOCK_SIZE*BLOCK_SIZE;
    #endif
    }
};

// This runs the function on all threads for which
// the current executing context needs to do some work.
template <typename F>
HEMI_DEV_CALLABLE_INLINE
void foreach_thread(F function) {
#ifdef HEMI_DEV_CODE
    const unsigned int idx = hemi::localThreadIndex();
    function(idx, 0);
#else
    for(unsigned int idx=0; idx < hemi::linSize(); idx++)
        function(idx, idx);
#endif
}

// compute one sub-matrix Csub of C
template <typename T>
HEMI_DEV_CALLABLE_INLINE
void MatMulBlk(GPUMat<T> &A, GPUMat<T> &B, GPUMat<T> &C,
               int blockRow, int blockCol) {
    GPUMat<T> Csub = C.submat(blockRow, blockCol, BLOCK_SIZE);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    T Cvalue[hemi::linSize()];
    foreach_thread([&](unsigned int idx, unsigned int lin) {
        Cvalue[lin] = 0;
    });

    for (int m = 0; m < (A.cols+BLOCK_SIZE-1) / BLOCK_SIZE; ++m) {
        GPUMat<T> Asub = A.submat(blockRow, m, BLOCK_SIZE);
        GPUMat<T> Bsub = B.submat(m, blockCol, BLOCK_SIZE);

        HEMI_SHARED T As[BLOCK_SIZE][BLOCK_SIZE];
        HEMI_SHARED T Bs[BLOCK_SIZE][BLOCK_SIZE];

        foreach_thread([&](unsigned int idx, unsigned int lin) {
            const int row = idx / BLOCK_SIZE;
            const int col = idx % BLOCK_SIZE;

            if(row < Asub.rows && col < Asub.cols)
                As[row][col] = Asub(row, col);
            if(row < Bsub.rows && col < Bsub.cols)
                Bs[row][col] = Bsub(row, col);
        });
        hemi::synchronize();

        foreach_thread([&](unsigned int idx, unsigned int lin) {
            const int row = idx / BLOCK_SIZE;
            const int col = idx % BLOCK_SIZE;
            for (int e = 0; e < Asub.cols; ++e)
                Cvalue[lin] += As[row][e] * Bs[e][col];
        });

        hemi::synchronize();
    }

    foreach_thread([&](unsigned int idx, unsigned int lin) {
        const int row = idx / BLOCK_SIZE;
        const int col = idx % BLOCK_SIZE;
        if(row < Csub.rows && col < Csub.cols)
            Csub(row, col) = Cvalue[lin];
    });
}

// GPUMat<T> multiplication kernel called by MatMul()
template <typename T>
struct MatMulKernel {
    HEMI_DEV_CALLABLE_MEMBER void operator()(
                GPUMat<T> A, GPUMat<T> B, GPUMat<T> C) const {
        const int nrows = (A.rows+BLOCK_SIZE-1)/BLOCK_SIZE;
        const int ncols = (B.cols+BLOCK_SIZE-1)/BLOCK_SIZE;
        const int stride = hemi::globalBlockCount();

        for(int i=hemi::globalBlockIndex(); i<nrows*ncols; i+=stride) {
            MatMulBlk(A, B, C, i/ncols, i%ncols);
        }
    }
};

// GPUMat<T> multiplication - Host code
// GPUMat<T> dimensions are assumed to be multiples of BLOCK_SIZE
template <typename T>
void MatMul(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C) {
    GPUMat<T> d_A(A), d_B(B), d_C(C);
    A.to_device();
    B.to_device();

    hemi::ExecutionPolicy ep;
    ep.setBlockSize(BLOCK_SIZE*BLOCK_SIZE);

    MatMulKernel<T> K;
    hemi::launch(ep, K, d_A, d_B, d_C);

    C.to_host();
}

int main(void) {
    int m = 32;
    int n = 64;
    int k = 64;
    Matrix<float> A(m,k), B(k,n), C(m,n);

    MatMul(A, B, C);
    return 0;
}

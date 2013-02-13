///////////////////////////////////////////////////////////////////////////////
// This version does not use Hemi to demonstrate the complexity Hemi saves.
// This is a simple example that performs a Black-Scholes options pricing
// calculation using code that is entirely shared between host (CPU)
// code compiled with any C/C++ compiler (including NVCC) and device code
// that is compiled with the NVIDIA CUDA compiler, NVCC.
// When compiled with "nvcc -x cu" (to force CUDA compilation on the .cpp file),
// this runs on the GPU. When compiled with "nvcc" or "g++" it runs on the host.
///////////////////////////////////////////////////////////////////////////////
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>

#include "timer.h"

#ifdef __CUDACC__
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

// Convenience function for checking CUDA error state including 
// errors caused by asynchronous calls (like kernel launches). Note that
// this causes device synchronization, but is a no-op in release builds.
inline
cudaError_t checkCudaErrors()
{
  cudaError_t result = cudaSuccess;
  checkCuda(result = cudaGetLastError()); // runtime API errors
#if defined(DEBUG) || defined(_DEBUG)
  result = cudaDeviceSynchronize(); // async kernel launch errors
  if (result != cudaSuccess)
    fprintf(stderr, "CUDA Launch Error: %s\n", cudaGetErrorString(result));  
#endif
  return result;
}
#endif


const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

// Polynomial approximation of cumulative normal distribution function
#ifdef __CUDACC__
__host__ __device__
#endif
inline float CND(float d)
{
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float
        K = 1.0f / (1.0f + 0.2316419f * fabsf(d));

    float
        cnd = RSQRT2PI * expf(-0.5f * d * d) * 
        (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if(d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}

// Black-Scholes formula for both call and put
#ifdef __CUDACC__
__global__ void BlackScholes_kernel
#else
void BlackScholes
#endif
    (float *callResult, float *putResult, const float *stockPrice,
     const float *optionStrike, const float *optionYears, float Riskfree,
     float Volatility, int optN)
{
#ifdef __CUDA_ARCH__
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
#else
  int offset = 0;
  int stride = 1;
#endif

    for(int opt = offset; opt < optN; opt += stride)
    {
        float S = stockPrice[opt];
        float X = optionStrike[opt];
        float T = optionYears[opt]; 
        float R = Riskfree;
        float V = Volatility;

        float sqrtT = sqrtf(T);
        float    d1 = (logf(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
        float    d2 = d1 - V * sqrtT;
        float CNDD1 = CND(d1);
        float CNDD2 = CND(d2);

        //Calculate Call and Put simultaneously
        float expRT = expf(- R * T);
        callResult[opt] = S * CNDD1 - X * expRT * CNDD2;
        putResult[opt]  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
    }
}

float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

void initOptions(int n, float *price, float *strike, float *years)
{
   srand(5347);
    //Generate options set
    for (int i = 0; i < n; i++) {
        price[i]  = RandFloat(5.0f, 30.0f);
        strike[i] = RandFloat(1.0f, 100.0f);
        years[i]  = RandFloat(0.25f, 10.0f);
    }
}

int main(int argc, char **argv)
{
    int OPT_N  = 4000000;
    int OPT_SZ = OPT_N * sizeof(float);

    printf("Initializing data...\n");
    
    float *callResult, *putResult, *stockPrice, *optionStrike, *optionYears;    

#ifdef __CUDACC__
    float *d_callResult, *d_putResult;
    float *d_stockPrice, *d_optionStrike, *d_optionYears;

    checkCuda( cudaMallocHost((void**)&callResult,     OPT_SZ) );
    checkCuda( cudaMallocHost((void**)&putResult,      OPT_SZ) );
    checkCuda( cudaMallocHost((void**)&stockPrice,     OPT_SZ) );
    checkCuda( cudaMallocHost((void**)&optionStrike,   OPT_SZ) );
    checkCuda( cudaMallocHost((void**)&optionYears,    OPT_SZ) );
    checkCuda( cudaMalloc    ((void**)&d_callResult,   OPT_SZ) );
    checkCuda( cudaMalloc    ((void**)&d_putResult,    OPT_SZ) );
    checkCuda( cudaMalloc    ((void**)&d_stockPrice,   OPT_SZ) );
    checkCuda( cudaMalloc    ((void**)&d_optionStrike, OPT_SZ) );
    checkCuda( cudaMalloc    ((void**)&d_optionYears,  OPT_SZ) );
#else
    callResult   = (float*)malloc(OPT_SZ);
    putResult    = (float*)malloc(OPT_SZ);
    stockPrice   = (float*)malloc(OPT_SZ);
    optionStrike = (float*)malloc(OPT_SZ);
    optionYears  = (float*)malloc(OPT_SZ);
#endif

    initOptions(OPT_N, stockPrice, optionStrike, optionYears);
       
    int blockDim = 128; // blockDim, gridDim ignored by host code
    int gridDim  = std::min<int>(1024, (OPT_N + blockDim - 1) / blockDim);

    StartTimer();

#ifdef __CUDACC__
    printf("Running Device Version...\n");
    checkCuda( cudaMemcpy(d_stockPrice,   stockPrice,   OPT_SZ, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_optionStrike, optionStrike, OPT_SZ, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_optionYears,  optionYears,  OPT_SZ, cudaMemcpyHostToDevice) );

    BlackScholes_kernel<<<gridDim, blockDim>>>(d_callResult, d_putResult, d_stockPrice, 
                                               d_optionStrike, d_optionYears, RISKFREE, 
                                               VOLATILITY, OPT_N);

    checkCudaErrors();

    checkCuda( cudaMemcpy(callResult, d_callResult, OPT_SZ, cudaMemcpyDeviceToHost) );
    checkCuda( cudaMemcpy(putResult,  d_putResult,  OPT_SZ, cudaMemcpyDeviceToHost) );
#else
    printf("Running Host Version...\n");
    BlackScholes(callResult, putResult, stockPrice, optionStrike, optionYears,
                 RISKFREE, VOLATILITY, OPT_N);
#endif

    printf("Option 0 call: %f\n", callResult[0]); 
    printf("Option 0 put:  %f\n", putResult[0]);

    double ms = GetTimer();

    //Both call and put are calculated
    printf("Options count             : %i     \n", 2 * OPT_N);
    printf("\tBlackScholes() time    : %f msec\n", ms);
    printf("\t%f GB/s, %f GOptions/s\n", 
           ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (ms * 1E-3),
           ((double)(2 * OPT_N) * 1E-9) / (ms * 1E-3));

#ifdef __CUDACC__
    checkCuda( cudaFree(d_stockPrice) );
    checkCuda( cudaFree(d_optionStrike) );
    checkCuda( cudaFree(d_optionYears) );
    checkCuda( cudaFreeHost(callResult) );
    checkCuda( cudaFreeHost(putResult) );
    checkCuda( cudaFreeHost(stockPrice) );
    checkCuda( cudaFreeHost(optionStrike) );
    checkCuda( cudaFreeHost(optionYears) );
#else
    free(callResult);
    free(putResult);
    free(stockPrice);
    free(optionStrike);
    free(optionYears);
#endif 
}

///////////////////////////////////////////////////////////////////////////////
// This is a simple example that performs a Black-Scholes options pricing
// calculation using code that is almost entirely shared between host (CPU)
// code compiled with any C/C++ compiler (including NVCC) and device code
// that is compiled with the NVIDIA CUDA compiler, NVCC. Note there are only
// 25 lines of code in the single .cu file in this example. The majority of 
// the computational code, which is in black_scholes_shared.h, is shared 
// between host and device compilers.
///////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "timer.h"
#include "hemi/hemi.h"
#include "hemi/launch.h"
#include "hemi/grid_stride_range.h"
#include "hemi/hemi_error.h"

const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
HEMI_DEV_CALLABLE_INLINE
float CND(float d)
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

///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
HEMI_KERNEL_FUNCTION(BlackScholes, 
                     float *callResult, float *putResult, float *stockPrice,
                     float *optionStrike, float *optionYears, float riskFree,
                     float volatility, int optN)
{
    for(int opt : hemi::grid_stride_range(0, optN))
    {
        float S = stockPrice[opt];
        float X = optionStrike[opt];
        float T = optionYears[opt]; 
        float R = riskFree;
        float V = volatility;

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

    BlackScholes bs;
               
    printf("Initializing data...\n");

    float *callResult, *putResult, *stockPrice, *optionStrike, *optionYears;
    
    checkCuda( cudaMallocHost((void**)&callResult,     OPT_SZ) );
    checkCuda( cudaMallocHost((void**)&putResult,      OPT_SZ) );
    checkCuda( cudaMallocHost((void**)&stockPrice,     OPT_SZ) );
    checkCuda( cudaMallocHost((void**)&optionStrike,   OPT_SZ) );
    checkCuda( cudaMallocHost((void**)&optionYears,    OPT_SZ) );
    
    initOptions(OPT_N, stockPrice, optionStrike, optionYears);

    printf("Running Host Version...\n");

    StartTimer();
    
    // run BlackScholes operator on host
    bs(callResult, putResult, stockPrice, optionStrike, 
       optionYears, RISKFREE, VOLATILITY, OPT_N);

    printf("Option 0 call: %f\n", callResult[0]); 
    printf("Option 0 put:  %f\n", putResult[0]);

    double ms = GetTimer();

    //Both call and put is calculated
    printf("Options count             : %i     \n", 2 * OPT_N);
       printf("\tBlackScholes() time    : %f msec\n", ms);
    printf("\t%f GB/s, %f GOptions/s\n", 
           ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (ms * 1E-3),
           ((double)(2 * OPT_N) * 1E-9) / (ms * 1E-3));

    float *d_callResult, *d_putResult;
    float *d_stockPrice, *d_optionStrike, *d_optionYears;

    checkCuda( cudaMalloc    ((void**)&d_callResult,   OPT_SZ) );
    checkCuda( cudaMalloc    ((void**)&d_putResult,    OPT_SZ) );
    checkCuda( cudaMalloc    ((void**)&d_stockPrice,   OPT_SZ) );
    checkCuda( cudaMalloc    ((void**)&d_optionStrike, OPT_SZ) );
    checkCuda( cudaMalloc    ((void**)&d_optionYears,  OPT_SZ) );

    printf("Running Device Version...\n");

    StartTimer();
    
    // Launch Black-Scholes operator on device
#ifdef HEMI_CUDA_COMPILER
    cudaMemcpy(d_stockPrice, stockPrice, OPT_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_optionStrike, optionStrike, OPT_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_optionYears, optionYears, OPT_SZ, cudaMemcpyHostToDevice);

    hemi::launch(bs, 
                 d_callResult, d_putResult, d_stockPrice, d_optionStrike, 
                 d_optionYears, RISKFREE, VOLATILITY, OPT_N);

    cudaMemcpy(callResult, d_callResult, OPT_SZ, cudaMemcpyDeviceToHost);
    cudaMemcpy(putResult, d_putResult, OPT_SZ, cudaMemcpyDeviceToHost);
#else // demonstrates that "launch" goes to host when not compiled with NVCC
    hemi::launch(bs, 
                 callResult, putResult, stockPrice, optionStrike, 
                 optionYears, RISKFREE, VOLATILITY, OPT_N);
#endif

    printf("Option 0 call: %f\n", callResult[0]); 
    printf("Option 0 put:  %f\n", putResult[0]);

    ms = GetTimer();

    //Both call and put is calculated
    printf("Options count             : %i     \n", 2 * OPT_N);
       printf("\tBlackScholes() time    : %f msec\n", ms);
    printf("\t%f GB/s, %f GOptions/s\n", 
           ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (ms * 1E-3),
           ((double)(2 * OPT_N) * 1E-9) / (ms * 1E-3));

    checkCuda( cudaFree(d_stockPrice) );
    checkCuda( cudaFree(d_optionStrike) );
    checkCuda( cudaFree(d_optionYears) );
    checkCuda( cudaFreeHost(callResult) );
    checkCuda( cudaFreeHost(putResult) );
    checkCuda( cudaFreeHost(stockPrice) );
    checkCuda( cudaFreeHost(optionStrike) );
    checkCuda( cudaFreeHost(optionYears) );
}

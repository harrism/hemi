///////////////////////////////////////////////////////////////////////////////
// This is a simple example that performs a Black-Scholes options pricing
// calculation using code that is almost entirely shared between host (CPU)
// code compiled with any C/C++ compiler (including NVCC) and device code
// that is compiled with the NVIDIA CUDA compiler, NVCC. Note there are only
// 25 lines of code in the single .cu file in this example. The majority of 
// the computational code, which is in black_scholes_shared.h, is shared 
// between host and device compilers.
///////////////////////////////////////////////////////////////////////////////
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "timer.h"
#include "hemi.h"

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#else
#include <math.h>
#endif

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
HEMI_KERNEL(BlackScholes, (float *callResult, float *putResult, float *stockPrice,
                           float *optionStrike, float *optionYears, float Riskfree,
                           float Volatility, int optN)
{
    int offset = hemiGetElementOffset();
    int stride = hemiGetElementStride();

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
)

float RandFloat(float low, float high){
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

int main(int argc, char **argv)
{
    int OPT_N  = 4000000;
    int OPT_SZ = OPT_N * sizeof(float);
    
    int iterations = 1;
    if (argc >= 2) iterations = atoi(argv[1]);
           
    printf("Initializing data...\n");
    
    float *callResult   = new float[OPT_SZ];
    float *putResult    = new float[OPT_SZ];
    float *stockPrice   = new float[OPT_SZ];
    float *optionStrike = new float[OPT_SZ];
    float *optionYears  = new float[OPT_SZ];
    
    srand(5347);
    //Generate options set
    for(int i = 0; i < OPT_N; i++){
        callResult[i] = 0.0f;
        putResult[i]  = -1.0f;
        stockPrice[i]    = RandFloat(5.0f, 30.0f);
        optionStrike[i]  = RandFloat(1.0f, 100.0f);
        optionYears[i]   = RandFloat(0.25f, 10.0f);
    }

    printf("Running CPU Version %d iterations...\n", iterations);

    StartTimer();
    for (int i = 0; i < iterations; i++)
    {
        BlackScholes(callResult, putResult, stockPrice, optionStrike,
                     optionYears, RISKFREE, VOLATILITY, OPT_N);
    }
    double ms = GetTimer() / iterations;

    //Both call and put is calculated
    printf("Options count             : %i     \n", 2 * OPT_N);
    printf("\tBlackScholes() time    : %f msec\n", ms);
    printf("\t%f GB/s, %f GOptions/s\n", 
           ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (ms * 1E-3),
           ((double)(2 * OPT_N) * 1E-9) / (ms * 1E-3));

#ifdef HEMI_CUDA_COMPILER 
    float *d_callResult, *d_putResult;
    float *d_stockPrice, *d_optionStrike, *d_optionYears;
    cudaMalloc((void**)&d_callResult, OPT_SZ);
    cudaMalloc((void**)&d_putResult, OPT_SZ);
    cudaMalloc((void**)&d_stockPrice, OPT_SZ);
    cudaMalloc((void**)&d_optionStrike, OPT_SZ);
    cudaMalloc((void**)&d_optionYears, OPT_SZ);

    printf("Running GPU Version %d iterations...\n", iterations);

    // Note: this code currently does no checking of CUDA errors
    // This is a bad idea in real code. Omitted here for brevity.
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    
    StartTimer();
    cudaMemcpy(d_stockPrice, stockPrice, OPT_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_optionStrike, optionStrike, OPT_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_optionYears, optionYears, OPT_SZ, cudaMemcpyHostToDevice);

    for (int i = 0; i < iterations; i++)
    {
        int blockDim = 128;
        int gridDim  = props.multiProcessorCount * 16;

        HEMI_KERNEL_LAUNCH(BlackScholes, gridDim, blockDim)
            (d_callResult, d_putResult, d_stockPrice, d_optionStrike, 
             d_optionYears, RISKFREE, VOLATILITY, OPT_N);
    }
   
    cudaMemcpy(callResult, d_callResult, OPT_SZ, cudaMemcpyDeviceToHost);
    cudaMemcpy(putResult, d_putResult, OPT_SZ, cudaMemcpyDeviceToHost);

    ms = GetTimer() / iterations;

    //Both call and put is calculated
    printf("Options count             : %i     \n", 2 * OPT_N);
    printf("\tBlackScholes() time    : %f msec\n", ms);
    printf("\t%f GB/s, %f GOptions/s\n", 
           ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (ms * 1E-3),
           ((double)(2 * OPT_N) * 1E-9) / (ms * 1E-3));

    cudaFree(d_stockPrice);
    cudaFree(d_optionStrike);
    cudaFree(d_optionYears);
#endif // HEMI_CUDA_COMPILER
           
    delete [] callResult;
    delete [] putResult;
    delete [] stockPrice;
    delete [] optionStrike;
    delete [] optionYears;
}

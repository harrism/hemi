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
#include <cuda_runtime_api.h>

#include "timer.h"
#include "blackscholes_shared.h"

const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

// Process an array of optN options on CUDA device
void BlackScholesCuda(float *callResult, float *putResult, float *stockPrice,
                      float *optionStrike, float *optionYears, float Riskfree,
                      float Volatility, int optN, cudaDeviceProp props);

// Process an array of optN options on host
void BlackScholesHost(float *callResult, float *putResult, float *stockPrice,
                      float *optionStrike, float *optionYears, float Riskfree,
                      float Volatility, int   optN)
{
   BlackScholes(callResult, putResult, stockPrice, optionStrike,
                optionYears, Riskfree, Volatility, optN);
}

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
        BlackScholesHost(callResult, putResult, stockPrice, optionStrike,
                         optionYears, RISKFREE, VOLATILITY, OPT_N);
    }
    double ms = GetTimer() / iterations;

    //Both call and put is calculated
    printf("Options count             : %i     \n", 2 * OPT_N);
       printf("\tBlackScholes() time    : %f msec\n", ms);
    printf("\t%f GB/s, %f GOptions/s\n", 
           ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (ms * 1E-3),
           ((double)(2 * OPT_N) * 1E-9) / (ms * 1E-3));

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
        BlackScholesCuda(d_callResult, d_putResult, d_stockPrice, d_optionStrike, 
                         d_optionYears, RISKFREE, VOLATILITY, OPT_N, props);
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
           
    delete [] callResult;
    delete [] putResult;
    delete [] stockPrice;
    delete [] optionStrike;
    delete [] optionYears;
}

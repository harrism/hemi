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

#include "timer.h"
#include "blackscholes_shared.h"

const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

// Process an array of optN options on CUDA device
void BlackScholesLaunch_device(float *callResult, float *putResult, float *stockPrice,
                               float *optionStrike, float *optionYears, float Riskfree,
                               float Volatility, int optN);

void BlackScholesLaunch_host(float *callResult, float *putResult, float *stockPrice,
                            float *optionStrike, float *optionYears, float Riskfree,
                            float Volatility, int optN);

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
    
    checkCuda( cudaMallocHost((void**)&callResult,     OPT_SZ) );
    checkCuda( cudaMallocHost((void**)&putResult,      OPT_SZ) );
    checkCuda( cudaMallocHost((void**)&stockPrice,     OPT_SZ) );
    checkCuda( cudaMallocHost((void**)&optionStrike,   OPT_SZ) );
    checkCuda( cudaMallocHost((void**)&optionYears,    OPT_SZ) );
    
    initOptions(OPT_N, stockPrice, optionStrike, optionYears);

    printf("Running Host Version...\n");

    StartTimer();
    
    BlackScholesLaunch_host(callResult, putResult, stockPrice, optionStrike,
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
    cudaMemcpy(d_stockPrice, stockPrice, OPT_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_optionStrike, optionStrike, OPT_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_optionYears, optionYears, OPT_SZ, cudaMemcpyHostToDevice);

    BlackScholesLaunch_device(d_callResult, d_putResult, d_stockPrice, d_optionStrike, 
                     d_optionYears, RISKFREE, VOLATILITY, OPT_N);
   
    cudaMemcpy(callResult, d_callResult, OPT_SZ, cudaMemcpyDeviceToHost);
    cudaMemcpy(putResult, d_putResult, OPT_SZ, cudaMemcpyDeviceToHost);

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

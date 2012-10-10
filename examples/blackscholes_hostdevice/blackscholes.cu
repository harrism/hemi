#include "blackscholes_shared.h"

// Process an array of optN options on the device
__global__
void BlackScholesKernel(float *callResult, float *putResult, float *stockPrice,
                        float *optionStrike, float *optionYears, float Riskfree,
                        float Volatility, int optN)
{
    BlackScholes(callResult, putResult, stockPrice, optionStrike, optionYears, 
                 Riskfree, Volatility, optN);
}

// Wrapper function that launches the device kernel
void BlackScholesCuda(float *callResult, float *putResult, float *stockPrice,
                      float *optionStrike, float *optionYears, float Riskfree,
                      float Volatility, int optN, cudaDeviceProp props)
{
    int blockDim = 128;
    int gridDim  = props.multiProcessorCount * 16;

    BlackScholesKernel<<<gridDim, blockDim>>>(callResult, putResult, 
                                              stockPrice, optionStrike, 
                                              optionYears, Riskfree, 
                                              Volatility, optN);
}

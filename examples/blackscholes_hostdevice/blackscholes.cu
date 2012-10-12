#include "blackscholes_shared.h"
#include <algorithm>

// Wrapper function that launches the device kernel
void BlackScholesCuda(float *callResult, float *putResult, float *stockPrice,
                      float *optionStrike, float *optionYears, float riskFree,
                      float volatility, int optN)
{
    int blockDim = 128;
    int gridDim  = std::min<int>(1024, (optN + blockDim - 1) / blockDim);
    
    HEMI_KERNEL_LAUNCH(BlackScholes, gridDim, blockDim, 0, 0,
                       callResult, putResult, stockPrice, optionStrike, 
                       optionYears, riskFree, volatility, optN);
}

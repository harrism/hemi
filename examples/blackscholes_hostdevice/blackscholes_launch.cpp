#include "hemi/hemi.h"
#include "hemi/launch.h"
#include "hemi/execution_policy.h"
#include <algorithm>
#include "blackscholes_shared.h"

// avoid duplicate symbols since we are compiling this file twice
#ifdef HEMI_CUDA_COMPILER
#define BlackScholesLaunch BlackScholesLaunch_device
#else
#define BlackScholesLaunch BlackScholesLaunch_host
#endif

// Wrapper function that launches the device kernel
void BlackScholesLaunch(float *callResult, float *putResult, float *stockPrice,
                        float *optionStrike, float *optionYears, float riskFree,
                        float volatility, int optN)
{
    hemi::cudaLaunch(HEMI_KERNEL_NAME(BlackScholes), callResult, putResult, stockPrice, 
                 optionStrike, optionYears, riskFree, volatility, optN);
}

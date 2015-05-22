///////////////////////////////////////////////////////////////////////////////
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
#include "hemi/hemi.h"
#include "hemi/device_api.h"
#include "hemi/parallel_for.h"
#include "hemi/array.h"

const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

// Polynomial approximation of cumulative normal distribution function
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

// Black-Scholes formula for both call and put
void BlackScholes(float *callResult, float *putResult, const float *stockPrice,
     const float *optionStrike, const float *optionYears, float Riskfree,
     float Volatility, int optN)
{
    hemi::parallel_for(0, optN, [=] HEMI_LAMBDA (int opt)
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
    });
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

    printf("Initializing data...\n");

    hemi::Array<float> callResult(  OPT_N, true);
    hemi::Array<float> putResult(   OPT_N, true);
    hemi::Array<float> stockPrice(  OPT_N, true);
    hemi::Array<float> optionStrike(OPT_N, true);
    hemi::Array<float> optionYears( OPT_N, true);

    initOptions(OPT_N, stockPrice.writeOnlyHostPtr(),
                optionStrike.writeOnlyHostPtr(),
                optionYears.writeOnlyHostPtr());
            
    printf("Running %s Version...\n", HEMI_LOC_STRING);

    StartTimer();

    BlackScholes(callResult.writeOnlyPtr(), 
                 putResult.writeOnlyPtr(), 
                 stockPrice.readOnlyPtr(), 
                 optionStrike.readOnlyPtr(),
                 optionYears.readOnlyPtr(), 
                 RISKFREE, VOLATILITY, OPT_N);

    // force copy back to host if needed and print a sanity check
    printf("Option 0 call: %f\n", callResult.readOnlyPtr(hemi::host)[0]); 
    printf("Option 0 put:  %f\n", putResult.readOnlyPtr(hemi::host)[0]);

    double ms = GetTimer();

    //Both call and put is calculated
    printf("Options count             : %i     \n", 2 * OPT_N);
    printf("\tBlackScholes() time    : %f msec\n", ms);
    printf("\t%f GB/s, %f GOptions/s\n", 
           ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (ms * 1E-3),
           ((double)(2 * OPT_N) * 1E-9) / (ms * 1E-3));
}

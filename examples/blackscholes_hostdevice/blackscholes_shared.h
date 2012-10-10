#ifndef __BS_SHARED_H__
#define __BS_SHARED_H__

#include "hemi.h"

#ifndef __CUDACC__
#include <math.h>
#endif

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
HEMI_DEV_CALLABLE_INLINE
void BlackScholes(float *callResult, float *putResult, float *stockPrice,
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

#endif // __BS_SHARED_H__
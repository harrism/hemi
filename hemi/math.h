#ifndef __HEMI_MATH_H__
#define __HEMI_MATH_H__

#include "hemi/hemi.h"
#include <cmath>


namespace hemi {
	// Power functions

	// templated POW function. For a CUDA device, it casts the arguments to double.
	// For a host, it defaults to std::pow in <cmath>
	template <class T>
		HEMI_DEV_CALLABLE_INLINE double pow(T x, T y) {
		#ifdef HEMI_DEV_CODE
			return pow((double)x,(double)y);
		#else
			return std::pow(x, y);
		#endif
	}
		HEMI_DEV_CALLABLE_INLINE float pow(float x, float y) {
		#ifdef HEMI_DEV_CODE
			return powf(x,y);
		#else
			return std::pow(x, y);
		#endif
	}

	// Templated to accept integral types (which effectively casts to double). 
	template <class T>
	HEMI_DEV_CALLABLE_INLINE double sqrt (T x) {
		#ifdef HEMI_DEV_CODE
			return sqrt((double)x);
		#else
			return std::sqrt(x);
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE float sqrt(float x) {
		#ifdef HEMI_DEV_CODE
			return sqrtf(x);
		#else
			return std::sqrt(x);
		#endif
	}

	template <class T>
	HEMI_DEV_CALLABLE_INLINE double cbrt (T x) {
		#ifdef HEMI_DEV_CODE
			return cbrt((double)x);
		#else
			return std::pow(x,(1.0f/3.0f));
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE float cbrt(float x) {
		#ifdef HEMI_DEV_CODE
			return cbrtf(x);
		#else
			return std::pow(x,(1.0f/3.0f));
		#endif
	}
	
	template <class T, class S>
	HEMI_DEV_CALLABLE_INLINE double hypot (T x, S y) {
		#ifdef HEMI_DEV_CODE
			return hypot((double)x, (double)y);
		#else
			return std::sqrt(std::pow(x,2) + std::pow(y,2));
		#endif
	}

	HEMI_DEV_CALLABLE_INLINE float hypot(float x, float y) {
		#ifdef HEMI_DEV_CODE
			return hypotf(x, y);
		#else
			return std::sqrt(std::pow(x,2)+std::pow(y,2));
		#endif
	}


	// Absolute value functions.
	template <class T>
	HEMI_DEV_CALLABLE_INLINE T abs (T x) {
		#ifdef HEMI_DEV_CODE
			return (T)abs((double)x);
		#else
			return std::abs(x);
		#endif
	}
	template <>
	HEMI_DEV_CALLABLE_INLINE int abs<int>(int x) {
		#ifdef HEMI_DEV_CODE
			return __sad(x,0,0);
		#else
			return ((x < 0) ? x * -1 : x);
		#endif
	}	
	template <>
	HEMI_DEV_CALLABLE_INLINE unsigned int abs<unsigned int>(unsigned int x) {
		return x;
	}
	template <>
	HEMI_DEV_CALLABLE_INLINE float abs<float>(float x) {
		#ifdef HEMI_DEV_CODE
			return fabsf(x);
		#else
			return std::abs(x);
		#endif
	}
	template <>
	HEMI_DEV_CALLABLE_INLINE double abs<double>(double x) {
		#ifdef HEMI_DEV_CODE
			return copysign(x, 1.0);
		#else
			return std::abs(x);
		#endif
	}

// Trig functions
	template <class T> // templated to catch int, long, etc. CUDA casts it to double first. 
	HEMI_DEV_CALLABLE_INLINE double acos (T x) {
		#ifdef HEMI_DEV_CODE
			return acos((double)x);
		#else
			return std::acos(x);
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE float acos (float x) {
		#ifdef HEMI_DEV_CODE
			return acosf(x);
		#else
			return std::acos(x);
		#endif
	}
	template <class T>
	HEMI_DEV_CALLABLE_INLINE double asin (T x) {
		#ifdef HEMI_DEV_CODE
			return asin((double)x);
		#else
			return std::asin(x);
		#endif
	}

	HEMI_DEV_CALLABLE_INLINE float asin(float x) {
		#ifdef HEMI_DEV_CODE
			return asinf(x);
		#else
			return std::asin(x);
		#endif
	}


	template <class T>
	HEMI_DEV_CALLABLE_INLINE double atan (T x) {
		#ifdef HEMI_DEV_CODE
			return atan((double)x);
		#else
			return std::atan(x);
		#endif
	}

	HEMI_DEV_CALLABLE_INLINE float atan (float x) {
		#ifdef HEMI_DEV_CODE
			return atanf(x);
		#else
			return std::atan(x);
		#endif
	}

	template <class T>
	HEMI_DEV_CALLABLE_INLINE double atan2 (T x, T y) {
		#ifdef HEMI_DEV_CODE
			return atan2((double)x, (double) y);
		#else
			return std::atan2(x,y);
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE float atan2 (float x, float y) {
		#ifdef HEMI_DEV_CODE
			return atan2f(x,y);
		#else
			return std::atan2(x,y);
		#endif
	}

// Hyperbolic functions


	template <class T>
	HEMI_DEV_CALLABLE_INLINE double acosh (T x) {
		#ifdef HEMI_DEV_CODE
			return acosh((double)x);
		#else
		// since hyperbolic arccosine support isn't widespread, using defintion from Wolfram Alpha
		// which is either acosh(z) = ln(z+sqrt(z+1)*sqrt(z-1)) or acosh(z) = (sqrt(z-1))/sqrt(1-z))*acos(z)
			return std::log(x + std::sqrt(x+1) * std::sqrt(x-1));
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE float acosh(float x) {
		#ifdef HEMI_DEV_CODE
			return acoshf(x);
		#else
		// since hyperbolic arccosine support isn't widespread, using defintion from Wolfram Alpha
		// which is either acosh(z) = ln(z+sqrt(z+1)*sqrt(z-1)) or acosh(z) = (sqrt(z-1))/sqrt(1-z))*acos(z)
			return std::log(x + std::sqrt(x+1) * std::sqrt(x-1));
		#endif
	}
	template <class T>
	HEMI_DEV_CALLABLE_INLINE double asinh (T x) {
		#ifdef HEMI_DEV_CODE
			return asinh((double)x);
		#else
			return std::log(x+sqrt(1+std::pow(x,2)));
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE float asinh(float x) {
		#ifdef HEMI_DEV_CODE
			return asinhf(x);
		#else
			return std::log(x+sqrt(1+std::pow(x,2)));
		#endif
	}

// Rounding/remainder functions
	HEMI_DEV_CALLABLE_INLINE float fmod(float x, float y) {
		#ifdef HEMI_DEV_CODE
			return fmodf(x,y);
		#else
			return std::fmod(x,y);
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE double fmod(double x, double y) {
		#ifdef HEMI_DEV_CODE
			return fmod(x,y);
		#else
			return std::fmod(x,y);
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE float floor(float x) {
		#ifdef HEMI_DEV_CODE
			return floorf(x);
		#else
			return std::floor(x);
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE double floor(double x) {
		#ifdef HEMI_DEV_CODE
			return floor(x);
		#else
			return std::floor(x);
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE float ceil(float x) {
		#ifdef HEMI_DEV_CODE
			return ceilf(x);
		#else
			return std::ceil(x);
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE double ceil(double x) {
		#ifdef HEMI_DEV_CODE
			return ceil(x);
		#else
			return std::ceil(x);
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE double round(double x) {
		#ifdef HEMI_DEV_CODE
			return round(x);
		#else
			return (std::floor(x + 0.5) == std::ceil(x) ? std::ceil(x) : std::floor(x));
		#endif
	}
	template <class T>
	HEMI_DEV_CALLABLE_INLINE double round(T x) {
		#ifdef HEMI_DEV_CODE
			return round((double)x);
		#else
			return (std::floor((double)x + 0.5) == std::ceil(x) ? std::ceil(x) : std::floor(x));
		#endif
	}

	HEMI_DEV_CALLABLE_INLINE float round(float x) {
		#ifdef HEMI_DEV_CODE
			return roundf(x);
		#else
			return (std::floor(x + 0.5f) == std::ceil(x) ? std::ceil(x) : std::floor(x));
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE long int lround(double x) {
		#ifdef HEMI_DEV_CODE
			return lround(x);
		#else
			return (std::floor(x + 0.5) == std::ceil(x) ? (long int)std::ceil(x) : (long int)std::floor(x));
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE long int lround(float x) {
		#ifdef HEMI_DEV_CODE
			return lroundf(x);
		#else
			return (std::floor(x + 0.5f) == std::ceil(x) ? (long int)std::ceil(x) : (long int)std::floor(x));
		#endif
	}
// Exponential/log functions
	template <class T>
	HEMI_DEV_CALLABLE_INLINE double exp(T x) {
		#ifdef HEMI_DEV_CODE
			return exp((double)x);
		#else
			return std::exp(x);
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE float exp(float x) {
		#ifdef HEMI_DEV_CODE
			return expf(x);
		#else
			return std::exp(x);
		#endif
	}
	template <class T>
	HEMI_DEV_CALLABLE_INLINE double log(T x) {
		#ifdef HEMI_DEV_CODE
			return log((double)x);
		#else
			return std::log(x);
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE float log(float x) {
		#ifdef HEMI_DEV_CODE
			return logf(x);
		#else
			return std::log(x);
		#endif
	}
	template <class T>
	HEMI_DEV_CALLABLE_INLINE double log10(T x) {
		#ifdef HEMI_DEV_CODE
			return log10((double)x);
		#else
			return std::log10(x);
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE float log10(float x) {
		#ifdef HEMI_DEV_CODE
			return log10f(x);
		#else
			return std::log10(x);
		#endif
	}
	template <class T>
	HEMI_DEV_CALLABLE_INLINE double frexp(T x, int* ntpr) {
		#ifdef HEMI_DEV_CODE
			return frexp((double)x, ntpr);
		#else
			return std::frexp(x, ntpr);
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE float frexp(float x, int* ntpr) {
		#ifdef HEMI_DEV_CODE
			return frexpf(x, ntpr);
		#else
			return std::frexp(x, ntpr);
		#endif
	}
	template <class T>
	HEMI_DEV_CALLABLE_INLINE double ldexp(T x, int ntpr) {
		#ifdef HEMI_DEV_CODE
			return ldexp((double)x, ntpr);
		#else
			return std::ldexp(x, ntpr);
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE float ldexp(float x, int ntpr) {
		#ifdef HEMI_DEV_CODE
			return ldexpf(x, ntpr);
		#else
			return std::ldexp(x,ntpr);
		#endif
	}

// integer functions
	HEMI_DEV_CALLABLE_INLINE unsigned int brev(unsigned int x) {
		#ifdef HEMI_DEV_CODE
			return __brev(x);
		#else
			unsigned int s = sizeof(x) * CHAR_BIT; // bit size; must be power of 2 
			unsigned int mask = ~0;         
			while ((s >>= 1) > 0) 
			{
				mask ^= (mask << s);
				x = ((x >> s) & mask) | ((x << s) & ~mask);
			}
			return x;
		#endif 
	}

	HEMI_DEV_CALLABLE_INLINE unsigned long long int brev(unsigned long long int x) {
		#ifdef HEMI_DEV_CODE
			return __brevll(x);
		#else
			unsigned int s = sizeof(x) * CHAR_BIT; // bit size; must be power of 2 
			unsigned int mask = ~0;         
			while ((s >>= 1) > 0) 
			{
				mask ^= (mask << s);
				x = ((x >> s) & mask) | ((x << s) & ~mask);
			}
			return x;
		#endif 
	}

	HEMI_DEV_CALLABLE_INLINE unsigned int popc(unsigned int x) {
		#ifdef HEMI_DEV_CODE
			return __popc(x);
		#else
			unsigned int s = sizeof(x) * CHAR_BIT; // bit size; must be power of 2 
			unsigned int count = 0;
			for (s = sizeof(x) * CHAR_BIT; s > 0; s--)
			{
				count += (x & 1) > 0;
				x >>= 1;
			}
			return count;
		#endif
	}
	HEMI_DEV_CALLABLE_INLINE unsigned long long int popc(unsigned long long int x) {
		#ifdef HEMI_DEV_CODE
			return __popcll(x);
		#else
			unsigned int s = sizeof(x) * CHAR_BIT; // bit size; must be power of 2 
			unsigned int count = 0;
			for (s = sizeof(x) * CHAR_BIT; s > 0; s--)
			{
				count += (x & 1) > 0;
				x >>= 1;
			}
			return count;
		#endif
	}

// misc
	//
	template<class T, class S, class R>
	HEMI_DEV_CALLABLE_INLINE double fma(T x, S y, R z) {
		#ifdef HEMI_DEV_CODE
			return fma((double)x, (double)y, (double) z);
		#else
			// this can have issues if fma doesn't exist in <cmath>
			return fma(x,y,z);

		#endif 
	}

	HEMI_DEV_CALLABLE_INLINE float fma(float x, float y, float z) {
		#ifdef HEMI_DEV_CODE
			return fmaf(x,y,z);
		#else
			// this can have issues if fma doesn't exist in <cmath>
			return fma(x,y,z);

		#endif 
	}
}

#endif // HEMI_MATH_H

#ifndef __HEMI_MATH_H__
#define __HEMI_MATH_H__

#include "hemi/hemi.h"
#include <cmath>

namespace hemi {
	// templated POW function. For a CUDA device, it casts the arguments to double.
	// For a host, it defaults to std::pow in <cmath>
	template <class T>
		HEMI_DEV_CALLABLE_INLINE T pow(T x, T y) {
		#ifdef HEMI_DEV_CODE
			return (T)pow((double)x,(double)y);
		#else
			return std::pow(x, y);
		#endif
	}

	// Absolute value functions. There apparently 
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
			return std::abs(x);
		#endif
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
	template <class T>
	HEMI_DEV_CALLABLE_INLINE double sqrt (T x) {
		#ifdef HEMI_DEV_CODE
			return sqrt((double)x);
		#else
			return std::sqrt(x);
		#endif
	}
}

#endif // HEMI_MATH_H

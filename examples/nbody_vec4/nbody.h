#include "hemi/hemi.h"

// Compute gravitational force between two bodies.
// Body mass is stored in w component of the Vec4f.
HEMI_DEV_CALLABLE
Vec4f gravitation(const Vec4f &i, const Vec4f &j);

// Compute the gravitational force induced on "target" body by all 
// masses in the bodies array.
HEMI_DEV_CALLABLE
Vec4f accumulateForce(const Vec4f &target, const Vec4f *bodies, int N);
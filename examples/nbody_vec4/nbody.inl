#include "hemi.h"
#include <stdio.h>

// Softening constant prevents division by zero
HEMI_DEV_CONSTANT
float softeningSquared = 0.01f;

// Compute gravitational force between two bodies.
// Body mass is stored in w component of the Vec4f.
HEMI_DEV_CALLABLE_INLINE
Vec4f gravitation(const Vec4f &i, const Vec4f &j)
{
  Vec4f ij = j - i;
  ij.w = 0;

  float invDist = ij.inverseLength(softeningSquared);
  
  return ij * (j.w * invDist * invDist * invDist);
}

// Compute the gravitational force induced on "target" body by all 
// masses in the bodies array.
HEMI_DEV_CALLABLE_INLINE
Vec4f accumulateForce(const Vec4f &target, const Vec4f *bodies, int N) 
{
  Vec4f force(0, 0, 0, 0);
  for (int j = 0; j < N; j++) {
    force += gravitation(target, bodies[j]);
  }

  return force * target.w;
}
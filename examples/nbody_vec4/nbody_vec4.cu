#include "vec4f.h"
#include "nbody.h"
#include <stdio.h>

// Compute gravitational force between two bodies.
// Body mass is stored in w component of the Vec4f.
HEMI_DEV_CALLABLE
Vec4f gravitation(const Vec4f &i, const Vec4f &j)
{
  Vec4f ij = j - i;
  ij.w = 0;

  float invDist = ij.inverseLength(HEMI_CONSTANT(softeningSquared));
  
  return ij * (j.w * invDist * invDist * invDist);
}

// Compute the gravitational force induced on "target" body by all 
// masses in the bodies array.
HEMI_DEV_CALLABLE
Vec4f accumulateForce(const Vec4f &target, const Vec4f *bodies, int N) 
{
  Vec4f force(0, 0, 0, 0);
  for (int j = 0; j < N; j++) {
    force += gravitation(target, bodies[j]);
  }

  return force * target.w;
}

// Simple CUDA kernel that computes all-pairs n-body gravitational forces.
HEMI_KERNEL(allPairsForces)(Vec4f *forceVectors, const Vec4f *bodies, int N)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  forceVectors[idx] = accumulateForce(bodies[idx], bodies, N);
}

// CUDA kernel that computes all-pairs n-body gravitational forces. Optimized
// version of allPairsForcesKernel that uses shared memory for data reuse.
HEMI_KERNEL(allPairsForcesShared)
  (Vec4f *forceVectors, const Vec4f *bodies, int N)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  
  __shared__ Vec4f jBodies[256];

  Vec4f iBody = bodies[idx];
  Vec4f force = Vec4f(0, 0, 0, 0);
 
  for (int tile = 0; tile < gridDim.x; tile++) 
  {
    jBodies[threadIdx.x] = bodies[tile * blockDim.x + threadIdx.x];

    __syncthreads();
        
    force += accumulateForce(iBody, jBodies, blockDim.x);
      
    __syncthreads();
  }

  forceVectors[idx] = force;
}

// Host wrapper function that launches the CUDA kernels
void allPairsForcesCuda(Vec4f *forceVectors, 
                        const Vec4f *bodies, 
                        int N, bool useShared)
{
  int blockDim = 256;
  int gridDim = (N + blockDim - 1) / blockDim;

  float ss = 0.01f;
  cudaMemcpyToSymbol(HEMI_DEV_CONSTANT(softeningSquared), 
                     &ss, sizeof(float), 0, cudaMemcpyHostToDevice);

  if (useShared)
    HEMI_KERNEL_LAUNCH(allPairsForcesShared, gridDim, blockDim, 0, 0,
                       forceVectors, bodies, N);
  else
    HEMI_KERNEL_LAUNCH(allPairsForces, gridDim, blockDim, 0, 0,
                       forceVectors, bodies, N);
}

// Example of using a host/device class from host code in a .cu file
// Sum the masses of the bodies
Vec4f centerOfMass(const Vec4f *bodies, int N) 
{
  float totalMass = 0.0f;
  Vec4f com = Vec4f(0, 0, 0, 0);
  for (int i = 0; i < N; i++) {
    totalMass += bodies[i].w;
    com += Vec4f(bodies[i].x, bodies[i].y, bodies[i].z, 0) * bodies[i].w;
  }
  com /= totalMass;
  return com;
}
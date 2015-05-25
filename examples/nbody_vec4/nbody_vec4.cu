#define HEMI_DEBUG
#include "hemi/launch.h"
#include "hemi/grid_stride_range.h"
#include "vec4f.h"
#include "nbody.h"
#include <stdio.h>

// Softening constant prevents division by zero
HEMI_DEFINE_CONSTANT(float softeningSquared, 0.01f);

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
HEMI_LAUNCHABLE
void allPairsForces(Vec4f *forceVectors, const Vec4f *bodies, int N)
{
  for (auto idx : hemi::grid_stride_range(0, N))
    forceVectors[idx] = accumulateForce(bodies[idx], bodies, N);
}

// CUDA kernel that computes all-pairs n-body gravitational forces. Optimized
// version of allPairsForcesKernel that uses shared memory for data reuse.
HEMI_LAUNCHABLE
void allPairsForcesShared(Vec4f *forceVectors, const Vec4f *bodies, int N)
{
  extern __shared__ Vec4f jBodies[];

  for (auto idx : hemi::grid_stride_range(0, N))
  {
    Vec4f iBody = bodies[idx];
    Vec4f force = Vec4f(0, 0, 0, 0);
   
    for (auto tile : range((unsigned)0, N / blockDim.x))
    {
      jBodies[threadIdx.x] = bodies[tile * blockDim.x + threadIdx.x];

      __syncthreads();
          
      force += accumulateForce(iBody, jBodies, blockDim.x);
        
      __syncthreads();
    }

    forceVectors[idx] = force;
  }
}

// Host wrapper function that launches the CUDA kernels
void allPairsForcesCuda(Vec4f *forceVectors, 
                        const Vec4f *bodies, 
                        int N, bool useShared)
{
  float ss = 0.01f;
  checkCuda( cudaMemcpyToSymbol(HEMI_DEV_CONSTANT(softeningSquared), 
                                &ss, sizeof(float), 0, cudaMemcpyHostToDevice) );

  if (useShared) {
    // we specify the block size and shared memory size, but grid
    // size is automatically chosen
    const int blockSize = 256;
    hemi::ExecutionPolicy ep;
    ep.setBlockSize(blockSize);
    ep.setSharedMemBytes(blockSize * sizeof(Vec4f));
    hemi::cudaLaunch(ep, allPairsForcesShared, forceVectors, bodies, N);
  }
  else // fully automatic configuration in this case
    hemi::cudaLaunch(allPairsForces, forceVectors, bodies, N);
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
#include "vec4f.h"
#include "nbody.inl"
#include <stdio.h>

// Simple CUDA kernel that computes all-pairs n-body gravitational forces.
__global__
void allPairsForcesKernel(Vec4f *forceVectors, const Vec4f *bodies, int N)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  forceVectors[idx] = accumulateForce(bodies[idx], bodies, N);
}

// CUDA kernel that computes all-pairs n-body gravitational forces. Optimized
// version of allPairsForcesKernel that uses shared memory for data reuse.
__global__
void allPairsForcesKernelShared(Vec4f *forceVectors, const Vec4f *bodies, int N)
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
void allPairsForcesCuda(Vec4f *forceVectors, const Vec4f *bodies, int N, bool useShared)
{
  int blockDim = 256;
  int gridDim = (N + blockDim - 1) / blockDim;

  float ss = 0.01f;
  cudaMemcpyToSymbol(softeningSquared, &ss, sizeof(float), 0, cudaMemcpyHostToDevice);

  if (useShared)
    allPairsForcesKernelShared<<<gridDim, blockDim>>>(forceVectors, bodies, N);
  else
    allPairsForcesKernel<<<gridDim, blockDim>>>(forceVectors, bodies, N);
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
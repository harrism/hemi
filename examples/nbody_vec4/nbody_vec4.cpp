///////////////////////////////////////////////////////////////////////////////
// This example implements a simple all-pairs n-body gravitational force
// calculation using a 4-vector class called Vec4f. Vec4f uses the HEMI 
// Portable CUDA C/C++ Macros to enable all of the code for the class, as well
// as the bulk of the n-body functionality, to be shared between host code
// compiled by the host compiler and device code compiled with the NVIDIA CUDA
// C/C++ compiler, NVCC.
///////////////////////////////////////////////////////////////////////////////
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

#include "vec4f.h"
#include "nbody.h"
#include "timer.h"

extern Vec4f centerOfMass(const Vec4f *bodies, int N);
extern void allPairsForcesCuda(Vec4f *forceVectors, const Vec4f *bodies, int N, bool useShared);

void allPairsForcesHost(Vec4f *forceVectors, const Vec4f *bodies, int N) 
{
  for (int i = 0; i < N; i++)
    forceVectors[i] = accumulateForce(bodies[i], bodies, N);
}

inline float randFloat(float low, float high){
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

void randomizeBodies(Vec4f *bodies, int N)
{
  srand(437893);
  for (int i = 0; i < N; i++) {
    Vec4f &b = bodies[i];
    b.x = randFloat(-1000.f, 1000.f);
    b.y = randFloat(-1000.f, 1000.f);
    b.z = randFloat(-1000.f, 1000.f);
    b.w = randFloat(0.1f, 1000.f);
  }
}

int main(void)
{
  int N = 16384;
  Vec4f targetBody(0.5f, 0.5f, 0.5f, 10.0f);
  Vec4f *bodies = new Vec4f[N];
  Vec4f *forceVectors = new Vec4f[N];

  randomizeBodies(bodies, N);

  // Call a host function defined in a .cu compilation unit
  // that uses host/device shared class member functions
  Vec4f com = centerOfMass(bodies, N);
  printf("Center of mass is (%f, %f, %f)\n", com.x, com.y, com.z);

  // Call host function defined in a .cpp compilation unit
  // that uses host/device shared functions and class member functions
  printf("CPU: Computing all-pairs gravitational forces on %d bodies\n", N);

  StartTimer();
  allPairsForcesHost(forceVectors, bodies, N);
  double ms = GetTimer();

  printf("CPU: Force vector 0: (%0.3f, %0.3f, %0.3f)\n", 
         forceVectors[0].x, forceVectors[0].y, forceVectors[0].z);
  printf("CPU: %f ms\n", ms);

  Vec4f *d_bodies, *d_forceVectors;
  cudaMalloc((void**)&d_bodies, N * sizeof(Vec4f));
  cudaMalloc((void**)&d_forceVectors, N * sizeof(Vec4f));

  StartTimer();
  cudaMemcpy(d_bodies, bodies, N * sizeof(Vec4f), cudaMemcpyHostToDevice);

  // Call device function defined in a .cu compilation unit
  // that uses host/device shared functions and class member functions
  printf("GPU: Computing all-pairs gravitational forces on %d bodies\n", N);
    
  allPairsForcesCuda(d_forceVectors, d_bodies, N, false);
    
  cudaMemcpy(forceVectors, d_forceVectors, N * sizeof(Vec4f), cudaMemcpyDeviceToHost);
  ms = GetTimer();

  printf("GPU: Force vector 0: (%0.3f, %0.3f, %0.3f)\n", 
         forceVectors[0].x, forceVectors[0].y, forceVectors[0].z);

  printf("GPU: %f ms\n", ms);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

  StartTimer();
  cudaMemcpy(d_bodies, bodies, N * sizeof(Vec4f), cudaMemcpyHostToDevice);

  // Call a different device function defined in a .cu compilation unit
  // that uses the same host/device shared functions and class member functions 
  // as above
  printf("GPU: Computing optimized all-pairs gravitational forces on %d bodies\n", N);
    
  allPairsForcesCuda(d_forceVectors, d_bodies, N, true);
    
  cudaMemcpy(forceVectors, d_forceVectors, N * sizeof(Vec4f), cudaMemcpyDeviceToHost);
  ms = GetTimer();
  
  printf("GPU: Force vector 0: (%0.3f, %0.3f, %0.3f)\n", 
         forceVectors[0].x, forceVectors[0].y, forceVectors[0].z);

  printf("GPU+shared: %f ms\n", ms);

  err = cudaGetLastError();
  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
    
  return 0;
}
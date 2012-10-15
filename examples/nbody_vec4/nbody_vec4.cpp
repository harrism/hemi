///////////////////////////////////////////////////////////////////////////////
// This example implements a simple all-pairs n-body gravitational force
// calculation using a 4D vector class called Vec4f. Vec4f uses the HEMI 
// Portable CUDA C/C++ Macros to enable all of the code for the class to be 
// shared between host code compiled by the host compiler and device or host 
// code compiled with the NVIDIA CUDA C/C++ compiler, NVCC. The example
// also shares most of the all-pairs gravitationl force calculation code 
// between device and host, while demonstrating how optimized device 
// implementations can be substituted as needed.
//
// This sample also uses hemi::Array to simplify host/device memory allocation
// and host-device data transfers.
///////////////////////////////////////////////////////////////////////////////
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime_api.h"

#include "vec4f.h"
#include "nbody.h"
#include "timer.h"
#include "hemi/array.h"

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
  hemi::Array<Vec4f> bodies(N, true), forceVectors(N, true);
  
  randomizeBodies(bodies.writeOnlyHostPtr(), N);

  // Call a host function defined in a .cu compilation unit
  // that uses host/device shared class member functions
  Vec4f com = centerOfMass(bodies.readOnlyHostPtr(), N);
  printf("Center of mass is (%f, %f, %f)\n", com.x, com.y, com.z);

  // Call host function defined in a .cpp compilation unit
  // that uses host/device shared functions and class member functions
  printf("CPU: Computing all-pairs gravitational forces on %d bodies\n", N);

  StartTimer();
  allPairsForcesHost(forceVectors.writeOnlyHostPtr(), bodies.readOnlyHostPtr(), N);
  
  printf("CPU: Force vector 0: (%0.3f, %0.3f, %0.3f)\n", 
         forceVectors.readOnlyHostPtr()[0].x, 
         forceVectors.readOnlyHostPtr()[0].y, 
         forceVectors.readOnlyHostPtr()[0].z);

  double ms = GetTimer();

  printf("CPU: %f ms\n", ms);

  StartTimer();

  // Call device function defined in a .cu compilation unit
  // that uses host/device shared functions and class member functions
  printf("GPU: Computing all-pairs gravitational forces on %d bodies\n", N);
    
  allPairsForcesCuda(forceVectors.writeOnlyDevicePtr(), bodies.readOnlyDevicePtr(), N, false);
    
  printf("GPU: Force vector 0: (%0.3f, %0.3f, %0.3f)\n", 
         forceVectors.readOnlyHostPtr()[0].x, 
         forceVectors.readOnlyHostPtr()[0].y, 
         forceVectors.readOnlyHostPtr()[0].z);
  
  ms = GetTimer();

  printf("GPU: %f ms\n", ms);
  
  StartTimer();
  
  // Call a different device function defined in a .cu compilation unit
  // that uses the same host/device shared functions and class member functions 
  // as above
  printf("GPU: Computing optimized all-pairs gravitational forces on %d bodies\n", N);
    
  allPairsForcesCuda(forceVectors.writeOnlyDevicePtr(), bodies.readOnlyDevicePtr(), N, true);
    
  printf("GPU: Force vector 0: (%0.3f, %0.3f, %0.3f)\n", 
         forceVectors.readOnlyHostPtr()[0].x, 
         forceVectors.readOnlyHostPtr()[0].y, 
         forceVectors.readOnlyHostPtr()[0].z);
  
  ms = GetTimer();
  
  printf("GPU+shared: %f ms\n", ms);
  
  return 0;
}
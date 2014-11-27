Hemi: CUDA Portable C/C++ Utilities
===================================

Read about Hemi on the [NVIDIA Parallel Forall Blog](http://devblogs.nvidia.com/parallelforall/developing-portable-cuda-cc-code-hemi/).

[<img align="right" src="https://raw.github.com/harrism/hemi/master/hemi-logo-transparent.png" width="272" height="152"/>](https://raw.github.com/harrism/hemi/master/hemi-logo.png)
CUDA C/C++ and the NVIDIA NVCC compiler toolchain support a number of features designed to make it easier to write portable code, including language integration of host and device code and data, declaration specifiers (e.g. `__host__` and `__device__`) and preprocessor definitions (`__CUDACC__`). These features combine to enable developers to write code that can be compiled and run on either the host, the device, or both. Other compilers don't recognize these features, however, so to really write portable code, we need preprocessor macros. This is where Hemi comes in.

Hemi simplifies writing portable CUDA C/C++ code. In the screenshot below, the code shown on the left is a simple black scholes code written to be compilable with either NVCC or a standard C++ host compiler, and also runnable on either the CPU or GPU. The right column is the same code written using Hemi's macros and smart heterogeneous Array container class, `hemi::Array`. Using Hemi, the length of this code is reduced by half.

![Hemi simplifies portable CUDA C/C++ code](https://raw.github.com/harrism/hemi/master/hemi_simplifies_portable_cuda.png)

Current Version
---------------

This is version: 1.0 (HEMI_VERSION == 100000)

Hemi on github
--------------

The home for Hemi is https://github.com/harrism/hemi, where you can find the latest changes and information.

hemi/hemi.h
-----------

The hemi.h header provides simple macros that are useful for reusing code between CUDA C/C++ and C/C++ written for other platforms (e.g. CPUs). 

The macros are used to decorate function prototypes and variable declarations so that they can be compiled by either NVCC or a host compiler (for example gcc or cl.exe, the MS Visual Studio compiler). 
 
The macros can be used within .cu, .cuh, .cpp, .h, and .inl files to define code that can
be compiled either for the host (e.g., CPU) or the device (e.g., GPU). 

hemi/array.h
------------

One of the biggest challenges in writing portable CUDA code is memory management. HEMI provides the `hemi::Array` C++ template class as a simple data management wrapper which allows arrays of arbitrary type to be created and used with both host and device code. hemi::Array maintains a host and a device pointer for each array. It lazily transfers data between the host and device as needed when the user requests a pointer to the host or device memory. Pointer requests specify read-only, read/write, or write-only options so that valid flags can be maintained and data is only copied when the requested pointer is invalid.

For example, here is an excerpt from the nbody_vec4 example.

    hemi::Array<Vec4f> bodies(N, true);
    hemi::Array<Vec4f> forceVectors(N, true);
  	
    randomizeBodies(bodies.writeOnlyHostPtr(), N);

    // Call host function defined in a .cpp compilation unit
  	allPairsForcesHost(forceVectors.writeOnlyHostPtr(), bodies.readOnlyHostPtr(), N);
  
  	printf("CPU: Force vector 0: (%0.3f, %0.3f, %0.3f)\n", 
           forceVectors.readOnlyHostPtr()[0].x, 
           forceVectors.readOnlyHostPtr()[0].y, 
           forceVectors.readOnlyHostPtr()[0].z);
    
  	...
    
    // Call device function defined in a .cu compilation unit
    // that uses host/device shared functions and class member functions
    allPairsForcesCuda(forceVectors.writeOnlyDevicePtr(), 
                       bodies.readOnlyDevicePtr(), N, false);
    
    printf("GPU: Force vector 0: (%0.3f, %0.3f, %0.3f)\n", 
           forceVectors.readOnlyHostPtr()[0].x, 
           forceVectors.readOnlyHostPtr()[0].y, 
           forceVectors.readOnlyHostPtr()[0].z);

Typical CUDA code requires explicit duplication of host allocations on the device,and explicit copy calls between them. `hemi::Array` allows CUDA code to be used without writing a single `cudaMalloc`, `cudaFree`, or `cudaMemcpy`. `hemi::Array` supports pinned host memory for efficient PCI-express transfers, and handles CUDA error checking internally.

Portable functions
------------------

A common use for host-device code sharing is commonly used utility functions. For example, if we wish to define an inline function to compute the average of two floats that can be called either from host code or device code, and can be compiled by either the host compiler or NVCC, we define it like this:

    HEMI_DEV_CALLABLE_INLINE float avgf(float x, float y) { return (x+y)/2.0f; }

The macro definition ensures that when compiled by NVCC, both a host and device version of the function are generated, and a normal inline function is generated when compiled by the host compiler.

For example use, see the `CND()` function in the "blackscholes" example, as well as several other functions used in the examples.

Portable classes
----------------

The `HEMI_DEV_CALLABLE_MEMBER` and `HEMI_DEV_CALLABLE_INLINE_MEMBER` macros can be used to create classes that are reusable between host and device code, by decorating any member function prototype that will be used by both device and host code. Here is an example excerpt of a portable class (a 4D vector type used in the "nbody_vec4" example).

    struct HEMI_ALIGN(16) Vec4f
    {
      float x, y, z, w;
    
      HEMI_DEV_CALLABLE_INLINE_MEMBER
      Vec4f() {};
    
      HEMI_DEV_CALLABLE_INLINE_MEMBER
      Vec4f(float xx, float yy, float zz, float ww) : x(xx), y(yy), z(zz), w(ww) {}
    
      HEMI_DEV_CALLABLE_INLINE_MEMBER
      Vec4f(const Vec4f& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}

      HEMI_DEV_CALLABLE_INLINE_MEMBER
      Vec4f& operator=(const Vec4f& v) {
        x = v.x; y = v.y; z = v.z; w = v.w;
        return *this;
      }

      HEMI_DEV_CALLABLE_INLINE_MEMBER
      Vec4f operator+(const Vec4f& v) const {
        return Vec4f(x+v.x, y+v.y, z+v.z, w+v.w);
      }
      ...
    };

The `HEMI_ALIGN` macro is used on types that will be passed in arrays or pointers as arguments to CUDA device kernel functions, to ensure proper alignment. `HEMI_ALIGN` generates correct alignment specifiers for the host compilers, too. For details on alignment, see the NVIDIA CUDA C Programming Guide (Section 5.3 in v5.0).

Portable kernels
----------------

`HEMI_KERNEL` can be used to declare a function that is launchable as a CUDA kernel when compiled with NVCC, or as a C/C++ (host) function when compiled with the host compiler. `HEMI_KERNEL_LAUNCH` is a convenience macro that can be used to launch a kernel function on the device when compiled with NVCC, or call the host function when compiled with the host compiler. 

For example, here is an excerpt from the "blackscholes" example.

    // Black-Scholes formula for both call and put
    HEMI_KERNEL(BlackScholes)
        (float *callResult, float *putResult, const float *stockPrice,
         const float *optionStrike, const float *optionYears, float Riskfree,
         float Volatility, int optN)
    {
      ...
    }

    .... in main() ...
    HEMI_KERNEL_LAUNCH(BlackScholes, gridDim, blockDim, 0, 0,
                       d_callResult, d_putResult, d_stockPrice, d_optionStrike, 
                       d_optionYears, RISKFREE, VOLATILITY, OPT_N);

`HEMI_KERNEL_NAME` can also be used to access the generated name of the kernel function, for example to pass a function pointer to CUDA api functions like `cudaFuncGetAttributes()`.

`HEMI_KERNEL_LAUNCH` requires grid and block dimensions to be passed to it, but these parameters are ignored when compiled for the host. When`DEBUG` is defined, `HEMI_KERNEL_LAUNCH` checks for CUDA launch and runtime errors.

Portable Constants
------------------

Global constant values can be defined using the `HEMI_DEFINE_CONSTANT` macro, which takes a name and an initial value. When compiled with NVCC as CUDA code, this declares two versions of the constant, one `__constant__` variable for the device, and one normal host variable. When compiled with a host compiler, only the host variable is defined.

For static or external linkage, use the `HEMI_DEFINE_STATIC_CONSTANT` and `HEMI_DEFINE_EXTERN_CONSTANT` versions of the macro, respectively.

To access variables defined using `HEMI_DEFINE_*_CONSTANT` macros, use the `HEMI_CONSTANT` macro which automatically resolves to either the device or host constant depending on whether it is called from device or host code. This means that the proper variable will chosen when the constant is accessed within functions declared with `HEMI_DEV_CALLABLE_*` and `HEMI_KERNEL` macros.

To explicitly access the device version of a constant, use `HEMI_DEV_CONSTANT`. This is useful when the constant is an argument to a CUDA API function such as `cudaMemcpyToSymbol`, as shown in the following code from the "nbody_vec4" example.

    cudaMemcpyToSymbol(HEMI_DEV_CONSTANT(softeningSquared), 
                       &ss, sizeof(float), 0, cudaMemcpyHostToDevice)

Note: Non-inline functions and methods
--------------------------------------

There are non-inline versions of the macros (`HEMI_DEV_CALLABLE` and `HEMI_DEV_CALLABLE_MEMBER`. Care needs to be taken to avoid multiple definition linker errors due to using these in headers that are included into multiple compilation units. The best way to use `HEMI_DEV_CALLABLE` is to declare functions using this macro in a header, and define their implementation in a .cu file, and compile it with NVCC. This will generate code for both host and device. The host code will be linked into your library or application and callable from other host code compilation units (.c and .cpp files).  

Likewise, for `HEMI_DEV_CALLABLE_MEMBER`, put the class and function declaration in a header, and the member function implementations in a .cu file, compiled by NVCC.

Note: Device-specific code
-----------------------------

Note: Code in functions like this must be portable. In other words it must be able to compile and run for both the host or device. If it is not, within the function you can use `HEMI_DEV_CODE` to define separate code for host and device. Example:

    HEMI_DEV_CALLABLE_INLINE_MEMBER
    float inverseLength(float softening = 0.0f) const {
    #ifdef HEMI_DEV_CODE
      return rsqrtf(lengthSqr() + softening); // use fast GPU intrinsic
    #else
      return 1.0f / sqrtf(lengthSqr() + softening);
    #endif
    }

Utility Functions
=================

CUDA Error Checking
-------------------

hemi.h provides two convenience functions for checking CUDA errors. `checkCuda` verifies that its single argument has the value `cudaSuccess`, and otherwise prints an error message and asserts if #DEBUG is defined. This function is typically wrapped around CUDA 
API calls, for example:

    checkCuda( cudaMemcpy(d_stockPrice,   stockPrice,   OPT_SZ, cudaMemcpyHostToDevice) );

`checkCudaErrors` takes no arguments and checks the current state of the CUDA context for errors. This function synchronizes the CUDA device (`cudaDeviceSynchronize()`) to ensure asynchronous launch errors are caught.

Both `checkCuda` and `checkCudaErrors` act as No-ops when DEBUG is not defined (i.e. release builds).

Iteration
---------

For kernel functions with simple independent element-wise parallelism, hemi.h provides two functions to enable iterating over elements sequentially in host code or in parallel in device code. 

 - `hemiGetElementOffset()` returns the offset of the current thread within the 1D grid, or zero for host code. In device code, it resolves to `blockDim.x * blockIdx.x + threadIdx.x`.
 - `hemiGetElementStride()` returns the size of the 1D grid in threads, or one in host code. In device code, it resolves to `gridDim.x * blockDim.x`.

From the "blackscholes" example:

    // Black-Scholes formula for both call and put
    HEMI_KERNEL(BlackScholes)
        (float *callResult, float *putResult, const float *stockPrice,
         const float *optionStrike, const float *optionYears, float Riskfree,
         float Volatility, int optN)
    {
        int offset = hemiGetElementOffset();
        int stride = hemiGetElementStride();

        for(int opt = offset; opt < optN; opt += stride)
        {
        	// ... compute call and put value based on Black-Scholes formula
        }
    }

This code can be compiled and run as a sequential function on the host or as a CUDA kernel for the device.

Hemi also provides explicit 1D and 2D versions of the `hemiGetElement*()` functions, e.g. `hemiGetElementXOffset()`, `hemiGetElementYOffset()`, `hemiGetElementXStride()`, `hemiGetElementYStride()`.

Note: the `hemiGetElement*()` functions are specialized to simple (but common) element-wise parallelism. As such, they may not be useful for arbitrary strides, data sharing, or other more complex parallelism arrangements, but they may serve as examples for creating your own.

Mix and Match
=============

HEMI is intended to provide a loosely-coupled set of utilities and examples for creating reusable, portable CUDA C/C++ code. Feel free to use the parts that you need and ignore others. You may modify and replace portions as needed. We have selected a permissive open source license to encourage these kinds of flexible use.

If you make changes that you feel would be generally useful, please fork the project on github, commit your changes, and submit a pull request. 

https://github.com/harrism/hemi


License and Copyright
=====================

Copyright 2012-2014, NVIDIA Corporation

Licensed under the Apache License, v2.0.  Please see the LICENSE file included with the HEMI source code.

[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/06d00d410f9df2a26a2c2a4e81d7c1eb "githalytics.com")](http://githalytics.com/harrism/hemi)

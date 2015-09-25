Hemi: Simpler, More Portable CUDA C++
=====================================

[<img align="right" src="https://raw.github.com/harrism/hemi/master/hemi-logo-transparent.png" width="272" height="152"/>](https://raw.github.com/harrism/hemi/master/hemi-logo.png)
Hemi simplifies writing portable CUDA C/C++ code. With Hemi, 

 - you can write parallel kernels like you write for loops—in line in your CPU code—and run them on your GPU;
 - you can easily write code that compiles and runs either on the CPU or GPU;
 - you can easily launch C++ Lambda functions as GPU kernels;
 - kernel launch configuration details like thread block size and grid size are optimization details, rather than requirements.

With Hemi, parallel code for the GPU can be as simple as the `parallel_for` loop in the following code, which can also be compiled and run on the CPU.

```
void saxpy(int n, float a, const float *x, float *y)
{
  hemi::parallel_for(0, n, [=] HEMI_LAMBDA (int i) {
      y[i] = a * x[i] + y[i];
  }); 
}
```

Current Version
---------------

This is version: 2.0 (HEMI_VERSION == 200000)

Hemi on github
--------------

The home for Hemi is [http://harrism.github.io/hemi/](http://harrism.github.io/hemi/), where you can find the latest changes and information.

Blog Posts
----------
Read about Hemi 2 on the [NVIDIA Parallel Forall Blog](http://devblogs.nvidia.com/parallelforall/simple-portable-parallel-c-hemi-2/). [An older post about Hemi 1.0](http://devblogs.nvidia.com/parallelforall/developing-portable-cuda-cc-code-hemi/).

Requirements
------------

Hemi 2 requires a host compiler with support for C++11 or later. For CUDA device execution, Hemi requires CUDA 7.0 or later. To launch lambda expressions on the GPU using `hemi::launch()` or `hemi::parallel_for()`, Hemi requires CUDA 7.5 or later with experimental support for "extended lambdas" (enabled using the `nvcc` command line option `--expt-extended-lambda`).

Features
========

GPU Lambdas and Parallel For
----------------------------

CUDA 7.5 provides an experimental feature, "GPU Lambdas", which enables C++11 Lambda functions with `__device__` annotation to be defined in host code and passed to kernels running on the device. Hemi 2 leverages this feature to provide the `hemi::parallel_for` function which, when compiled for the GPU, launches a parallel kernel which executes the provided GPU lambda function as the body of a parallel loop. When compiled for the CPU, the lambda is executed as the body of a sequential CPU loop. This makes parallel functions nearly as easy to write as a for loop, as the following code shows:

    parallel_for(0, 100, [] HEMI_LAMBDA (int i) { 
        printf("%d\n", i); 
    });

GPU Lambdas can also be launched directly on the GPU using `hemi::launch`:

    hemi::launch([=] HEMI_LAMBDA() {
        printf("Hello World from Lambda in thread %d of %d\n",
            hemi::globalThreadIndex(),
            hemi::globalThreadCount());
    });

To launch lambda expressions on the GPU using `hemi::launch()` or `hemi::parallel_for()`, Hemi requires [CUDA 7.5](http://developer.nvidia.com/cuda-toolkit) or later with experimental support for "extended lambdas" (enabled using the `nvcc` command line option `--expt-extended-lambda`).

Portable Parallel Execution
---------------------------

`hemi::launch` can also be used to portably launch function objects (or *functors*), which are objects of classes that define an `operator()` member. To be launched on the GPU, the `operator()` should be declared with `HEMI_DEV_CALLABLE_MEMBER`. To make this easy, Hemi 2 provides the convenience macro `HEMI_KERNEL_FUNCTION()`. The simple example `hello.cpp` demonstrates its use:

    HEMI_KERNEL_FUNCTION(hello) {
      printf("Hello World from thread %d of %d\n",
             hemi::globalThreadIndex(),
             hemi::globalThreadCount());
    }
    
    int main(void) {
      hello hi;
      hemi::launch(hi);          // launch on the GPU
      hemi::deviceSynchronize(); // make sure print flushes before exit
    
      hi();                      // call on CPU
      return 0;
    }

As you can see, `HEMI_KERNEL_FUNCTION()` actually defines a functor which must be instantiated. Once instantiated, it can either be launched on the GPU or called from the CPU.

You can define portable CUDA kernel functions using `HEMI_LAUNCHABLE`, which defines the function using CUDA `__global__` when compiled using `nvcc`, or as a normal host function otherwise. Launch these functions portably using `hemi::cudaLaunch()`. The example `hello_global.cu` demonstrates:

    HEMI_LAUNCHABLE void hello() { 
      printf("Hello World from thread %d of %d\n", 
             hemi::globalThreadIndex(),
             hemi::globalThreadCount());
    }
    
    int main(void) {
      hemi::cudaLaunch(hello);
      hemi::deviceSynchronize(); // make sure print flushes before exit
      return 0;
    }

Automatic Execution Configuration
---------------------------------

In both of the examples in the previous section, the execution configuration (the number of thread blocks and size of each block) is automatically decided by Hemi based on the GPU it is running on.  In general, when compiled for the GPU, `hemi::launch()`, `hemi::cudaLaunch()` and `hemi::parallel_for()` will choose a grid configuration that occupies all multiprocessors (SMs) on the GPU.

Automatic Execution Configuration is flexible, though. You can explicitly specify the entire execution configuration---grid size, thread block size, and dynamic shared memory allocation---or you can partially specify the execution configuration. For example, you might need to specify just the thread block size. Hemi makes it easy to take full control when you need it for performance tuning, but when you are getting started parallelizing your code, or for functions where ultimate performance is not crucial, you can just let Hemi configure the parallelism for you.

As an example, the `nbody_vec4` example provides an optimized version of its main kernel that tiles data in CUDA shared memory. For this, it needs to specify the block size and shared memory allocation explicitly.

    const int blockSize = 256;
    hemi::ExecutionPolicy ep;
    ep.setBlockSize(blockSize);
    ep.setSharedMemBytes(blockSize * sizeof(Vec4f));
    hemi::cudaLaunch(ep, allPairsForcesShared, forceVectors, bodies, N);

However, note that the number of blocks in the grid is left to Hemi to choose at run time.

Simple Grid-Stride Loops
----------------------

A common design pattern in writing scalable, portable parallel CUDA kernels is to use [grid-stride loops](http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/). Grid-stride loops let you decouple the size of your CUDA grid from the data size it is processing, resulting in less coupling between your host and device code. This also has portability and debugging benefits. 

Hemi 2 includes a [grid-stride range](http://devblogs.nvidia.com/parallelforall/power-cpp11-cuda-7/) helper, `grid_stride_range()`, which makes it trivial to use C++11 range-based for loops to iterate in parallel. `grid_stride_range()` can be used in traditional CUDA kernels, such as the following `saxpy` kernel, or it can be combined with other Hemi portability features (in fact it is used in the implementation of `hemi::parallel_for()`).

    __global__
    void saxpy(int n, float a, float *x, float *y)
    {
      for (auto i : grid_stride_range(0, n)) {
        y[i] = a * x[i] + y[i];
      }
    }

hemi/hemi.h
-----------

The `hemi.h` header provides simple macros that are useful for reusing code between CUDA C/C++ and C/C++ written for other platforms (e.g. CPUs). 

The macros are used to decorate function prototypes and variable declarations so that they can be compiled by either NVCC or a host compiler (for example gcc or cl.exe, the MS Visual Studio compiler). 
 
The macros can be used within .cu, .cuh, .cpp, .h, and .inl files to define code that can be compiled either for the host (e.g., CPU) or the device (e.g., GPU). 

hemi/array.h
------------

One of the biggest challenges in writing portable CUDA code is memory management. HEMI provides the `hemi::Array` C++ template class as a simple data management wrapper which allows arrays of arbitrary type to be created and used with both host and device code. hemi::Array maintains a host and a device pointer for each array. It lazily transfers data between the host and device as needed when the user requests a pointer to the host or device memory. Pointer requests specify read-only, read/write, or write-only options so that valid flags can be maintained and data is only copied when the requested pointer is invalid.

For example, here is an excerpt from the nbody_vec4 example.

    hemi::Array<Vec4f> bodies(N, true);
    hemi::Array<Vec4f> forceVectors(N, true);
      	
    randomizeBodies(bodies.writeOnlyHostPtr(), N);
    
    // Call host function defined in a .cpp compilation unit
    allPairsForcesHost(forceVectors.writeOnlyHostPtr(), bodies.
                       readOnlyHostPtr(), N);
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

Portable Functions
------------------

A common use for host-device code sharing is commonly used utility functions. For example, if we wish to define an inline function to compute the average of two floats that can be called either from host code or device code, and can be compiled by either the host compiler or NVCC, we define it like this:

    HEMI_DEV_CALLABLE_INLINE float avgf(float x, float y) { 
      return (x+y)/2.0f; 
    }

The macro definition ensures that when compiled by NVCC, both a host and device version of the function are generated, and a normal inline function is generated when compiled by the host compiler.

For example use, see the `CND()` function in the "blackscholes" example, as well as several other functions used in the examples.

Portable Classes
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

Portable Kernels (Legacy Interface)
-----------------------------------

**`HEMI_KERNEL` and `HEMI_KERNEL_LAUNCH` are the Hemi 1.x interface for defining portable kernels. `HEMI_LAUNCHABLE`
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

Note: Non-Inline Functions and Methods
--------------------------------------

There are non-inline versions of the macros (`HEMI_DEV_CALLABLE` and `HEMI_DEV_CALLABLE_MEMBER`. Take care to avoid multiple definition linker errors due to using these in headers that are included into multiple compilation units. The best way to use `HEMI_DEV_CALLABLE` is to declare functions using this macro in a header, and define their implementation in a .cu file, and compile it with NVCC. This will generate code for both host and device. The host code will be linked into your library or application and callable from other host code compilation units (.c and .cpp files).  

Likewise, for `HEMI_DEV_CALLABLE_MEMBER`, put the class and function declaration in a header, and the member function implementations in a .cu file, compiled by NVCC.

Note: Device-Specific Code
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

    checkCuda(cudaMemcpy(d_stockPrice, stockPrice, OPT_SZ,
                         cudaMemcpyHostToDevice) );

`checkCudaErrors` takes no arguments and checks the current state of the CUDA context for errors. This function synchronizes the CUDA device (`cudaDeviceSynchronize()`) to ensure asynchronous launch errors are caught.

Both `checkCuda` and `checkCudaErrors` act as No-ops when DEBUG is not defined (i.e. release builds).

Iteration
---------

For kernel functions with simple independent element-wise parallelism, `hemi/device_api.h` provides functions to enable iterating over elements sequentially in host code or in parallel in device code. 

 - `globalThreadIndex()` returns the offset of the current thread within the 1D grid, or zero for host code. In device code, it resolves to `blockDim.x * blockIdx.x + threadIdx.x`.
 - `globalThreadCount()` returns the size of the 1D grid in threads, or one in host code. In device code, it resolves to `gridDim.x * blockDim.x`.

Here's a SAXPY implementation using the above functions.

    HEMI_LAUNCHABLE
    void saxpy(int n, float a, float *x, float *y)
    {
      using namespace hemi;
      for (int i = globalThreadIndex(); i < n; i += globalThreadCount()) {
        y[i] = a * x[i] + y[i];
      }
    }

Note it's simpler to use a range-based for loop using `grid_stride_range()` as shown previously.

This code can be compiled and run as a sequential function on the host or as a CUDA kernel for the device.

Hemi provides a complete set of portable element accessors in `hemi\device_api.h` including `localThreadIndex()`, `globalBlockCount()`, etc.

Note: the `globalThreadIndex()`, `globalThreadCount()`, etc. functions are specialized to simple (but common) element-wise parallelism. As such, they may not be useful for arbitrary strides, data sharing, or other more complex parallelism arrangements, but they may serve as examples for creating your own.

Mix and Match
=============

HEMI is intended to provide a loosely-coupled set of utilities and examples for creating reusable, portable CUDA C/C++ code. Feel free to use the parts that you need and ignore others. You may modify and replace portions as needed. We have selected a permissive open source license to encourage these kinds of flexible use.

If you make changes that you feel would be generally useful, please fork the project on github, commit your changes, and submit a pull request. 

https://github.com/harrism/hemi


License and Copyright
=====================

Copyright 2012-2015, NVIDIA Corporation

Licensed under the BSD License. Please see the LICENSE file included with the HEMI source code.
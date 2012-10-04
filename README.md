"HEMI" CUDA Portable C++ Macros
===============================

The hemi.h header contains simple macros that are useful for reusing code between CUDA C/C++ and C/C++ written for other platforms (e.g. CPUs). 

The macros are used to decorate function prototypes and variable declarations so that they can be compiled by either NVCC or a host compiler (for example gcc or cl.exe, the MS Visual Studio compiler). 
 
The macros can be used within .cu, .h, or .inl files, although only the latter two types should be compiled by compilers other than NVCC. Typically these functions are commonly used utility functions. For example, if we wish to define a function to compute the average of two floats that can be called either from host code or device code, and can be compiled by either the host compiler or NVCC, we define it like this:

    HEMI_DEV_CALLABLE_INLINE float avgf(float x, float y) { return (x+y)/2.0f; }

The macro definition ensure that when compiled by NVCC, both a host and device version of the function are generated, and a normal inline function is generated when compiled by the host compiler.

There are also non-inline versions of the macros, but care should be taken to avoid using these in headers that are included into multiple compilation units.

The HEMI_DEV_CALLABLE_MEMBER and HEMI_DEV_CALLABLE_INLINE macros can be used to create classes that are reuseable between host and device code, by decorating any member function prototype that will be used by both device and host code.

License and Copyright
---------------------

Copyright 2012, NVIDIA Corporation

Licensed under the Apache License, v2.0.  Please see the LICENSE file included with the HEMI source code.
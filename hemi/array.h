///////////////////////////////////////////////////////////////////////////////
// 
// "Hemi" CUDA Portable C/C++ Utilities
// 
// Copyright 2012-2015 NVIDIA Corporation
//
// License: BSD License, see LICENSE file in Hemi home directory
//
// The home for Hemi is https://github.com/harrism/hemi
//
///////////////////////////////////////////////////////////////////////////////
// Please see the file README.md (https://github.com/harrism/hemi/README.md) 
// for full documentation and discussion.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "hemi/hemi.h"
#include "hemi/stream.h"
#include <cstring>

#ifndef HEMI_ARRAY_DEFAULT_LOCATION
  #ifdef HEMI_CUDA_COMPILER
    #define HEMI_ARRAY_DEFAULT_LOCATION hemi::device
  #else
    #define HEMI_ARRAY_DEFAULT_LOCATION hemi::host
  #endif
#endif

namespace hemi {

    template <typename T> class Array; // forward decl

    enum Location {
        host   = 0,
        device = 1
    };

    template <typename T>
    class Array 
    {
    public:
        Array(size_t n, bool usePinned=true) : 
          nSize(n), 
          hPtr(nullptr),
          dPtr(nullptr),
          isForeignHostPtr(false),
          isPinned(usePinned),
          isHostValid(false),
          isDeviceValid(false),
          isAsync(false),
          streamID(0) 
        {
        }

        // Use a pre-allocated host pointer (use carefully!)
        Array(T *hostMem, size_t n) : 
          nSize(n), 
          hPtr(hostMem),
          dPtr(nullptr),
          isForeignHostPtr(true),
          isPinned(false),
          isHostValid(true),
          isDeviceValid(false),
          isAsync(false),
          streamID(0)
        {
        }

        ~Array() 
        {           
            deallocateDevice();
            if (!isForeignHostPtr)
                deallocateHost();
        }

        size_t size() const { return nSize; }


        // Make all memory transfers asynchronous.
        // The user must take care himself not to access the host memory until
        // the transfers are completed using hemi::deviceSynchronize.
        void setAsync(bool async, stream_t stream = 0)
        {
            isAsync = async;
            this->streamID = streamID;
        }

        // Make all memory transfers asynchronous and use the target stream.
        // The user must take care himself not to access the host memory until
        // the transfers are completed using hemi::Stream::synchronize.
        void setAsync(bool async, Stream & stream)
        {
            isAsync = async;
            this->streamID = stream.id();
        }


        bool async() const { return isAsync; }
        stream_t stream() const { return streamID; }


        // copy from/to raw external pointers (host or device)

        void copyFromHost(const T *other, size_t n)
        {
            if ((hPtr || dPtr) && nSize != n) {
                deallocateHost();
                deallocateDevice();
                nSize = n;
            }
            memcpy(writeOnlyHostPtr(), other, nSize * sizeof(T));
        }

        void copyToHost(T *other, size_t n) const
        {
            assert(hPtr);
            assert(n <= nSize);            
            memcpy(other, readOnlyHostPtr(), n * sizeof(T));
        }            

#ifndef HEMI_CUDA_DISABLE
        void copyFromDevice(const T *other, size_t n)
        {
            if ((hPtr || dPtr) && nSize != n) {
                deallocateHost();
                deallocateDevice();
                nSize = n;
            }
            if (isAsync)
            {
                checkCuda( cudaMemcpyAsync(writeOnlyDevicePtr(), other, 
                                           nSize * sizeof(T), cudaMemcpyDeviceToDevice, streamID) );
            }
            else
            {
                checkCuda( cudaMemcpy(writeOnlyDevicePtr(), other, 
                                      nSize * sizeof(T), cudaMemcpyDeviceToDevice) );
            }
        }

        void copyToDevice(T *other, size_t n)
        {
            assert(dPtr);
            assert(n <= nSize);
            if (isAsync)
            {            
                checkCuda( cudaMemcpyAsync(other, readOnlyDevicePtr(), 
                                           nSize * sizeof(T), cudaMemcpyDeviceToDevice, streamID) );
            }
            else
            {                                      
                checkCuda( cudaMemcpy(other, readOnlyDevicePtr(), 
                                      nSize * sizeof(T), cudaMemcpyDeviceToDevice) );
            }
        }
#endif

        // read/write pointer access

        T* ptr(Location loc = HEMI_ARRAY_DEFAULT_LOCATION) 
        { 
            if (loc == host) return hostPtr();
            else return devicePtr();
        }

        T* hostPtr()
        {
            if (isDeviceValid && !isHostValid) copyDeviceToHost();
            else if (!hPtr) allocateHost();
            else assert(isHostValid);
            isDeviceValid = false;
            return hPtr;
        }

        T* devicePtr()
        {
            if (!isDeviceValid && isHostValid) copyHostToDevice();
            else if (!dPtr) allocateDevice();
            else assert(isDeviceValid);
            isHostValid = false;
            return dPtr;
        }

        // read-only pointer access

        const T* readOnlyPtr(Location loc = HEMI_ARRAY_DEFAULT_LOCATION) const
        {
            if (loc == host) return readOnlyHostPtr();
            else return readOnlyDevicePtr();
        }

        const T* readOnlyHostPtr() const
        {
            if (isDeviceValid && !isHostValid) copyDeviceToHost();
            else assert(isHostValid);
            return hPtr;
        }

        const T* readOnlyDevicePtr() const
        {
            if (!isDeviceValid && isHostValid) copyHostToDevice();
            else assert(isDeviceValid);
            return dPtr;
        }

        // write-only pointer access -- ignore validity of existing data

        T* writeOnlyPtr(Location loc = HEMI_ARRAY_DEFAULT_LOCATION) 
        {
            if (loc == host) return writeOnlyHostPtr();
            else return writeOnlyDevicePtr();
        }

        T* writeOnlyHostPtr()
        {
            if (!hPtr) allocateHost();
            isDeviceValid = false;
            isHostValid   = true;
            return hPtr;
        }

        T* writeOnlyDevicePtr()
        {
            if (!dPtr) allocateDevice();
            isDeviceValid = true;
            isHostValid   = false;
            return dPtr;
        }

    private:
        size_t          nSize;

        mutable T       *hPtr;
        mutable T       *dPtr;

        bool            isForeignHostPtr;
        bool            isPinned;
        
        mutable bool    isHostValid;
        mutable bool    isDeviceValid;

        bool            isAsync;
        stream_t        streamID;

    protected:
        void allocateHost() const
        {
            assert(!hPtr);
#ifndef HEMI_CUDA_DISABLE
            if (isPinned)
                checkCuda( cudaHostAlloc((void**)&hPtr, nSize * sizeof(T), 0));
            else
#endif
                hPtr = new T[nSize];    
                
            isHostValid = false;

        }
        
        void allocateDevice() const
        {
#ifndef HEMI_CUDA_DISABLE
            assert(!dPtr);
            checkCuda( cudaMalloc((void**)&dPtr, nSize * sizeof(T)) );
            isDeviceValid = false;
#endif
        }

        void deallocateHost()
        {
            assert(!isForeignHostPtr);
            if (hPtr) {
#ifndef HEMI_CUDA_DISABLE
                if (isPinned)
                    checkCuda( cudaFreeHost(hPtr) );
                else
#endif
                    delete [] hPtr;
                nSize = 0;
                hPtr = nullptr;
                isHostValid = false;
            }
        }

        void deallocateDevice()
        {
#ifndef HEMI_CUDA_DISABLE
            if (dPtr) {
                checkCuda( cudaFree(dPtr) );
                dPtr = nullptr;
                isDeviceValid = false;
            }
#endif
        }

        void copyHostToDevice() const
        {
#ifndef HEMI_CUDA_DISABLE
            assert(hPtr && isHostValid);
            if (!dPtr) allocateDevice();
            if (isAsync)
            {
                assert(isPinned || isForeignHostPtr);
                checkCuda( cudaMemcpyAsync(dPtr, hPtr, nSize * sizeof(T), 
                                           cudaMemcpyHostToDevice, streamID) );
            }
            else
            {                                           
                checkCuda( cudaMemcpy(dPtr, hPtr, nSize * sizeof(T), 
                                      cudaMemcpyHostToDevice) );
            }                                          
            isDeviceValid = true;
#endif
        }

        void copyDeviceToHost() const
        {
#ifndef HEMI_CUDA_DISABLE
            assert(dPtr && isDeviceValid);
            if (!hPtr) allocateHost();
            if (isAsync)
            {
                assert(isPinned || isForeignHostPtr);
                checkCuda( cudaMemcpyAsync(hPtr, dPtr, nSize * sizeof(T), 
                                           cudaMemcpyDeviceToHost, streamID) );
            }
            else
            {
                checkCuda( cudaMemcpy(hPtr, dPtr, nSize * sizeof(T), 
                                      cudaMemcpyDeviceToHost) );
            }
            isHostValid = true;
#endif
        }
        
    };
} 

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
          hPtr(0),
          dPtr(0),
          isForeignHostPtr(false),
          isPinned(usePinned),
          isHostAlloced(false),
          isDeviceAlloced(false),
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
          dPtr(0),
          isForeignHostPtr(true),
          isPinned(false),
          isHostAlloced(true),
          isDeviceAlloced(false),
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
            if ((isHostAlloced || isDeviceAlloced) && nSize != n) {
                deallocateHost();
                deallocateDevice();
                nSize = n;
            }
            memcpy(writeOnlyHostPtr(), other, nSize * sizeof(T));
        }

        void copyToHost(T *other, size_t n) const
        {
            assert(isHostAlloced);
            assert(n <= nSize);            
            memcpy(other, readOnlyHostPtr(), n * sizeof(T));
        }            

#ifndef HEMI_CUDA_DISABLE
        void copyFromDevice(const T *other, size_t n)
        {
            if ((isHostAlloced || isDeviceAlloced) && nSize != n) {
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
            assert(isDeviceAlloced);
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
            else if (!isHostAlloced) allocateHost();
            else assert(isHostValid);
            isDeviceValid = false;
            return hPtr;
        }

        T* devicePtr()
        {
            if (!isDeviceValid && isHostValid) copyHostToDevice();
            else if (!isDeviceAlloced) allocateDevice();
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
            if (!isHostAlloced) allocateHost();
            isDeviceValid = false;
            isHostValid   = true;
            return hPtr;
        }

        T* writeOnlyDevicePtr()
        {
            if (!isDeviceAlloced) allocateDevice();
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
        
        mutable bool    isHostAlloced;
        mutable bool    isDeviceAlloced;        

        mutable bool    isHostValid;
        mutable bool    isDeviceValid;

        bool            isAsync;
        stream_t        streamID;

    protected:
        void allocateHost() const
        {
            assert(!isHostAlloced);
#ifndef HEMI_CUDA_DISABLE
            if (isPinned)
                checkCuda( cudaHostAlloc((void**)&hPtr, nSize * sizeof(T), 0));
            else
#endif
                hPtr = new T[nSize];    
                
            isHostAlloced = true;
            isHostValid = false;

        }
        
        void allocateDevice() const
        {
#ifndef HEMI_CUDA_DISABLE
            assert(!isDeviceAlloced);
            checkCuda( cudaMalloc((void**)&dPtr, nSize * sizeof(T)) );
            isDeviceAlloced = true;
            isDeviceValid = false;
#endif
        }

        void deallocateHost()
        {
            assert(!isForeignHostPtr);
            if (isHostAlloced) {
#ifndef HEMI_CUDA_DISABLE
                if (isPinned)
                    checkCuda( cudaFreeHost(hPtr) );
                else
#endif
                    delete [] hPtr;
                nSize = 0;
                isHostAlloced = false;
                isHostValid   = false;
            }
        }

        void deallocateDevice()
        {
#ifndef HEMI_CUDA_DISABLE
            if (isDeviceAlloced) {
                checkCuda( cudaFree(dPtr) );
                isDeviceAlloced = false;
                isDeviceValid   = false;
            }
#endif
        }

        void copyHostToDevice() const
        {
#ifndef HEMI_CUDA_DISABLE
            assert(isHostAlloced);
            if (!isDeviceAlloced) allocateDevice();
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
            assert(isDeviceAlloced);
            if (!isHostAlloced) allocateHost();
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

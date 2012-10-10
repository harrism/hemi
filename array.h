#include <assert.h>
#include <cuda_runtime_api.h>
#include <iostream>

namespace hemi {

    template <typename T> class Array; // forward decl

    enum Location {
        host =   0,
        device = 0
    };

#ifndef HEMI_ARRAY_DEFAULT_LOCATION
  #ifdef HEMI_CUDA_COMPILER
    #define HEMI_ARRAY_DEFAULT_LOCATION device
  #else
    #define HEMI_ARRAY_DEFAULT_LOCATION host
  #endif
#endif


    template <typename T>
    class Array 
    {
    public:
        Array(size_t n, bool usePinned) : 
          nSize(n), 
          hPtr(0),
          dPtr(0),
          isPinned(usePinned),
          isHostAlloced(false),
          isDeviceAlloced(false),
          isHostValid(false),
          isDeviceValid(false) 
        {
            allocateHost();
        }

        ~Array() 
        {           
            deallocateDevice();
            deallocateHost();
        }

        // read/write pointer access

        T* getPtr(Location loc = HEMI_ARRAY_DEFAULT_LOCATION) 
        { 
            if (loc == host) return getHostPtr();
            else return getDevicePtr();
        }

        T* getHostPtr()
        {
            assert(isHostAlloced);
            if (isDeviceValid && !isHostValid) copyDeviceToHost();
            else assert(isHostValid);
            isDeviceValid = false;
            return hPtr;
        }

        T* getDevicePtr()
        {
            if (!isDeviceValid && isHostValid) copyHostToDevice();
            else assert(isDeviceValid);
            isHostValid = false;
            return dPtr;
        }

        // read-only pointer access

        const T* getReadOnlyPtr(Location loc = HEMI_ARRAY_DEFAULT_LOCATION) const
        {
            if (loc == host) return getReadOnlyHostPtr();
            else return getReadOnlyDevicePtr();
        }

        const T* getReadOnlyHostPtr() const
        {
            if (isDeviceValid && !isHostValid) copyDeviceToHost();
            else assert(isHostValid);
            return hPtr;
        }

        const T* getReadOnlyDevicePtr() const
        {
            if (!isDeviceValid && isHostValid) copyHostToDevice();
            else assert(isDeviceValid);
            return dPtr;
        }

        // write-only pointer access -- ignore validity of existing data

        T* getWriteOnlyPtr(Location loc = HEMI_ARRAY_DEFAULT_LOCATION) 
        {
            if (loc == host) return getWriteOnlyHostPtr();
            else return getWriteOnlyDevicePtr();
        }

        T* getWriteOnlyHostPtr()
        {
            assert(isHostAlloced);
            isDeviceValid = false;
            isHostValid   = true;
            return hPtr;
        }

        T* getWriteOnlyDevicePtr()
        {
            assert(isHostAlloced);
            if (!isDeviceAlloced) allocateDevice();
            isDeviceValid = true;
            isHostValid   = false;
            return dPtr;
        }

    private:
        size_t          nSize;

        bool            isPinned;
        
        mutable T       *hPtr;
        mutable T       *dPtr;

        mutable bool    isHostAlloced;
        mutable bool    isDeviceAlloced;        

        mutable bool    isHostValid;
        mutable bool    isDeviceValid;

    protected:
        inline
        cudaError_t checkCuda(cudaError_t result) const {
#if defined(DEBUG) || defined(_DEBUG)
            if (result != cudaSuccess) {
                std::cerr << "CUDA Error" 
                          << cudaGetErrorString(result) << std::endl;
            }
            assert(result == cudaSuccess);
#endif
            return result;
        }

        void allocateHost() const
        {
            assert(!isHostAlloced);
            assert(!isDeviceAlloced);

            if (isPinned)
                checkCuda( cudaHostAlloc((void**)&hPtr, nSize * sizeof(T), 0));
            else
                hPtr = new T[nSize];    
                
            isHostAlloced = true;
            isHostValid   = true;
        }
        
        void allocateDevice() const
        {
            assert(!isDeviceAlloced);
            checkCuda( cudaMalloc((void**)&dPtr, nSize * sizeof(T)) );
            isDeviceAlloced = true;
        }

        void deallocateHost()
        {
            if (isHostAlloced) {
                if (isPinned)
                    checkCuda( cudaFreeHost(hPtr) );
                else
                    delete [] hPtr;
                nSize = 0;
                isHostAlloced = false;
                isHostValid   = false;
            }
        }

        void deallocateDevice()
        {
            if (isDeviceAlloced) {
                checkCuda( cudaFree(dPtr) );
                isDeviceAlloced = false;
                isDeviceValid   = false;
            }
        }

        void copyHostToDevice() const
        {
            assert(isHostAlloced);
            if (!isDeviceAlloced) allocateDevice();
            checkCuda( cudaMemcpy(dPtr, 
                                  hPtr, 
                                  nSize * sizeof(T), 
                                  cudaMemcpyHostToDevice) );
            isDeviceValid = true;
        }

        void copyDeviceToHost() const
        {
            assert(isDeviceAlloced && isHostAlloced);
            checkCuda( cudaMemcpy(hPtr, 
                                  dPtr, 
                                  nSize * sizeof(T), 
                                  cudaMemcpyDeviceToHost) );
            isHostValid = true;
        }
        
    };
} 
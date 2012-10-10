#include "hemi/hemi.h"

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

        T* ptr(Location loc = HEMI_ARRAY_DEFAULT_LOCATION) 
        { 
            if (loc == host) return hostPtr();
            else return devicePtr();
        }

        T* hostPtr()
        {
            assert(isHostAlloced);
            if (isDeviceValid && !isHostValid) copyDeviceToHost();
            else assert(isHostValid);
            isDeviceValid = false;
            return hPtr;
        }

        T* devicePtr()
        {
            if (!isDeviceValid && isHostValid) copyHostToDevice();
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
            assert(isHostAlloced);
            isDeviceValid = false;
            isHostValid   = true;
            return hPtr;
        }

        T* writeOnlyDevicePtr()
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
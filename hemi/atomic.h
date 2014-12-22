#ifndef __HEMI_ATOMIC_H__
#define __HEMI_ATOMIC_H__

#include "hemi/hemi.h"

// Open-MP specifics
#ifdef _OPENMP
  #include <omp.h>
#else
  /* Typedef here is to let the compiler have something reasonable for when you
   * are not using OpenMP, so that it compiles. The functions do fall through
   * to the ones above, though, so it should be safe to pass 0 to that without
   * openmp defined and all should work.
   * We assume that the lock is initialized prior to calling any of these functions.
   */
typedef unsigned int omp_lock_t;
#endif // _OPENMP


/* Straightforward atomic implementations for device or host code.  Examples
 * use OpenMP. If you need something for your threading environment of choice,
 * then overload the function, put up whatever barriers are needed, and then
 * call this.
 *
 * The only real important thing is that the overload also be
 * HEMI_DEV_CALLABLE_INLINE and the basic ifdef/else/endif structure goes
 * there.
 *
 */

namespace hemi
{
  // Basic functions, assumed sequentials.
  HEMI_DEV_CALLABLE_INLINE int atomicAdd(int* address, int val)
  {
    #ifdef HEMI_DEV_CODE
      return ::atomicAdd(address, val);
    #else
      float old = *address;
      *address = old + val;
      return old;
    #endif
  }

  HEMI_DEV_CALLABLE_INLINE unsigned int atomicAdd(unsigned int* address, unsigned int val)
  {   
    #ifdef HEMI_DEV_CODE
      return ::atomicAdd(address, val);
    #else
      unsigned int old = *address;
      *address = old + val;
      return old;
    #endif
  } 
  HEMI_DEV_CALLABLE_INLINE unsigned long long int atomicAdd(unsigned long long int* address, unsigned long long int val)
  {   
    #ifdef HEMI_DEV_CODE
      return ::atomicAdd(address, val);
    #else
      unsigned long long int old = *address;
      *address = old + val;
      return old;
    #endif
  } 
  
  HEMI_DEV_CALLABLE_INLINE float atomicAdd(float* address, float val)
  {   
    #ifdef HEMI_DEV_CODE
      return ::atomicAdd(address, val);
    #else
      float old = *address;
      *address = old + val;
      return old;
    #endif
  } 

  HEMI_DEV_CALLABLE_INLINE int atomicCAS(int* address, int compare, int val) 
  {
    #ifdef HEMI_DEV_CODE
      return ::atomicCAS(address, compare, val);
    #else
      int old = *address;
      *address = (old == compare ? val : old);
      return old; 
    #endif
  }
  HEMI_DEV_CALLABLE_INLINE unsigned int atomicCAS(unsigned int* address, unsigned int compare, unsigned int val)
  {
    #ifdef HEMI_DEV_CODE
      return ::atomicCAS(address, compare, val);
    #else
      unsigned int old = *address;
      *address = (old == compare ? val : old);
      return old; 
    #endif
  }
  HEMI_DEV_CALLABLE_INLINE unsigned long long int atomicCAS(unsigned long long int* address, unsigned long long int compare, unsigned long long int val)
  {
    #ifdef HEMI_DEV_CODE
      return ::atomicCAS(address, compare, val);
    #else
      unsigned long long int old = *address;
      *address = (old == compare ? val : old);
      return old; 
    #endif
  }

  /* OpenMP-supported functions. These functions lock/unlock instead of using
   * named critical sections, because the named critical sections are global in
   * scope. If the lock acquisition fails, the function immediately returns
   * with the current value of address.
   */



  HEMI_DEV_CALLABLE_INLINE int atomicCAS(int* address, int compare, int val, omp_lock_t* lock) 
  {
    #ifdef HEMI_DEV_CODE
      return ::atomicCAS(address, compare, val);
    #else
      #ifdef _OPENMP
        omp_set_lock(lock);
        int old = hemi::atomicCAS(address, compare, val);
        omp_unset_lock(lock);
        return old; 
      #else
        return hemi::atomicCAS(address, compare, val);
      #endif // _OPENMP
    #endif // HEMI_DEV_CODE
  }

  HEMI_DEV_CALLABLE_INLINE unsigned int atomicCAS(unsigned int* address, unsigned int compare, unsigned int val, omp_lock_t* lock) 
  {
    #ifdef HEMI_DEV_CODE
      return ::atomicCAS(address, compare, val);
    #else
      #ifdef _OPENMP
        omp_set_lock(lock);
        unsigned int old = hemi::atomicCAS(address, compare, val);
        omp_unset_lock(lock);
        return old; 
      #else
        return hemi::atomicCAS(address, compare, val);
      #endif // _OPENMP
    #endif // HEMI_DEV_CODE
  }

  HEMI_DEV_CALLABLE_INLINE unsigned long long int atomicCAS(unsigned long long int* address, unsigned long long int compare, unsigned long long int val, omp_lock_t* lock) 
  {
    #ifdef HEMI_DEV_CODE
      return ::atomicCAS(address, compare, val);
    #else
      #ifdef _OPENMP
        omp_set_lock(lock);
        unsigned long long int old = hemi::atomicCAS(address, compare, val);
        omp_unset_lock(lock);
        return old; 
      #else
        return hemi::atomicCAS(address, compare, val);
      #endif // _OPENMP
    #endif // HEMI_DEV_CODE
  }

  HEMI_DEV_CALLABLE_INLINE int atomicExch(int* address, int val) 
  {
    #ifdef HEMI_DEV_CODE
      return ::atomicExch(address,val);
    #else
      int old = *address;
      *address = val;
      return old;
    #endif // HEMI_DEV_CODE
  }
  HEMI_DEV_CALLABLE_INLINE unsigned int atomicExch(unsigned int* address, unsigned int val) 
  {
    #ifdef HEMI_DEV_CODE
      return ::atomicExch(address,val);
    #else
      unsigned int old = *address;
      *address = val;
      return old;
    #endif // HEMI_DEV_CODE
  }
  HEMI_DEV_CALLABLE_INLINE unsigned long long int atomicExch(unsigned long long int* address, unsigned long long int val) 
  {
    #ifdef HEMI_DEV_CODE
      return ::atomicExch(address,val);
    #else
      unsigned long long int old = *address;
      *address = val;
      return old;
    #endif // HEMI_DEV_CODE
  }
  HEMI_DEV_CALLABLE_INLINE float atomicExch(float* address, float val) 
  {
    #ifdef HEMI_DEV_CODE
      return ::atomicExch(address,val);
    #else
      float old = *address;
      *address = val;
      return old;
    #endif // HEMI_DEV_CODE
  }
}

#endif

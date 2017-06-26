///////////////////////////////////////////////////////////////////////////////
//
// "Hemi" CUDA Portable C/C++ Utilities
//
// Extended by Brandon Wilson
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

// Functions available in host code to query for gpu devices and set properties

#include "hemi.h"

namespace hemi {

	///////Device Runtime Cache///////////////////////////////////////////
	class DeviceRuntimeCache
	{
	public:
		DeviceRuntimeCache()
		{
			flush();
		}

		// Return a static reference to this machines DeviceRuntimeCache for the current device
		static DeviceRuntimeCache& get()
		{
			static DeviceRuntimeCache instance;
			return instance;
		}

		inline void flush()
		{
			bHasGPUCached = false;
			bNumGPUsCached = false;
		}

		bool bHasGPUCached;
		bool bHasGPU;
		bool bNumGPUsCached;
		int iNumGPUs;
	};

	static void FlushRuntimeCache()
	{
		DeviceRuntimeCache::get().flush();
	}

  /*
   * Returns true if a physical device (GPU) is available on this machine
   */
	static bool queryForDevice()
	{
#ifdef HEMI_CUDA_COMPILER
		if(DeviceRuntimeCache::get().bHasGPUCached)
			return DeviceRuntimeCache::get().bHasGPU;

		int deviceCount, device;
		struct cudaDeviceProp properties;
		cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
		if (cudaResultCode != cudaSuccess)
			deviceCount = 0;
		/* machines with no GPUs can still report one emulation device */
		for (device = 0; device < deviceCount; ++device) {
			cudaGetDeviceProperties(&properties, device);
			if (properties.major != 9999) /* 9999 means emulation only */
				DeviceRuntimeCache::get().bHasGPUCached = true;
				DeviceRuntimeCache::get().bHasGPU = true;
				return true; /* success */
		}
		DeviceRuntimeCache::get().bHasGPUCached = true;
		DeviceRuntimeCache::get().bHasGPU = false;
#endif
		return false; /* failure */
	}

  /*
   * Gets the number of physical devices (GPUs) available on this machine
   */
	static int getNumDevices()
	{
#ifdef HEMI_CUDA_COMPILER
		if (DeviceRuntimeCache::get().bNumGPUsCached)
			return DeviceRuntimeCache::get().iNumGPUs;

		int gpuDeviceCount = 0;
		int deviceCount, device;
		struct cudaDeviceProp properties;
		cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
		if (cudaResultCode != cudaSuccess)
			deviceCount = 0;
		/* machines with no GPUs can still report one emulation device */
		for (device = 0; device < deviceCount; ++device) {
			cudaGetDeviceProperties(&properties, device);
			if (properties.major != 9999) /* 9999 means emulation only */
				gpuDeviceCount++;
		}
		DeviceRuntimeCache::get().bNumGPUsCached = true;
		DeviceRuntimeCache::get().iNumGPUs = gpuDeviceCount;
		return gpuDeviceCount;
#else
		return 0;
#endif
	}

	///////Device Properties///////////////////////////////////////////////////

  /*
   * Sets the current device for this thread
   */
	static void setDevice(int deviceID) {
#ifdef HEMI_CUDA_COMPILER
		if(queryForDevice())
			cudaSetDevice(deviceID);
#endif
	}

  /*
   * Gets the current device ID for this thread
   */
	static int getDevice() {
		int deviceID = -1;
#ifdef HEMI_CUDA_COMPILER
		if(queryForDevice())
			cudaGetDevice(&deviceID);
#endif
		return deviceID;
	}

  /*
   * Sets the shared memory bank size as follows:
   * true - sets the bank size to 8 bytes
   * false - sets the bank size to 4 bytes
   */
	static void setSharedMemConfig(bool ShouldBeEightByteBankSize) {
#ifdef HEMI_CUDA_COMPILER
		if(queryForDevice()) {
			if(ShouldBeEightByteBankSize)
				cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
			else
				cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
		}
#endif
	}

  /* 0  - prefer none
	 * 1  - prefer equal (CC 3.x or higher)
	 * 2  - prefer shared
	 * 3  - prefer L1 cache
	 */
	enum CachePreference {
	        prefer_none     = 0,
	        prefer_equal	= 1,
	        prefer_shared   = 2,
	        prefer_L1		= 3
	};

  /* Set the device CachePreference flag, as shown below:
   * 0  - prefer none
	 * 1  - prefer equal (CC 3.x or higher)
	 * 2  - prefer shared
	 * 3+ - prefer L1 cache
	 */
	static void setCacheConfig(int PreferenceFlag) {
#ifdef HEMI_CUDA_COMPILER
		if(queryForDevice()) {
			switch(PreferenceFlag)
			{
			case prefer_none :
				cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
				break;
			case prefer_equal :
				cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
				break;
			case prefer_shared :
				cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
				break;
			case prefer_L1 :
				cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
				break;
			}
		}
#endif
	}

  /* Set the device CachePreference flag, as shown below:
   * 0  - prefer none
	 * 1  - prefer equal (CC 3.x or higher)
	 * 2  - prefer shared
	 * 3+ - prefer L1 cache
	 */
	static void setHeapSize(size_t HeapSize) {
#ifdef HEMI_CUDA_COMPILER
		if(queryForDevice()) {
			cudaDeviceSetLimit(cudaLimitMallocHeapSize, HeapSize);
		}
#endif
	}

  /*
   * Get the available and total memory for the current device
   */
	static void getMemInfo(size_t *free_mem, size_t *total_mem) {
#ifdef HEMI_CUDA_COMPILER
		if(queryForDevice())
			cudaMemGetInfo(free_mem, total_mem);
		else
#endif
		{
			*free_mem = 0;
			*total_mem = 0;
		}
	}

}

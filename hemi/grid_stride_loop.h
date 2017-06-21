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

///////////////////////////////////////////////////////////////////
// Some utility code to easily loop in a grid-stride pattern on
// the device, or iteratively on the host
///////////////////////////////////////////////////////////////////

#include "hemi.h"
#include "device_api.h"

namespace hemi {

#define GRID_STRIDE_LOOP(name, start, size) \
	for (int name = hemi::globalThreadIndex() + (start); \
		 name < size; \
		 name += hemi::globalThreadCount())

#define GRID_STRIDE_LOOP_X(name, start, size) \
	for (int name = hemi::xGlobalThreadIndex() + (start); \
		 name < size; \
		 name += hemi::xGlobalThreadCount())

#define GRID_STRIDE_LOOP_Y(name, start, size) \
	for (int name = hemi::yGlobalThreadIndex() + (start); \
		 name < size; \
		 name += hemi::yGlobalThreadCount())

#define GRID_STRIDE_LOOP_Z(name, start, size) \
	for (int name = hemi::zGlobalThreadIndex() + (start); \
		 name < size; \
		 name += hemi::zGlobalThreadCount())

#define LOCAL_GRID_STRIDE_LOOP(name, start, size) \
	for (int name = hemi::localThreadIndex() + (start); \
		 name < size; \
		 name += hemi::localThreadCount())

#define LOCAL_GRID_STRIDE_LOOP_X(name, start, size) \
	for (int name = hemi::xLocalThreadIndex() + (start); \
		 name < size; \
		 name += hemi::xLocalThreadCount())

#define LOCAL_GRID_STRIDE_LOOP_Y(name, start, size) \
	for (int name = hemi::yLocalThreadIndex() + (start); \
		 name < size; \
		 name += hemi::yLocalThreadCount())

#define LOCAL_GRID_STRIDE_LOOP_Z(name, start, size) \
	for (int name = hemi::zLocalThreadIndex() + (start); \
		 name < size; \
		 name += hemi::zLocalThreadCount())

}

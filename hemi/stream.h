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

namespace hemi {

#ifndef HEMI_CUDA_DISABLE
    typedef cudaStream_t stream_t;

    class Stream
    {
    public:
        Stream()
        {
            checkCuda( cudaStreamCreate(&stream) );
        }

        ~Stream()
        {
            checkCuda( cudaStreamDestroy(stream) );
        }

        void synchronize() const
        {
            checkCuda( cudaStreamSynchronize(stream) );
        }

        // return true if all operations in the stream have completed; false otherwise
        bool query() const
        {
            auto res = cudaStreamQuery(stream);
            switch(res) {
                case cudaSuccess: return true;
                case cudaErrorNotReady: return false;
                default: checkCuda(res); return false;
            }
        }

        stream_t id() const
        {
            return stream;
        }

    private:
        stream_t stream;
    };

#else
    typedef int stream_t;

    class Stream
    {
    public:
        void synchronize() const {};
        bool query() const {return true};
        stream_t id() const {return 0};
    };

    typedef Stream NullStream;
#endif

} //namespace

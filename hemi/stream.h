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
#include <utility>

namespace hemi {

#ifndef HEMI_CUDA_DISABLE
    typedef cudaStream_t stream_t;
    typedef cudaEvent_t event_t;
#else
    
    typedef int stream_t;
    typedef int event_t;
#endif

    // forward declaration
    class Stream;

    class Event
    {
    public:
        // blockingSync: Specifies that event should use blocking synchronization.
        // A host thread that uses synchronize() to wait on an event created with
        // this flag will block until the event actually completes.
        // disableTiming: Specifies that the created event does not need to record
        // timing data. Events created with this flag specified and the blockingSync
        // flag not specified will provide the best performance when used with
        // hemi::Stream::waitEvent() and hemi::Event::query().
        Event(bool blockingSync = false, bool disableTiming = false):
            event(0),
            isForeign(false)
        {
#ifndef HEMI_CUDA_DISABLE
            auto flags = cudaEventDefault;
            if (blockingSync) flags |= cudaEventBlockingSync;
            if (disableTiming) flags |= cudaEventDisableTiming;
            checkCuda( cudaEventCreateWithFlags(&event, flags) );
#endif
        }

        Event(event_t id):
            event(id),
            isForeign(true)
        {}

        Event(Event & rhs):
            event(rhs.id()),
            isForeign(true)
        {}

        ~Event()
        {
#ifndef HEMI_CUDA_DISABLE        
            if (!isForeign) checkCuda( cudaEventDestroy(event) );
#endif
        }

        // Block host code until event is completed
        // Return immediately if record() was never called
        void synchronize() const
        {
#ifndef HEMI_CUDA_DISABLE        
            checkCuda( cudaEventSynchronize(event) );
#endif
        }
        
        // Any future work submitted in any stream will wait for this event to complete before
        // beginning execution. This effectively creates a barrier for all future work submitted
        // to the device on this thread.
        void syncStreams() const
        {
#ifndef HEMI_CUDA_DISABLE        
            checkCuda( cudaStreamWaitEvent(NULL, event, 0) );
#endif
        }

        // return true if all operations in the stream have completed; false otherwise
        bool query() const
        {
#ifndef HEMI_CUDA_DISABLE
            auto res = cudaEventQuery(event);
            switch(res) {
                case cudaSuccess: return true;
                case cudaErrorNotReady: return false;
                default: checkCuda(res); return false;
            }
#else
            return true;
#endif                        
        }

        void record(stream_t stream = 0) const
        {
#ifndef HEMI_CUDA_DISABLE        
            checkCuda( cudaEventRecord(event, stream) );
#endif            
        }

        void record(Stream & stream) const;
        // Implementation must happen after the definition of Stream
        //{
        //    record(stream.id());
        //}

        // if both Events have completed, returns <true, (this - other)>
        // if one of the Events has not completed yet, returns <false, 0>
        std::pair<bool, float> elapsedTime(event_t other) const
        {
            float ms = 0.0;
            bool isCompleted;

#ifndef HEMI_CUDA_DISABLE
            auto res = cudaEventElapsedTime(&ms, other, event);
            switch(res) {
                case cudaSuccess:
                    isCompleted = true;
                    break;
                case cudaErrorInvalidResourceHandle:
                    // record() has not been called on either stream,
                    // or either event was created with disableTiming
                    isCompleted = true;
                    break;
                case cudaErrorNotReady:
                    isCompleted = false;
                    break;
                default:
                    checkCuda(res);
            }
#else
            isCompleted = true;
#endif
            return std::make_pair(isCompleted, ms);
        }

        std::pair<bool, float> elapsedTime(Event & other) const
        {
            return elapsedTime(other.id());   
        }

        event_t id() const
        {
            return event;
        }

    private:
        event_t event;
        bool isForeign;
    };


    class Stream
    {
    public:
        Stream():
            isForeign(false)
        {
#ifndef HEMI_CUDA_DISABLE        
            checkCuda( cudaStreamCreate(&stream) );
#else
            stream = 0;
#endif
        }
        
        Stream(stream_t id):
            stream(id),
            isForeign(true)
        {}
        
        Stream(Stream & rhs):
            stream(rhs.id()),
            isForeign(true)
        {}        

        ~Stream()
        {
#ifndef HEMI_CUDA_DISABLE        
            if (!isForeign) checkCuda( cudaStreamDestroy(stream) );
#endif
        }

        void synchronize() const
        {
#ifndef HEMI_CUDA_DISABLE        
            checkCuda( cudaStreamSynchronize(stream) );
#endif            
        }

        // return true if all operations in the stream have completed; false otherwise
        bool query() const
        {
#ifndef HEMI_CUDA_DISABLE        
            auto res = cudaStreamQuery(stream);
            switch(res) {
                case cudaSuccess: return true;
                case cudaErrorNotReady: return false;
                default: checkCuda(res); return false;
            }
#else
            return true;
#endif            
        }

        void waitEvent(event_t event) const
        {
#ifndef HEMI_CUDA_DISABLE
            checkCuda( cudaStreamWaitEvent(stream, event, 0) );
#endif            
        }

        void waitEvent(Event & event) const
        {
            waitEvent(event.id());
        }

        stream_t id() const
        {
            return stream;
        }

    private:
        stream_t stream;
        bool isForeign;
    };


    void Event::record(Stream & stream) const
    {
        record(stream.id());
    }

} //namespace

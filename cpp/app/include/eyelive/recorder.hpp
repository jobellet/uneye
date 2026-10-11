// Preallocated, memory-mapped session file (CONTRACT). Implemented by task T01. The file IS the buffer: data survive a crash and other processes can map it read-only.
//
// File layout (little endian, all offsets multiples of 64):
//   [0, 256)            FileHeader  { char magic[8] = "EYELIVE1"; uint32 version = 1; uint32 header_bytes = 256; uint64 capacity; std::atomic-like uint64 count; double rate_hz; double deg_per_unit;
//                                     uint32 n_classes; uint32 flags (bit0 = cleanly closed); uint64 created_unix_ns; char reserved[...] }
//   [256, 512)          StateBlock  (state_block.hpp, task T03 writes it; the recorder only reserves the space)
//   [512, ...)          arrays of `capacity` elements, in this order: t (float64), x (float32), y (float32), flags (uint8), label (uint8, detector class per sample), expert (uint8, rendered expert class, 255 = ignore/none)
//   [..., end)          events table: uint64 n_events, then Event records (types.hpp) appended by the engine (capacity: 100000 events)
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include "eyelive/trace.hpp"
#include "eyelive/types.hpp"

namespace eyelive {

struct RecorderConfig {
    std::string path;                  // session file to create (parent directory must exist)
    double rate_hz = 1000.0;           // expected rate (from RateEstimator or declared)
    double duration_s = 3600.0;        // planned duration of the experiment
    double margin = 1.05;              // capacity = ceil(rate_hz * duration_s * margin) + 1024
    bool reserve_disk = true;          // really allocate the blocks (posix_fallocate / F_PREALLOCATE / SetEndOfFile + zero) so the disk cannot fill up during the session
    bool lock_memory = true;           // best-effort mlock / VirtualLock of the mapping; failure is not an error
    SessionInfo session;
};

struct RecorderSizing { uint64_t capacity_samples; uint64_t bytes; };
RecorderSizing recorder_sizing(double rate_hz, double duration_s, double margin = 1.05);   // 19 bytes per sample + header + event table; pure function, used by the GUI to show the memory estimate

class Recorder : public ITraceView {
 public:
    static std::unique_ptr<Recorder> create(const RecorderConfig& cfg, std::string* error);       // allocates and maps; fails loudly (error message) if the space cannot be reserved
    static std::unique_ptr<Recorder> open_readonly(const std::string& path, std::string* error);  // reload a finished or crashed session (count is trusted, arrays beyond it ignored)
    virtual ~Recorder() = default;
    virtual bool push(const Sample& s) = 0;                    // single writer; false when full (the engine stops the session with a warning). Publishes `count` with release semantics
    virtual int64_t capacity() const = 0;
    virtual void set_label(int64_t i, uint8_t cls) = 0;        // detector class of sample i
    virtual void set_expert(int64_t i, uint8_t cls) = 0;
    virtual uint8_t expert(int64_t i) const = 0;
    virtual bool append_event(const Event& e) = 0;             // events table
    virtual int64_t event_count() const = 0;
    virtual Event event(int64_t k) const = 0;
    virtual void* state_block() = 0;                           // pointer to the reserved 256-byte StateBlock region (T03)
    virtual bool close(std::string* error) = 0;                // msync, mark cleanly closed
    virtual bool export_csv(const std::string& path, std::string* error) const = 0;
    virtual const std::string& path() const = 0;
};

}  // namespace eyelive

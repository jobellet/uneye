// Sample sources and sampling-rate estimation (CONTRACT). Implemented by task T02.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "eyelive/types.hpp"

namespace eyelive {

// Input protocol v1 over UDP, little endian:  uint32 magic 0x31455945 ("EYE1"), uint32 n, then n x { double t_seconds, float x, float y, uint32 flags } (20 bytes each; flags: kFlagValid, kFlagBlink).
// A text mode is also accepted (one sample per line: "t x y [valid]"), recognised by a first byte that is a digit, '-', '+' or '.'.
constexpr uint32_t kUdpMagic = 0x31455945u;

struct SourceSpec {
    std::string kind = "sim";          // "udp" | "file" | "sim"
    std::string bind_address = "127.0.0.1"; int port = 5005;                   // udp
    std::string path;                  // file: CSV "t,x,y[,valid]" (header line optional)
    double speed = 1.0;                // file: 1.0 = real time, 0 = as fast as possible, 4.0 = four times faster
    double rate_hz = 1000.0;           // sim: sampling rate; file without timestamps: declared rate
    uint64_t seed = 1;                 // sim
    double pixels_per_degree = 0;      // >0: positions in the stream are pixels, converted to degrees at the source (SessionInfo::deg_per_unit stays 1); 0: used as received
};

class ISource {
 public:
    virtual ~ISource() = default;
    virtual bool open(std::string* error) = 0;
    // Blocks at most timeout_ms; appends up to max_samples to `out`; returns the number appended (0 on timeout). Never throws. A source that has ended (file finished) sets finished().
    virtual size_t read(std::vector<Sample>& out, size_t max_samples, int timeout_ms) = 0;
    virtual bool finished() const = 0;
    virtual void close() = 0;
    virtual std::string description() const = 0;
    virtual uint64_t dropped_packets() const { return 0; }                     // udp: sequence gaps or truncated datagrams
};

std::unique_ptr<ISource> make_source(const SourceSpec& spec, std::string* error);       // defined by T02

// Measures the sampling rate from the first samples: median of dt, jitter (MAD / median), gap count; proposes the nearest standard rate.
class RateEstimator {
 public:
    void add(double t_seconds);                                                // call for every sample (monotonic or not: backwards jumps are counted as gaps)
    bool ready() const;                                                        // true after >= 200 samples or 0.5 s of data
    double rate_hz() const;                                                    // median-based estimate
    double standard_rate_hz() const;                                           // nearest of 60, 120, 250, 300, 500, 1000, 1200, 2000 Hz if within 2 %, else rate_hz()
    double jitter() const;                                                     // relative spread of dt
    uint64_t gaps() const;                                                     // dt larger than 3 x median
 private:
    std::vector<double> dts_; double last_ = 0; bool has_last_ = false; uint64_t gaps_ = 0; size_t seen_ = 0;
};

}  // namespace eyelive

// Read access to a recording (CONTRACT). Detectors, trainers, calibration and the GUI only see a recording through ITraceView, so that every module can be tested
// with the in-memory VectorTrace below and none depends on the memory-mapped Recorder.
#pragma once
#include <cstdint>
#include <vector>
#include "eyelive/types.hpp"

namespace eyelive {

class ITraceView {
 public:
    virtual ~ITraceView() = default;
    virtual int64_t size() const = 0;                          // number of samples available (grows while recording; read once per call, never cache across calls)
    virtual double t(int64_t i) const = 0;
    virtual float x(int64_t i) const = 0;
    virtual float y(int64_t i) const = 0;
    virtual uint32_t flags(int64_t i) const = 0;
    virtual uint8_t label(int64_t i) const = 0;                // per-sample class written by the pre-labeler / rendered expert labels, 0 = background
    virtual const SessionInfo& info() const = 0;
    bool valid(int64_t i) const { return (flags(i) & kFlagValid) != 0; }
};

class VectorTrace : public ITraceView {                        // simple owning implementation (tests, batch use)
 public:
    explicit VectorTrace(SessionInfo info = {}) : info_(std::move(info)) {}
    void push(const Sample& s) { t_.push_back(s.t); x_.push_back(s.x); y_.push_back(s.y); f_.push_back(s.flags); l_.push_back(0); }
    void set_label(int64_t i, uint8_t c) { l_[(size_t)i] = c; }
    SessionInfo& mutable_info() { return info_; }
    int64_t size() const override { return (int64_t)t_.size(); }
    double t(int64_t i) const override { return t_[(size_t)i]; }
    float x(int64_t i) const override { return x_[(size_t)i]; }
    float y(int64_t i) const override { return y_[(size_t)i]; }
    uint32_t flags(int64_t i) const override { return f_[(size_t)i]; }
    uint8_t label(int64_t i) const override { return l_[(size_t)i]; }
    const SessionInfo& info() const override { return info_; }
 private:
    SessionInfo info_; std::vector<double> t_; std::vector<float> x_, y_; std::vector<uint32_t> f_; std::vector<uint8_t> l_;
};

}  // namespace eyelive

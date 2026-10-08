// Online (sample-by-sample) saccade / microsaccade labeling with U'n'Eye.
#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "uneye/causal_engine.hpp"
#include "uneye/engine.hpp"

namespace uneye {

// Number of samples in a time bin of `ms` milliseconds, rounded up to a multiple of mp^2 (=25), as the U-Net needs.
inline int window_samples(double fs, double ms, int mp = 5) {
    const int q = mp * mp;
    const int n = int(ms * fs / 1000.0 + 0.5);
    return std::max(q, (n + q - 1) / q * q);
}

struct Config {
    double fs = 1000.0;          // sampling rate of the incoming stream (Hz); models are trained at 1000 (or 500) Hz
    double window_ms = 200.0;    // length of the time bin the network sees (use window_samples() to build the engine)
    int hop = 1;                 // run the network every `hop` samples (1 = on every new sample, as fast as possible)
    int fast_confirm = 3;        // consecutive fast labels needed before on_fast_onset fires
    int lookahead = 25;          // a sample is labelled once `lookahead` newer samples exist (latency = lookahead ms @1kHz)
    double threshold = 0.5;      // saccade probability threshold (2-class models)
    double min_sacc_dur_ms = 6;  // drop events shorter than this (2-class models)
    double min_sacc_dist_ms = 1; // merge events closer than this (1 = off)
    double inf_correction = 1.5; // replacement for inf velocity
    double input_scale = 1.0;    // multiply input positions to obtain degrees
};

struct Event {
    int cls = 1;            // 1 = saccade (2-class models); other ids for multi-class models
    int64_t onset = 0;      // first sample index (0-based)
    int64_t offset = 0;     // last sample index (inclusive)
    int64_t confirmed = 0;  // number of samples received when the event became known (latency = confirmed - 1 - onset)
};

// Immediate (zero-lookahead) result for the newest sample of the current time bin.
struct Fast {
    int64_t index;  // = newest sample
    int cls;
    float p_sacc;
};

struct Label {
    int64_t index;  // sample index
    int cls;        // committed class of this sample (after `lookahead`)
    float p_sacc;   // probability of class != 0
};

class StreamingDetector {
public:
    StreamingDetector(std::unique_ptr<Engine> engine, const Config& cfg);
    // Causal network: one cheap step per sample, labels are final immediately (no window, no lookahead).
    StreamingDetector(std::unique_ptr<StepEngine> engine, const Config& cfg);

    // Feed one gaze sample (degrees; NaN = missing/blink). May emit labels / events.
    void push(double x, double y);

    // Start a new recording (clears all state, keeps the loaded network).
    void reset();

    // Flush: commit everything still pending (end of recording).
    void finish();

    // callbacks (optional)
    std::function<void(const Label&)> on_label;   // every sample, in order, `lookahead` late
    std::function<void(const Event&)> on_event;   // every finished event (after min-duration / merging)
    std::function<void(const Fast&)> on_fast;     // after EVERY network call: label of the newest sample (no lookahead)
    std::function<void(int64_t)> on_fast_onset;   // fast_confirm consecutive fast saccade labels; arg = first sample
    std::function<void(int64_t)> on_onset;        // provisional: event running for >= min duration, not yet finished

    // latest un-committed probability for the newest sample (no lookahead, noisy edge) – for ultra-low-latency use
    float provisional_p() const { return provisional_; }
    int64_t samples() const { return n_; }
    const Config& config() const { return cfg_; }
    // timing of the last network call (ns)
    int64_t last_infer_ns() const { return last_ns_; }

private:
    void init_common();
    void run_network();
    void step_causal();
    void commit(int64_t upto, const std::vector<float>& prob, int64_t win_start);
    void feed_label(int64_t idx, int cls, float p);
    void close_run(int64_t end_idx);
    void flush_pending(int64_t now, bool force);

    std::unique_ptr<Engine> eng_;
    std::unique_ptr<StepEngine> step_;
    std::vector<float> sprob_;
    Config cfg_;
    int W_, C_;
    int min_dur_, min_dist_;
    int64_t n_ = 0, committed_ = 0, last_infer_n_ = 0;
    double px_ = 0, py_ = 0, last_dx_ = 0, last_dy_ = 0;
    bool have_prev_ = false;
    std::vector<float> dxy_;     // ring of last W (dX,dY) pairs, linearised on demand
    std::vector<float> ring_;    // 2*W
    std::vector<float> win_, prob_;
    float provisional_ = 0;
    int64_t last_ns_ = 0;
    int fast_run_ = 0;

    // event state machine (runs on committed labels)
    int run_cls_ = 0;
    int64_t run_start_ = 0;
    bool run_announced_ = false;
    bool have_pending_ = false;
    Event pending_;
};

}  // namespace uneye

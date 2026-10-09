// Deterministic safety guard: the last stage between the gaze engine (networks + physics layer) and whoever uses the labels.
//
// Principle: simple rules that do not depend on the network, and the network can never override them. When unsure -> SAFE.
//
//   raw sample ----------------------------+
//                                          v
//   GazeEngine (networks) --> Output --> SafetyGuard --> label (Fixation / Saccade / Invalid) + state + reason
//                                                     --> events, each checked against physiology before it is emitted
//
// Three states, documented transitions, hysteresis:
//   NORMAL    the engine's label is used.
//   DEGRADED  the network is not trusted: the guard's OWN classical detector (causal velocity threshold, Engbert-Kliegl style) labels.
//   SAFE      the output is Invalid and NO event is emitted, never a guess.
// Going up (NORMAL -> DEGRADED -> SAFE, or NORMAL -> SAFE) happens at once. Going down is one level at a time and only after
// `recover_samples` consecutive samples whose own verdict is lower (SAFE -> DEGRADED -> NORMAL, never SAFE -> NORMAL directly).
//
// Run-time rules of step(): noexcept, no heap, no I/O, no clock (the caller passes the measured compute time), no randomness, bounded loops.
// Units: degrees and microseconds, like gaze::Sample. Every number in GuardConfig says where it comes from.
#pragma once
#include <array>
#include <cstdint>

#include "uneye/gaze_engine.hpp"

namespace uneye {
namespace safety {

enum class GuardState : uint8_t { Normal = 0, Degraded = 1, Safe = 2 };
enum class Label : uint8_t { Fixation = 0, Saccade, Invalid };

// Why the guard did what it did. One "main" reason per sample (the most severe) + a bit mask of all reasons (bit = 1u << reason).
enum class Reason : uint8_t {
    None = 0,
    // (a) input plausibility
    NonFinite,          // x or y is NaN / inf
    OutOfRange,         // |x| or |y| beyond max_abs_deg
    Saturated,          // an axis is held at the same value beyond saturation_deg (tracker at its range limit)
    Flat,               // both axes bit-identical for flat_samples: frozen signal
    Spike,              // jump faster than max_speed_deg_s from the last valid sample
    TimeBackwards,      // timestamp not increasing
    Gap,                // timestamp step > 1.5 x nominal (samples were lost)
    RateMismatch,       // measured mean sampling interval differs from nominal by more than rate_tolerance
    LongInvalid,        // invalid input for longer than safe_invalid_ms
    // engine / network
    EngineFailed,       // engine health Failed
    NetworkUntrusted,   // engine health Degraded (its network watchdog)
    // (c) confidence / out of distribution
    LowConfidence,      // the network output sits near 0.5 too often
    OutOfDistribution,  // input noise outside the training range
    NoMovement,         // the engine says saccade while the eye does not move, too often (dead / stuck network)
    // (d) timing
    DeadlineMiss,       // compute time of this sample above the budget
    // (b) output plausibility of an event (event dropped)
    EventTooFast,       // peak speed above max_peak_deg_s
    EventTooLarge,      // amplitude above max_amplitude_deg
    EventTooLong,       // duration above max_duration_ms
    EventTooShort,      // duration below min_duration_ms
    EventTooSlow,       // peak speed below min_peak_deg_s: the eye did not really move
    EventRefractory,    // onset closer than refractory_ms to the previous event's offset
    EventAborted,       // the event was interrupted by invalid data, or the state went up while it ran (its source is no longer trusted)
    RepeatedViolations, // violation_degraded dropped events within violation_window -> DEGRADED
    Hysteresis,         // nothing wrong at this sample, but the state is held until recover_samples clean samples have passed
    Count
};
const char* reason_name(Reason r) noexcept;

struct GuardConfig {
    double fs_hz = 1000.0;               // nominal rate (the engine supports ~1 kHz only)
    // (a) input
    double max_abs_deg = 90.0;           // same physical limit as gaze::Config
    double saturation_deg = 30.0;        // training data never exceed |27.9| deg (datasets 1, 2, set A): a value held beyond 30 deg = rail
    int saturation_samples = 5;
    int flat_samples = 50;               // both axes identical for 50 ms. Training data: x alone is identical up to 352 samples (dataset 1),
                                         // y alone up to 15, both together never -> one frozen axis is NOT flagged
    double max_speed_deg_s = 1000.0;     // faster than any saccade (same as gaze::Config)
    double rate_tolerance = 0.1;         // |measured interval / nominal - 1|
    double safe_invalid_ms = 500.0;      // invalid input this long -> SAFE (shorter runs, e.g. blinks, only make the samples Invalid)
    // (c) confidence and out of distribution
    double low_conf_band = 0.2;          // p in [0.5 - band, 0.5 + band] counts as "unsure"
    double low_conf_rate = 0.2;          // moving average (~1 s) of unsure samples above this -> DEGRADED
    double ood_sigma_hi_deg_s = 23.1;    // 2 x p99.5 of the training noise (11.55 deg/s, cpp/scripts/training_range.py) -> DEGRADED
    double nomove_deg_s = 10.0;          // "the eye does not move": 3-sample speed below this (saccades peak at > 20 deg/s even at 0.2 deg)
    double nomove_rate = 0.05;           // moving average (~0.5 s) of "saccade label without movement" above this -> DEGRADED
    // (d) timing
    double budget_us = 1000.0;           // one sample period at 1 kHz
    int deadline_window = 1000;          // misses are counted over the last 1000 samples
    int deadline_degraded = 1, deadline_safe = 10;
    // (b) output physiology (event limits)
    double max_peak_deg_s = 1000.0;
    double min_peak_deg_s = 10.0;        // same value as nomove_deg_s
    double max_amplitude_deg = 40.0;
    double max_duration_ms = 150.0;
    double min_duration_ms = 3.0;
    double refractory_ms = 15.0;         // minimum time from the previous event's offset to the next onset
    int merge_gap_samples = 2;           // a fixation gap this short inside a saccade does not end it (as gaze::Config::merge_gap_samples)
    int violation_window = 2000;         // dropped events counted over the last 2 s
    int violation_degraded = 3;
    // DEGRADED fallback detector (causal Engbert-Kliegl, same idea as the engine's physics layer, own implementation)
    double lambda = 6.0;
    double floor_deg_s = 25.0;
    int min_run = 3;
    // hysteresis
    int recover_samples = 500;           // 0.5 s of lower verdict before going down one level
};

struct GuardEvent {
    int64_t onset = -1, offset = -1;     // sample indices of the guard (0 = first step() call)
    float amplitude_deg = 0, peak_speed_deg_s = 0, duration_ms = 0;
    bool accepted = false;               // false: dropped, see reason
    Reason reason = Reason::None;
};

struct GuardOutput {
    int64_t index = -1;
    GuardState state = GuardState::Normal;
    Label label = Label::Invalid;
    Reason reason = Reason::None;        // the most severe reason of this sample
    uint32_t reasons = 0;                // all reasons of this sample, bit = 1u << Reason
    bool event_ended = false;            // an event ended at this sample; `event` says whether it was emitted (accepted) or dropped
    GuardEvent event;
};

class SafetyGuard {
public:
    explicit SafetyGuard(const GuardConfig& cfg = GuardConfig{}) noexcept;
    // One call per raw sample, after engine.push(raw, engine_out). compute_us: time the caller measured for that push (< 0 = unknown).
    void step(const gaze::Sample& raw, const gaze::Output& engine_out, double compute_us, GuardOutput& out) noexcept;
    void reset() noexcept;
    GuardState state() const noexcept { return state_; }

    struct Stats { int64_t samples = 0, normal = 0, degraded = 0, safe = 0, events_emitted = 0, events_dropped = 0, deadline_misses = 0;
                   std::array<int64_t, static_cast<int>(Reason::Count)> by_reason{}; };
    const Stats& stats() const noexcept { return st_; }

private:
    static constexpr int kNoiseWin = 200;      // samples for the robust noise estimate (as in the engine)
    static constexpr int kRing = 2048;         // power of two, > deadline_window and violation_window (checked in the constructor)

    GuardConfig cfg_;
    GuardState state_ = GuardState::Normal;
    int calm_run_ = 0;                          // consecutive samples whose verdict is below the current state
    int64_t n_ = 0;
    Stats st_;

    // input history
    bool have_valid_ = false, have_t_ = false;
    double last_x_ = 0, last_y_ = 0, last_t_ = 0, dt_ema_ = 0;
    int dt_count_ = 0;
    int same_xy_run_ = 0, sat_run_x_ = 0, sat_run_y_ = 0, invalid_run_ = 0;
    // velocity (3-sample mean of displacements, 0 across gaps) and robust noise
    std::array<double, 3> dxh_{}, dyh_{};
    int dpos_ = 0;
    std::array<float, kNoiseWin> nvx_{}, nvy_{}, scratch_{};
    int ncount_ = 0, npos_ = 0, nsince_ = 0;
    double sigx_ = 20.0, sigy_ = 20.0;
    int above_run_ = 0;
    // confidence
    double unsure_ema_ = 0, nomove_ema_ = 0;
    double prev_raw_x_ = 0, prev_raw_y_ = 0;    // previous finite raw sample (frozen / saturated signal checks)
    bool have_raw_ = false;
    // sliding counts (ring of per-sample bits)
    std::array<uint8_t, kRing> miss_bit_{}, viol_bit_{};
    int miss_count_ = 0, viol_count_ = 0;
    // event in progress (label stream of the guard)
    bool in_event_ = false, suppress_ = false;  // suppress_: an over-long run was already dropped, wait for its end (ev_onset_ then
                                                // marks the start of the current max_duration slice of that run)
    Reason ev_reason_ = Reason::None;           // a problem found while the event runs (refractory, invalid data)
    int64_t ev_onset_ = -1, ev_last_ = -1, last_offset_ = -1;   // ev_last_: last Saccade sample of the event in progress
    int ev_gap_ = 0;
    double ev_x0_ = 0, ev_y0_ = 0, ev_x1_ = 0, ev_y1_ = 0, ev_peak_ = 0;

    void update_noise() noexcept;
    Label fallback_label(bool valid, double speed, double vx, double vy) noexcept;
    void end_event(int64_t offset, Reason forced, GuardOutput& out) noexcept;
    void slide(std::array<uint8_t, kRing>& bits, int& count, int window, bool now) noexcept;
};

}  // namespace safety
}  // namespace uneye

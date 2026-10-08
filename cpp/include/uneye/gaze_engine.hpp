// Deterministic streaming gaze-state engine (see cpp/GAZE_ENGINE.md for the design, the guarantees and what is NOT verified).
//
// One raw gaze sample in -> at every sample out:
//   * the state of the current bin (fixation / saccade / blink / invalid) with a confidence and WHO decided it,
//   * a forecast of the eye position 10 and 20 ms ahead with an uncertainty,
//   * a health flag and reason codes.
// A rolling buffer of bins is revised as later samples arrive (a 10 ms-delayed network, blink margins); snapshot() returns the semantic
// segmentation of the last 100 bins (saccade / microsaccade events with onset, offset, amplitude, peak velocity, provisional or final).
//
// Neural networks (causal TCNs) are only ONE input: a hard-coded physics layer (input guard, velocity-threshold detector, fixed
// landing tables) can veto them ("no movement in the data -> no saccade") and replace them when they fail.
//
// Run-time rules for push() / snapshot(): single owner thread, noexcept, no heap allocation, no I/O, no clock, no randomness, bounded loops.
#pragma once
#include <array>
#include <cstdint>
#include <memory>

#include "uneye/causal_engine.hpp"

namespace uneye {
namespace gaze {

enum class State : uint8_t { Unknown = 0, Fixation, Saccade, Blink, Invalid };
enum class Source : uint8_t { None = 0, Guard, Heuristic, Network, Veto, Override };   // who decided the label of a bin
enum class Health : uint8_t { Ok = 0, Degraded, Failed };
enum class EventClass : uint8_t { Microsaccade, Saccade };                              // amplitude < 1 deg / larger
enum class Mode : uint8_t { Fused = 0, NetworkOnly, HeuristicOnly };                    // the last two are for evaluation only

namespace flag {  // bit mask in Output::flags / BinView::flags
constexpr uint32_t NonFinite = 1u << 0, OutOfRange = 1u << 1, TooFast = 1u << 2, Gap = 1u << 3, TimeBackwards = 1u << 4, Missing = 1u << 5,
                   BlinkMargin = 1u << 6, NnInvalid = 1u << 7, NnStuck = 1u << 8, NnVetoed = 1u << 9, HeuristicOverride = 1u << 10,
                   Reset = 1u << 11, UnsupportedRate = 1u << 12, NnDegraded = 1u << 13, NoForecast = 1u << 14;
}

struct Config {
    double fs_hz = 1000.0;               // nominal sampling rate. The networks and tables are for ~1 kHz; other rates -> Failed + UnsupportedRate
    Mode mode = Mode::Fused;
    // input guard
    double max_abs_deg = 90.0;           // |x|, |y| beyond this are impossible
    double max_speed_deg_s = 1000.0;     // faster jumps than any saccade = artifact (blink / dropout)
    int blink_min_samples = 3;           // invalid run needed to call it a blink (shorter = glitch, label held)
    int blink_margin_ms = 30;            // bins before / after an invalid run are marked as blink margin
    int gap_reset_samples = 50;          // more missing samples than this: all state is reset
    // velocity-threshold detector (Engbert & Kliegl, causal, 3-sample velocity). Defaults tuned on 300 trials of set A (datasets 1+2) only.
    double lambda = 6.0;                 // detection threshold in robust standard deviations
    double lambda_weak = 1.5;            // 'there is some movement' (gate for the networks)
    double lambda_strong = 20.0;         // 'obviously moving' (may override a network that says fixation; rarely active by design)
    double floor_deg_s = 25.0;           // absolute minimum speed of a detection
    double floor_weak_deg_s = 2.5;
    double floor_strong_deg_s = 150.0;
    int min_run = 3;                     // samples above threshold
    // network fusion
    double p_threshold = 0.5;
    int gate_half_window = 4;            // movement evidence searched in [bin-4, bin+4]
    // network watchdog (looks at the raw network output at all times, also while the network is not used)
    double veto_ema_degraded = 0.05;     // share of samples where the network says saccade but the data show no movement -> untrusted
    double veto_ema_recover = 0.02;
    double clear_speed_deg_s = 100.0;    // movement faster than this (and above the detector threshold) is 'clearly a saccade'
    int clear_min_samples = 5;
    double flip_ema_degraded = 0.1;      // label flips per sample of the raw network output (a real network flips ~0.01, noise ~0.5)
    double flip_ema_recover = 0.03;
    double miss_degraded = 0.5;          // moving average over clear movements the network did not see (dead network) -> untrusted
    double miss_recover = 0.2;
    int trust_hold_samples = 1000;       // once untrusted, stay so at least this long
    int stuck_samples = 1000;            // identical network output for this long = stuck
    // segmentation
    int window_bins = 100;               // bins returned by snapshot()
    int min_event_samples = 3;
    int merge_gap_samples = 2;
    double micro_amplitude_deg = 1.0;
};

struct Sample {                          // one raw sample from the eye tracker
    double t_us = -1.0;                  // timestamp in microseconds; < 0 = not available (samples are assumed to be regular)
    double x_deg = 0.0, y_deg = 0.0;
};

struct Interval { double x = 0, y = 0, scale_x = 0, scale_y = 0; bool valid = false; };   // Laplace(location, scale): 80 % interval = +- 1.61 scale
struct Forecast { Interval at10, at20; bool ballistic = false; };                        // ballistic: from the saccade tables, else 'stay + noise'

struct Output {                          // returned by every push()
    int64_t index = -1;                  // sample (bin) number
    State state = State::Unknown;        // state of THIS bin (provisional: the delayed network may revise it 10 ms later)
    float p_saccade = 0;
    Source source = Source::None;
    uint32_t flags = 0;
    Health health = Health::Ok;
    int64_t revised_first = -1, revised_last = -1;   // bins relabeled during this step (-1: none)
    int64_t event_onset = -1;            // onset of the saccade in progress (-1: none)
    Forecast forecast;
};

struct BinView { int64_t index = -1; State state = State::Unknown; float p_saccade = 0; Source source = Source::None; uint32_t flags = 0; uint8_t revisions = 0; bool final = false; };
struct Event {
    int64_t onset = -1, offset = -1;
    bool ongoing = false, provisional = false;   // provisional: part of it is not yet confirmed by the delayed network
    EventClass cls = EventClass::Saccade;
    float amplitude_deg = 0, peak_speed_deg_s = 0, direction_deg = 0;
    float predicted_amplitude_deg = 0;           // ongoing events: amplitude so far + remaining distance from the fixed tables
};
constexpr int kMaxBins = 256, kMaxEvents = 8;
struct Snapshot {                        // semantic segmentation of the last window_bins bins, oldest first
    int64_t newest = -1;
    int n_bins = 0, n_events = 0;
    std::array<BinView, kMaxBins> bins{};
    std::array<Event, kMaxEvents> events{};
};

struct Models {                          // two causal networks; null = run without networks (heuristic only)
    std::unique_ptr<StepEngine> fast;    // label of the current bin
    std::unique_ptr<StepEngine> refine;  // trained with `refine_delay` samples of lookahead: its output at time t describes bin t - refine_delay
    int refine_delay = 10;
};

class GazeEngine {
public:
    GazeEngine(const Config& cfg, Models models);   // init time: may allocate. Everything after it does not.
    ~GazeEngine();
    GazeEngine(const GazeEngine&) = delete;
    GazeEngine& operator=(const GazeEngine&) = delete;

    void push(const Sample& s, Output& out) noexcept;
    void snapshot(Snapshot& out) const noexcept;
    void reset() noexcept;
    int64_t samples() const noexcept;

    struct Stats { int64_t samples = 0, invalid = 0, vetoed = 0, overridden = 0, nn_invalid = 0, resets = 0; double veto_ema = 0;
                   int64_t clear_events = 0, clear_missed = 0, trust_drops_flicker = 0, trust_drops_invalid = 0, trust_drops_stuck = 0, trust_drops_veto = 0, trust_drops_miss = 0; };
    Stats stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace gaze
}  // namespace uneye

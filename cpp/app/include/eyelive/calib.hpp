// Convention calibrator and unit-scale selection (CONTRACT). Implemented by task T08; the Python references are foundation/conventions.py and foundation/scale_select.py.
#pragma once
#include <memory>
#include <utility>
#include <vector>
#include "eyelive/detector.hpp"
#include "eyelive/trace.hpp"
#include "eyelive/types.hpp"

namespace eyelive {

struct CalibParams {                    // how the expert reports events, learned from a few annotated stretches
    float amp_min_deg = 0;              // detected events smaller than this are dropped
    int onset_shift = 0;                // samples; positive = onset earlier
    int offset_shift = 0;               // samples; positive = offset later
    int merge_gap = 0;                  // samples; events closer than this are merged
    bool offset_first = false;          // reserved: offset rule "first inflection" (the acceleration refinement itself is done by the detector)
};

std::vector<Event> apply_calibration(const ITraceView& trace, const std::vector<Event>& detected, const CalibParams& p);
// Grid search maximising 0.5 * (event F1 + kappa) of the calibrated events against the expert events of class `cls` (grid of the Python reference: amp 0, 0.2, 0.4, 0.6, 1.0 deg; shifts -8..8 / -24..8; gap 0, 8, 20)
CalibParams fit_calibration(const ITraceView& trace, const std::vector<Event>& detected, const std::vector<Event>& expert, uint8_t cls, double* score_out = nullptr);

struct ScaleChoice {
    double deg_per_unit = 1.0;                              // chosen scale c: positions are multiplied by c before the degree-trained network
    double agreement = 0;                                   // 0.5 * (event F1 + kappa) between the network and the reference at c
    std::vector<std::pair<double, double>> curve;           // (c, agreement) for every candidate
    bool low_confidence = false;                            // best agreement below 0.4, or best and second best within 0.02 at very different c
};
// Runs `network` on the trace for every candidate scale (logarithmic grid, default 1/200 .. 4 at ratio 1.4) and compares its events with `reference` (the unit-free HMM events of the same trace).
ScaleChoice select_scale(const ITraceView& trace, IDetector& network, const std::vector<Event>& reference, const std::vector<double>& grid = {});

}  // namespace eyelive

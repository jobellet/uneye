// Detector "hmm_accel": the label-free universal minimum-duration HMM + acceleration-zero boundaries (task T04). Batch mode = dsp::hmm_detect + dsp::refine_events exactly; streaming mode keeps incremental velocity /
// detrended velocity (sorted running-median windows), a robust noise from the last 4 s refreshed every 250 ms, and decodes the recent past (Viterbi + refinement) every hop; an event is published when `lag` of data follow it.
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "eyelive/detector.hpp"
#include "eyelive/dsp.hpp"

namespace eyelive {

class HmmDetector final : public IDetector {
 public:
    DetectorInfo info() const override;
    void reset(const SessionInfo& session) override;
    void set_scale(double deg_per_unit) override { scale_ = deg_per_unit > 0 ? deg_per_unit : 1.0; }
    bool set_option(const std::string& key, const std::string& value) override;
    void advance(const ITraceView& trace, int64_t to, std::vector<Event>& final_events, std::vector<Event>& provisional_events) override;
    void detect_all(const ITraceView& trace, std::vector<Event>& events) override;
    const std::string& error() const { return error_; }          // non-empty when the sampling rate has no HMM parameters (events are then not produced)

 private:
    bool select_params(double rate_hz);
    void ingest(double x, double y, bool valid);
    void decode(const ITraceView& trace, std::vector<Event>& out);
    Event make_event(const ITraceView& trace, int64_t on, int64_t off) const;
    double sigma_of_last(const std::vector<double>& ring, int64_t t_end) const;
    size_t at(int64_t i) const { return size_t(i) & mask_; }

    SessionInfo session_; double fs_ = 1000.0, scale_ = 1.0; dsp::HmmParams hmm_; dsp::Refine rf_; bool refine_ = true; double refine_win_ms_ = 15.0, margin_ms_ = 25.0; std::string error_;
    // ring buffers over absolute sample indices
    size_t mask_ = 0; std::vector<double> fx_, fy_, vx_, vy_, dvx_, dvy_; std::vector<float> score_; std::vector<uint8_t> valid_;
    int64_t n_ = 0;                           // samples ingested
    int64_t v_end_ = 0, dv_end_ = 0, score_end_ = 0, anchor_ = 0, last_pub_off_ = -1, last_noise_t_ = 0, hop_count_ = 0;
    int h_ = 50, hop_ = 5, margin_s_ = 25, lag_s_ = 30, keep_s_ = 500;
    double noise_ = 0.0, lx_ = 0.0, ly_ = 0.0; bool have_valid_ = false, init_ = false;
    std::vector<double> sorted_x_, sorted_y_, scratch_a_, scratch_b_;
};

}  // namespace eyelive

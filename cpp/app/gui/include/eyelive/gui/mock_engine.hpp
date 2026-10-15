// MockEngine: an IEngine with synthetic data (gaze from the repository's simulator, its truth events as detections, an ExpertLabels, a fake trainer and a fake trigger log) so that every tab can be developed and
// screenshotted without a tracker or the real engine (`eyelive --mock`). The "recording" grows with the clock while running, as a live session would.
#pragma once
#include <chrono>
#include <memory>
#include <algorithm>
#include <cmath>
#include "eyelive/engine.hpp"
#include "uneye/simulator.hpp"

namespace eyelive { namespace gui {

class MockEngine final : public IEngine {
 public:
    MockEngine() : expert_(3), view_(*this) { uneye::SimConfig c; c.duration_s = 120; c.seed = 7; sim_ = uneye::simulate(c); build(); }

    bool start(const SessionConfig& cfg, std::string* error) override {
        if (cfg.session_path.empty()) { if (error) *error = "no session file path"; return false; } cfg_ = cfg; running_ = true; t0_ = Clock::now(); warn_.clear(); return true;
    }
    void stop() override { if (running_) { frozen_ = visible(); running_ = false; } }
    EngineStats stats() const override {
        EngineStats s; s.running = running_; s.rate_hz_measured = 999.7; s.rate_hz_standard = 1000; s.samples = uint64_t(visible()); s.recorded_s = double(visible()) / 1000.0; s.capacity_s = cfg_.duration_s * 1.05; s.bytes_allocated = recorder_bytes();
        s.cpu_percent = 1.8; s.detect_latency_ms_p50 = 6.1; s.detect_latency_ms_p95 = 11.4; s.publish_jitter_ms_p95 = 0.2; for (const auto& e : events_) if (e.offset < visible()) ++s.events_final; s.events_provisional = s.events_final;
        s.deg_per_unit = cfg_.deg_per_unit > 0 ? cfg_.deg_per_unit : 1.0; s.scale_estimated = cfg_.deg_per_unit == 0; s.scale_agreement = 0.83; s.warning = warn_; return s;
    }
    const ITraceView& trace() const override { return view_; }
    void events_from(uint64_t first, std::vector<Event>& out) const override { const auto all = events(); out.assign(all.begin() + long(std::min<uint64_t>(first, all.size())), all.end()); }
    std::vector<Event> events() const override { std::vector<Event> v; for (const auto& e : events_) if (e.offset < visible()) v.push_back(e); return v; }
    bool forecast_available() const override { return true; }
    void forecast_enable(bool) override {}
    void candidates_set(const std::vector<Candidate>& c) override { candidates_ = c; }
    std::vector<Candidate> candidates_;
    size_t forecasts(int64_t from, int64_t to, std::vector<ForecastSample>& out) const override {        // the future of the simulated trace plus an error that grows during saccades (a stand-in with the right shape, not a prediction)
        out.assign(size_t(std::max<int64_t>(0, to - from)), ForecastSample()); size_t got = 0; const int64_t vis = visible();
        for (int64_t i = std::max<int64_t>(from, 0); i < to && i + 20 < int64_t(sim_.x.size()); ++i) { if (i >= vis) break; ForecastSample f; const bool sacc = sim_.label[size_t(i)] != 0; const double e = sacc ? 0.5 : 0.02; const double w = std::sin(double(i) * 0.37);
            f.valid = true; f.ballistic = sacc; f.x10 = float(sim_.x[size_t(i + 10)] + e * 0.4 * w); f.y10 = float(sim_.y[size_t(i + 10)] - e * 0.4 * w); f.x20 = float(sim_.x[size_t(i + 20)] + e * w); f.y20 = float(sim_.y[size_t(i + 20)] - e * w); f.sx10 = f.sy10 = float(0.3 * e + 0.02); f.sx20 = f.sy20 = float(0.5 * e + 0.03); out[size_t(i - from)] = f; ++got; }
        return got;
    }
    std::vector<DetectorInfo> available_detectors() const override {
        std::vector<DetectorInfo> v; auto add = [&](const char* id, const char* name, int c, bool causal, double d, double p) { DetectorInfo i; i.id = id; i.name = name; i.n_classes = c; i.causal = causal; i.decision_delay_ms = d; i.provisional_latency_ms = p; v.push_back(i); };
        add("hmm_accel", "Label-free HMM + acceleration boundaries", 2, false, 115, -1); add("fast_threshold", "Fast threshold (provisional events)", 2, true, 5, 3); add("unet_1+2+3", "U'n'Eye (datasets 1+2+3)", 2, false, 116, -1);
        add("unet_andersson5", "U'n'Eye (Andersson, 5 classes)", 3, false, 116, -1); add("tcn_causal", "Causal TCN", 2, true, 16, -1); add("bitcn_v2", "BiTCN v2", 2, false, 140, -1); for (auto& n : deployed_) add(("user:" + n).c_str(), ("User model " + n).c_str(), 3, false, 116, -1); return v;
    }
    bool set_detector(const std::string& id, std::string* error) override { for (auto& d : available_detectors()) if (d.id == id) { cfg_.detector_id = id; return true; } if (error) *error = "unknown detector " + id; return false; }
    ExpertLabels& expert() override { return expert_; }
    bool prelabel(const std::string&, int64_t from, int64_t to, std::string*) override { std::vector<Event> ev; for (auto e : events_) if (e.onset >= from && e.offset < to) { e.origin = EventOrigin::Detector; e.confidence = 0.5f + 0.5f * float((e.onset / 7) % 10) / 10.f; ev.push_back(e); } expert_.load_prelabels(ev, from, to); return true; }
    AgreementStats agreement_with_expert(uint8_t cls) const override { return agreement(events(), expert_.events(), visible(), cls); }
    CalibParams fit_calibration_to_expert(uint8_t, std::string*) override { CalibParams p; p.amp_min_deg = 0.2f; p.onset_shift = 2; p.offset_shift = 3; p.merge_gap = 8; return p; }
    bool start_training(const TrainConfig& cfg, std::string*) override { train_cfg_ = cfg; train_t0_ = Clock::now(); training_ = true; return true; }
    TrainProgress training_progress() const override {
        TrainProgress p; if (!training_) return p; const double el = std::chrono::duration<double>(Clock::now() - train_t0_).count(); p.epoch = int(el * 4); p.elapsed_s = el; p.train_loss = 0.45 * std::exp(-el / 6.0) + 0.05; p.val_loss = 0.5 * std::exp(-el / 7.0) + 0.07; p.val_event_f1 = 0.95 - 0.5 * std::exp(-el / 5.0); p.val_kappa = 0.88 - 0.5 * std::exp(-el / 5.0); p.finished = el > 25; p.message = p.finished ? "early stopping" : ""; return p;
    }
    void cancel_training() override { training_ = false; }
    bool deploy_trained_model(const std::string& name, std::string* error) override { if (name.empty()) { if (error) *error = "give the model a name"; return false; } deployed_.push_back(name); return true; }
    void autolearn_set(const AutoLearnConfig& c) override { al_ = c; if (c.enabled) al_t0_ = Clock::now(); }
    AutoLearnStatus autolearn_poll(bool edited) override {
        AutoLearnStatus st; st.enabled = al_.enabled; st.expected_epochs = al_.expected_epochs; if (!al_.enabled) return st; const double now = std::chrono::duration<double>(Clock::now() - al_t0_).count(); if (edited) { al_edit_ = now; al_dirty_ = true; }
        const double since = now - al_edit_; const bool training = al_dirty_ && since > al_.retrain_delay_s && since < al_.retrain_delay_s + 12; if (al_dirty_ && since >= al_.retrain_delay_s + 12) { al_dirty_ = false; ++al_gen_; }
        st.generation = al_gen_; st.labeler = al_gen_ == 0 ? "hmm_accel" : "user:auto_g" + std::to_string(al_gen_); st.training = training; st.labelled_s = double(visible()) / 1000.0; st.val_loss = 0.2; st.val_f1 = 0.9;
        if (training) { st.epoch = int((since - al_.retrain_delay_s) * 4); st.progress = std::min(0.99, st.epoch / 50.0); st.train_loss = 0.2; } else if (al_dirty_) st.retrain_in_s = std::max(0.0, al_.retrain_delay_s - since);
        for (int g = 0; g <= al_gen_; ++g) { AutoGeneration a; a.id = g; a.labeler = g == 0 ? "hmm_accel" : "user:auto_g" + std::to_string(g); a.reviewed_s = 4.0 * (g + 1); a.n_events = 10 + 4 * g; a.f1 = 0.78 + 0.05 * g; a.kappa = 0.7 + 0.05 * g; a.val_f1 = 0.85; a.epochs = 40 + g; st.history.push_back(a); } return st;
    }
    bool autolearn_train_now(std::string*) override { al_dirty_ = true; al_edit_ = -1e9; return true; }
    void set_trigger_rules(const std::vector<TriggerRule>& rules) override { rules_ = rules; }
    std::vector<TriggerLogEntry> trigger_log() const override {
        std::vector<TriggerLogEntry> v; if (rules_.empty()) return v; const double now = std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); int k = 0;
        for (const auto& e : events_) { if (e.offset >= visible()) break; const auto& r = rules_[size_t(k++ % int(rules_.size()))]; double req = now - 5 + double(e.onset) / 1000.0 + r.delay_ms / 1000.0; v.push_back({r.name, req, req + 0.00015 + 0.0001 * (k % 5), e.t_onset, false}); } return v;
    }

 private:
    using Clock = std::chrono::steady_clock;
    struct View final : ITraceView {
        explicit View(const MockEngine& e) : e_(e) { info_.rate_hz = 1000; }
        int64_t size() const override { return e_.visible(); } double t(int64_t i) const override { return double(i) / 1000.0; } float x(int64_t i) const override { return float(e_.sim_.x[size_t(i)]); } float y(int64_t i) const override { return float(e_.sim_.y[size_t(i)]); }
        uint32_t flags(int64_t i) const override { return e_.blink(i) ? kFlagBlink : kFlagValid; } uint8_t label(int64_t i) const override { return e_.blink(i) ? 2 : uint8_t(e_.sim_.label[size_t(i)]); } const SessionInfo& info() const override { return info_; }
        const MockEngine& e_; SessionInfo info_;
    };
    int64_t visible() const { const int64_t total = int64_t(sim_.x.size()); if (!running_ && frozen_ >= 0) return frozen_; if (!running_) return std::min<int64_t>(total, 60000); return std::min<int64_t>(total, int64_t(std::chrono::duration<double>(Clock::now() - t0_).count() * 1000.0) + 20000); }
    bool blink(int64_t i) const { return i >= 31000 && i < 31120; }
    uint64_t recorder_bytes() const { return uint64_t(19.0 * cfg_.duration_s * 1000.0 * 1.05) + 8000000; }
    void build() { for (const auto& e : sim_.events) { Event v; v.onset = e.on; v.offset = e.off; v.t_onset = double(e.on) / 1000.0; v.t_offset = double(e.off) / 1000.0; v.amplitude_deg = float(e.amp); v.peak_speed_deg_s = float(e.amp * 45); v.cls = 1; v.confidence = 0.8f; events_.push_back(v); }
        Event b; b.onset = 31000; b.offset = 31119; b.t_onset = 31.0; b.t_offset = 31.119; b.cls = 2; events_.push_back(b); std::sort(events_.begin(), events_.end(), [](const Event& a, const Event& c) { return a.onset < c.onset; }); prelabel("hmm_accel", 0, 60000, nullptr); }

    uneye::SimTrace sim_; std::vector<Event> events_; ExpertLabels expert_; View view_; SessionConfig cfg_; bool running_ = false; int64_t frozen_ = -1; Clock::time_point t0_ = Clock::now(), train_t0_ = Clock::now(); bool training_ = false; TrainConfig train_cfg_;
    std::vector<std::string> deployed_; std::vector<TriggerRule> rules_; std::string warn_; AutoLearnConfig al_; Clock::time_point al_t0_ = Clock::now(); double al_edit_ = -1e9; bool al_dirty_ = false; int al_gen_ = 0;
};

}}  // namespace eyelive::gui

// IEngine (task T18): source thread -> recorder (the queue) -> analysis thread (detectors, events, state block, triggers); rate measurement, preallocation, blink events, scale selection, hot swap, prelabel, training glue.
#include "eyelive/engine.hpp"
#include "eyelive/engine_hooks.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <thread>
#include <map>
#include "eyelive/fast_detector.hpp"
#include "eyelive/hmm_detector.hpp"
#include "eyelive/net_detectors.hpp"
#include "eyelive/recorder.hpp"
#include "eyelive/state_io.hpp"
#include "eyelive/trigger_clock.hpp"
#include "platform.hpp"

namespace eyelive {
namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t kArrivalRing = 16384;          // arrival times of the last 16384 samples (events are published within a second)

void register_detectors() {                      // referencing the classes also forces the linker to keep the detector libraries, whose static registrars would otherwise be dropped
    auto& r = DetectorRegistry::instance(); const auto ids = r.ids();
    auto has = [&](const char* id) { return std::find(ids.begin(), ids.end(), id) != ids.end(); };
    if (!has("hmm_accel")) r.add("hmm_accel", [](std::string*) { return std::unique_ptr<IDetector>(new HmmDetector()); });
    if (!has("fast_threshold")) r.add("fast_threshold", [](std::string*) { return std::unique_ptr<IDetector>(new FastDetector()); });
    std::string e; auto probe = make_user_model_detector("__none__", &e); (void)probe;                 // pulls in net_detectors.cpp and its registrars (unet_*, tcn_causal, bitcn_v2)
}

class Engine final : public IEngine {
 public:
    explicit Engine(std::unique_ptr<ITrainer> trainer = nullptr) { register_detectors(); expert_ = std::make_unique<ExpertLabels>(3); trig_ = make_trigger_engine(); trainer_ = trainer ? std::move(trainer) : make_trainer(); }
    ~Engine() override { stop(); cancel_training(); }

    bool start(const SessionConfig& cfg, std::string* error) override {
        stop(); std::lock_guard<std::mutex> life(life_mu_);
        auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
        cfg_ = cfg; std::string err; src_ = make_source(cfg.source, &err); if (!src_) return fail("source: " + err);
        // rate: declared, or measured from the first samples (RateEstimator), which are kept and recorded afterwards
        std::vector<Sample> head; double rate = cfg.rate_hz; double measured = 0, standard = 0; uint64_t gaps = 0;
        if (rate <= 0) {
            RateEstimator est; const auto t0 = Clock::now(); std::vector<Sample> buf; buf.reserve(1024);
            while (!est.ready() && std::chrono::duration<double>(Clock::now() - t0).count() < 5.0) { buf.clear(); const size_t n = src_->read(buf, 256, 20); for (auto& s : buf) { est.add(s.t); head.push_back(s); } if (n == 0 && src_->finished()) break; }
            if (!est.ready()) { src_->close(); return fail("source: no samples received to measure the sampling rate (is the tracker streaming to " + src_->description() + "?)"); }
            measured = est.rate_hz(); standard = est.standard_rate_hz(); gaps = est.gaps(); rate = standard > 0 ? standard : measured;
        } else { measured = standard = rate; }
        info_ = SessionInfo(); info_.rate_hz = rate; info_.n_classes = cfg.n_classes; info_.class_names = {"fixation", "saccade", "blink", "class 3", "class 4"}; info_.class_names.resize(size_t(cfg.n_classes));
        scale_ = cfg.deg_per_unit > 0 ? cfg.deg_per_unit : 1.0; scale_estimated_ = false; info_.deg_per_unit = scale_;
        RecorderConfig rc; rc.path = cfg.session_path; rc.rate_hz = rate; rc.duration_s = cfg.duration_s; rc.session = info_;
        rec_ = Recorder::create(rc, &err); if (!rec_) { src_->close(); return fail("cannot allocate the session file: " + err); }
        sbw_ = std::make_unique<StateBlockWriter>(rec_->state_block()); sbw_->init(rate, scale_, cfg.n_classes);
        det_ = DetectorRegistry::instance().create(cfg.detector_id, &err); if (!det_) { src_->close(); rec_.reset(); return fail("detector " + cfg.detector_id + ": " + err); }
        fast_ = DetectorRegistry::instance().create("fast_threshold", nullptr); active_id_ = cfg.detector_id;
        for (auto* d : {det_.get(), fast_.get()}) if (d) { d->reset(info_); d->set_scale(scale_); }
        det_prov_ = det_->info().provisional_latency_ms >= 0; slow_hop_ = std::max<int64_t>(1, int64_t(std::nearbyint(0.005 * rate))); slow_done_ = 0; analysed_ = 0;
        expert_ = std::make_unique<ExpertLabels>(cfg.n_classes); finals_.clear(); provs_.clear(); n_final_ = n_prov_ = 0; last_final_off_ = -1; blink_start_ = -1; scanned_ = 0; id_counter_ = 0; lat_.clear();
        rate_measured_ = measured; rate_standard_ = standard; gaps_seen_ = gaps; recorder_full_ = false; src_done_ = false; scale_started_ = false; scale_confidence_ = 0; scale_low_ = false; warning_.clear();
        stop_ = false; running_ = true; last_cpu_ = plat::cpu_seconds(); last_cpu_wall_ = Clock::now(); cpu_pct_ = 0;
        trig_->start();
        src_thread_ = std::thread([this, head] { source_loop(head); }); ana_thread_ = std::thread([this] { analysis_loop(); });
        return true;
    }

    void stop() override {
        std::lock_guard<std::mutex> life(life_mu_); if (!running_ && !src_thread_.joinable() && !ana_thread_.joinable()) return;
        stop_ = true; if (src_thread_.joinable()) src_thread_.join(); drain_ = true; stop_ana_ = true; if (ana_thread_.joinable()) ana_thread_.join();
        if (swap_thread_.joinable()) swap_thread_.join(); if (scale_thread_.joinable()) scale_thread_.join();
        trig_->stop(); sync_expert(); std::string err; if (rec_) rec_->close(&err); if (src_) src_->close(); running_ = false; stop_ana_ = false; drain_ = false;
    }

    EngineStats stats() const override {
        EngineStats s; std::lock_guard<std::mutex> l(mu_); s.running = running_; s.rate_hz_measured = rate_measured_; s.rate_hz_standard = rate_standard_; s.samples = rec_ ? uint64_t(rec_->size()) : 0;
        s.dropped_packets = src_ ? src_->dropped_packets() : 0; s.gaps = gaps_seen_; s.recorded_s = rec_ && info_.rate_hz > 0 ? double(rec_->size()) / info_.rate_hz : 0; s.capacity_s = rec_ && info_.rate_hz > 0 ? double(rec_->capacity()) / info_.rate_hz : 0;
        s.bytes_allocated = rec_ ? uint64_t(recorder_sizing(info_.rate_hz, cfg_.duration_s).bytes) : 0;
        const double now_cpu = plat::cpu_seconds(); const auto now = Clock::now(); const double dt = std::chrono::duration<double>(now - last_cpu_wall_).count();
        if (dt > 0.5) { cpu_pct_ = 100.0 * (now_cpu - last_cpu_) / dt; last_cpu_ = now_cpu; last_cpu_wall_ = now; } s.cpu_percent = cpu_pct_;
        s.detect_latency_ms_p50 = pct(lat_, 0.5); s.detect_latency_ms_p95 = pct(lat_, 0.95); s.events_final = n_final_; s.events_provisional = n_prov_;
        s.deg_per_unit = scale_; s.scale_estimated = scale_estimated_; s.scale_agreement = scale_confidence_; s.scale_low_confidence = scale_low_; s.warning = recorder_full_ ? "session file full: acquisition stopped" : warning_;
        return s;
    }
    const ITraceView& trace() const override { return rec_ ? static_cast<const ITraceView&>(*rec_) : empty_; }
    std::vector<Event> events() const override { std::lock_guard<std::mutex> l(mu_); std::vector<Event> v = finals_; v.insert(v.end(), provs_.begin(), provs_.end()); return v; }

    std::vector<DetectorInfo> available_detectors() const override {
        std::lock_guard<std::mutex> l(infos_mu_); auto& reg = DetectorRegistry::instance(); std::vector<DetectorInfo> v;
        for (const auto& id : reg.ids()) { auto it = info_cache_.find(id); if (it == info_cache_.end()) { auto d = reg.create(id, nullptr); if (!d) continue; DetectorInfo di = d->info(); di.id = id; it = info_cache_.emplace(id, di).first; } v.push_back(it->second); }
        return v;
    }
    bool set_detector(const std::string& id, std::string* error) override {
        std::string err; auto d = DetectorRegistry::instance().create(id, &err); if (!d) { if (error) *error = err; return false; }
        if (!running_) { cfg_.detector_id = id; active_id_ = id; return true; }
        std::lock_guard<std::mutex> l(swap_mu_); if (swap_thread_.joinable()) swap_thread_.join();          // one swap at a time
        { std::lock_guard<std::mutex> l2(swap_slot_mu_); swap_ready_ = false; swap_det_.reset(); }
        SessionInfo si = info_; si.deg_per_unit = scale_; swap_thread_ = std::thread([this, id, si, det = std::move(d)]() mutable { catch_up(std::move(det), id, si); }); return true;
    }

    ExpertLabels& expert() override { return *expert_; }
    bool prelabel(const std::string& detector_id, int64_t from, int64_t to, std::string* error) override {
        if (!rec_) { if (error) *error = "no recording"; return false; } const int64_t n = rec_->size(); from = std::max<int64_t>(0, from); to = std::min<int64_t>(to, n); if (to - from < 10) { if (error) *error = "range too short"; return false; }
        std::string err; auto d = DetectorRegistry::instance().create(detector_id, &err); if (!d) { if (error) *error = err; return false; }
        SessionInfo si = info_; si.deg_per_unit = scale_; VectorTrace part(si); for (int64_t i = from; i < to; ++i) { Sample s; s.t = rec_->t(i); s.x = rec_->x(i); s.y = rec_->y(i); s.flags = rec_->flags(i); part.push(s); }
        d->reset(si); d->set_scale(scale_); std::vector<Event> ev; d->detect_all(part, ev);
        for (auto& e : ev) { e.onset += from; e.offset += from; e.provisional = false; e.origin = EventOrigin::Detector; }
        { std::vector<Event> blinks; for (const auto& e : blink_events(from, to)) blinks.push_back(e); ev.insert(ev.end(), blinks.begin(), blinks.end()); std::sort(ev.begin(), ev.end(), [](const Event& a, const Event& b) { return a.onset < b.onset; }); }
        expert_->load_prelabels(ev, from, to - 1); return true;
    }
    AgreementStats agreement_with_expert(uint8_t cls) const override {
        const int64_t n = rec_ ? rec_->size() : 0; std::vector<uint8_t> rev; expert_->render(0, n, true, rev); std::vector<Event> det, ex;
        { std::lock_guard<std::mutex> l(mu_); for (const auto& e : finals_) if (e.cls == cls && e.onset < n && rev[size_t(e.onset)] != kIgnoreLabel) det.push_back(e); }
        for (const auto& e : expert_->events()) if (e.cls == cls && e.onset < n && rev[size_t(e.onset)] != kIgnoreLabel) ex.push_back(e);
        return agreement(det, ex, n, cls);
    }
    CalibParams fit_calibration_to_expert(uint8_t cls, std::string* error) override {
        if (!rec_ || rec_->size() < 100) { if (error) *error = "no recording"; return CalibParams(); }
        std::vector<Event> det, ex; { std::lock_guard<std::mutex> l(mu_); for (const auto& e : finals_) if (e.cls == cls) det.push_back(e); } for (const auto& e : expert_->events()) if (e.cls == cls) ex.push_back(e);
        if (ex.empty()) { if (error) *error = "no expert event of this class: review some events first"; return CalibParams(); } return fit_calibration(*rec_, det, ex, cls);
    }

    bool start_training(const TrainConfig& cfg_in, std::string* error) override {
        if (!rec_) { if (error) *error = "no recording"; return false; } if (trainer_->running()) { if (error) *error = "training already running"; return false; }
        sync_expert(); TrainConfig cfg = cfg_in; cfg.deg_per_unit = scale_; if (running_) cfg.n_threads = 1;            // one core while the experiment runs; all cores after the session
        const int64_t n = rec_->size(); std::vector<uint8_t> lab; expert_->render(0, n, true, lab); std::vector<TrainTrace> traces;
        for (int64_t i = 0; i < n;) { if (lab[size_t(i)] == kIgnoreLabel) { ++i; continue; } int64_t e = i; while (e + 1 < n && lab[size_t(e + 1)] != kIgnoreLabel) ++e;
            if (e - i + 1 >= int64_t(0.25 * info_.rate_hz)) { TrainTrace t; t.rate_hz = info_.rate_hz; for (int64_t k = i; k <= e; ++k) { t.x.push_back(float(rec_->x(k) * scale_)); t.y.push_back(float(rec_->y(k) * scale_)); t.labels.push_back(lab[size_t(k)]); } traces.push_back(std::move(t)); } i = e + 1; }
        if (traces.empty()) { if (error) *error = "no reviewed stretch of at least 0.25 s: mark the reviewed ranges in the Review tab"; return false; }
        progress_ = TrainProgress(); const auto cb = [this](const TrainProgress& p) { std::lock_guard<std::mutex> l(train_mu_); progress_ = p; };
        // the trainer's worker thread inherits the priority of the thread that starts it: start it from a helper thread with a lowered priority while acquiring
        const bool low = running_; std::thread h([&, low] { if (low) plat::lower_thread_priority(); trainer_->start(std::move(traces), cfg, cb); }); h.join(); return true;
    }
    TrainProgress training_progress() const override { std::lock_guard<std::mutex> l(train_mu_); return progress_; }
    void cancel_training() override { trainer_->cancel(); }
    bool deploy_trained_model(const std::string& name, std::string* error) override {
        namespace fs = std::filesystem; if (name.empty() || name.find_first_of("/\\:") != std::string::npos) { if (error) *error = "invalid model name"; return false; }
        fs::path dir = fs::path(cfg_.session_path).parent_path() / "models"; std::error_code ec; fs::create_directories(dir, ec); const std::string path = (dir / ("user_" + name + ".bin")).string();
        if (!trainer_->save_model(path, error)) return false;
        const std::string abs = fs::absolute(path).string(); DetectorRegistry::instance().add("user:" + name, [abs](std::string* e) { return make_user_model_detector(abs, e); });
        { std::lock_guard<std::mutex> l(infos_mu_); info_cache_.erase("user:" + name); } return true;
    }
    void set_trigger_rules(const std::vector<TriggerRule>& rules) override { trig_->set_rules(rules); }
    std::vector<TriggerLogEntry> trigger_log() const override { return trig_->log_snapshot(); }

 private:
    static double pct(const std::vector<double>& v, double q) { if (v.empty()) return 0; std::vector<double> c = v; std::sort(c.begin(), c.end()); return c[size_t(q * double(c.size() - 1))]; }
    double arrival(int64_t i) const { return arrival_[size_t(i) % kArrivalRing].load(std::memory_order_relaxed); }

    void source_loop(std::vector<Sample> head) {
        const bool pace = cfg_.source.kind == "sim" && cfg_.source.speed > 0; const size_t chunk = pace ? std::max<size_t>(1, size_t(info_.rate_hz / 1000.0)) : 256; std::vector<Sample> buf; buf.reserve(1024);
        const double start = trigger_host_now_s(); int64_t idx = 0; double t_first = head.empty() ? 0.0 : head.front().t; bool have_first = !head.empty(); bool first_batch = true;
        auto push_all = [&](std::vector<Sample>& b, double arrive_last) {
            const double dt = 1.0 / info_.rate_hz; for (size_t i = 0; i < b.size(); ++i) { arrival_[size_t(idx) % kArrivalRing].store(arrive_last - double(b.size() - 1 - i) * dt, std::memory_order_relaxed); if (!rec_->push(b[i])) { recorder_full_ = true; stop_ = true; return; } ++idx; }
        };
        if (!head.empty()) push_all(head, trigger_host_now_s());
        while (!stop_) {
            buf.clear(); const size_t n = src_->read(buf, chunk, 5);
            if (n == 0) { if (src_->finished()) break; continue; }
            if (!have_first) { t_first = buf.front().t; have_first = true; }
            if (pace) { const double due = start + (buf.back().t - t_first) / cfg_.source.speed; while (!stop_) { const double now = trigger_host_now_s(); if (now >= due) break; std::this_thread::sleep_for(std::chrono::duration<double>(std::min(due - now, 0.002))); } if (stop_) break; }
            push_all(buf, trigger_host_now_s()); (void)first_batch; first_batch = false; ana_cv_.notify_one();
        }
        src_done_ = true;
    }

    std::vector<Event> blink_events(int64_t from, int64_t to) const {          // runs of >= 3 invalid or blink-flagged samples
        std::vector<Event> v; int64_t start = -1; for (int64_t i = from; i <= to; ++i) { const bool bad = i < to && (!rec_->valid(i) || (rec_->flags(i) & kFlagBlink)); if (bad && start < 0) start = i; else if (!bad && start >= 0) { if (i - start >= 3) v.push_back(make_blink(start, i - 1)); start = -1; } } return v;
    }
    Event make_blink(int64_t on, int64_t off) const { Event e; e.onset = on; e.offset = off; e.t_onset = rec_->t(on); e.t_offset = rec_->t(off); e.cls = 2; e.origin = EventOrigin::Detector; e.detector_id = 3; return e; }

    void publish_final(const Event& e) {
        uint64_t id; { std::lock_guard<std::mutex> l(mu_); finals_.push_back(e); ++n_final_; id = ++id_counter_; provs_.erase(std::remove_if(provs_.begin(), provs_.end(), [&](const Event& p) { return p.onset <= e.offset && p.cls == e.cls; }), provs_.end()); }
        rec_->append_event(e); for (int64_t t = e.onset; t <= e.offset && t - e.onset < 20000; ++t) rec_->set_label(t, e.cls);
        sbw_->publish_event(e, id); trig_->on_event(e, arrival(e.onset)); last_final_off_ = std::max(last_final_off_, e.offset);
    }
    void publish_prov(const Event& e) {
        { std::lock_guard<std::mutex> l(mu_); provs_.push_back(e); if (provs_.size() > 8) provs_.erase(provs_.begin()); ++n_prov_; }
        sbw_->publish_event(e, n_prov_); trig_->on_event(e, arrival(e.onset));
    }

    void analysis_loop() {
        std::vector<Event> fin, prov, ffin, fprov; uint64_t loops = 0; fin.reserve(64); prov.reserve(64); ffin.reserve(64); fprov.reserve(64);
        for (;;) {
            const bool last = stop_ana_.load() || (src_done_.load() && analysed_ >= int64_t(rec_->size()));
            const int64_t n = rec_->size(); install_swap(n); apply_scale();
            fin.clear(); prov.clear(); ffin.clear(); fprov.clear();
            if (n > analysed_) {
                if (det_prov_ || n - slow_done_ >= slow_hop_ || last) { det_->advance(*rec_, n, fin, prov); slow_done_ = n; }       // the active detector runs every 5 ms (every wake-up when it publishes provisional events itself); the fast detector at every wake-up
                if (fast_ && active_id_ != "fast_threshold") fast_->advance(*rec_, n, ffin, fprov);
                for (auto& e : fin) if (e.onset > last_final_off_ || e.cls != 1) publish_final(e);
                for (auto& e : prov) { measure_latency(n); publish_prov(e); } for (auto& e : fprov) { measure_latency(n); publish_prov(e); }
                // blink events from runs of invalid samples
                for (int64_t i = scanned_; i < n; ++i) { const bool bad = !rec_->valid(i) || (rec_->flags(i) & kFlagBlink); if (bad && blink_start_ < 0) blink_start_ = i; else if (!bad && blink_start_ >= 0) { if (i - blink_start_ >= 3) publish_final(make_blink(blink_start_, i - 1)); blink_start_ = -1; } } scanned_ = n;
                const int64_t i = n - 1; Sample s; s.t = rec_->t(i); s.x = float(rec_->x(i) * scale_); s.y = float(rec_->y(i) * scale_); s.flags = rec_->flags(i); const double sp = i >= 2 ? std::hypot(rec_->x(i) - rec_->x(i - 2), rec_->y(i) - rec_->y(i - 2)) * scale_ * info_.rate_hz / 2.0 : 0.0;
                sbw_->publish_sample(s, float(sp), rec_->label(i), blink_start_ >= 0 || !provs_empty(), uint64_t(n)); analysed_ = n;
            }
            if (++loops % 20 == 0) { std::lock_guard<std::mutex> l(mu_); sbw_->set_health((src_done_ ? 0u : 0u) | (scale_estimated_ ? 16u : 0u) | (recorder_full_ ? 4u : 0u), uint32_t(src_ ? src_->dropped_packets() : 0), float(pct(lat_, 0.5)), float(pct(lat_, 0.95))); }
            maybe_start_scale(n);
            if (last && analysed_ >= int64_t(rec_->size())) break;
            if (!drain_) { std::unique_lock<std::mutex> lk(ana_mu_); ana_cv_.wait_for(lk, std::chrono::milliseconds(5)); }          // woken by the source thread after every batch of samples: the detection latency is not bounded by a polling period
        }
        if (blink_start_ >= 0 && analysed_ - blink_start_ >= 3) publish_final(make_blink(blink_start_, analysed_ - 1)); blink_start_ = -1;
    }
    bool provs_empty() const { std::lock_guard<std::mutex> l(mu_); return provs_.empty(); }
    void measure_latency(int64_t n) { const double ms = (trigger_host_now_s() - arrival(n - 1)) * 1000.0; std::lock_guard<std::mutex> l(mu_); lat_.push_back(ms); if (lat_.size() > 1024) lat_.erase(lat_.begin(), lat_.begin() + 512); }

    // ---- hot swap: the new detector runs on its own thread over the whole recording so far; the analysis thread installs it when it has caught up and drops the events it already published
    void catch_up(std::unique_ptr<IDetector> d, const std::string& id, SessionInfo si) {
        d->reset(si); d->set_scale(si.deg_per_unit); std::vector<Event> f, p; int64_t k = 0;
        while (!stop_ana_ && !stop_) { const int64_t n = rec_->size(); if (k >= n - 200) break; k = std::min<int64_t>(n, k + 20000); f.clear(); p.clear(); d->advance(*rec_, k, f, p); }
        std::lock_guard<std::mutex> l(swap_slot_mu_); swap_det_ = std::move(d); swap_id_ = id; swap_ready_ = true;
    }
    void install_swap(int64_t n) {                           // (also refreshes det_prov_ / slow_hop_)
        std::unique_ptr<IDetector> d; std::string id; { std::lock_guard<std::mutex> l(swap_slot_mu_); if (!swap_ready_ || !swap_det_) return; d = std::move(swap_det_); id = swap_id_; swap_ready_ = false; }
        std::vector<Event> f, p; det_ = std::move(d); active_id_ = id; det_prov_ = det_->info().provisional_latency_ms >= 0; det_->advance(*rec_, n, f, p); slow_done_ = n; for (auto& e : f) if (e.onset > last_final_off_) publish_final(e); for (auto& e : p) publish_prov(e); analysed_ = n;
    }
    // ---- automatic scale: once >= 20 events exist the network and the unit-free HMM are compared over a grid of scales on the last 60 s (calib.hpp)
    void maybe_start_scale(int64_t n) {
        if (cfg_.deg_per_unit > 0 || scale_started_ || n < int64_t(10.0 * info_.rate_hz)) return; { std::lock_guard<std::mutex> l(mu_); if (n_final_ < 20) return; }
        scale_started_ = true; if (scale_thread_.joinable()) scale_thread_.join(); const int64_t from = std::max<int64_t>(0, n - int64_t(60.0 * info_.rate_hz));
        scale_thread_ = std::thread([this, from, n] {
            SessionInfo si = info_; si.deg_per_unit = 1.0; VectorTrace part(si); for (int64_t i = from; i < n; ++i) { Sample s; s.t = rec_->t(i); s.x = rec_->x(i); s.y = rec_->y(i); s.flags = rec_->flags(i); part.push(s); }
            std::string err; auto net = DetectorRegistry::instance().create("unet_1+2+3", &err); auto ref = DetectorRegistry::instance().create("hmm_accel", nullptr);
            if (!net || !ref) { std::lock_guard<std::mutex> l(mu_); warning_ = "automatic scale not available: " + err; return; }
            ref->reset(si); std::vector<Event> hmm; ref->detect_all(part, hmm); ScaleChoice c = select_scale(part, *net, hmm);
            { std::lock_guard<std::mutex> l(mu_); pending_scale_ = c.deg_per_unit; scale_confidence_ = c.agreement; scale_low_ = c.low_confidence; scale_estimated_ = true; if (c.low_confidence) warning_ = "automatic unit scale has low confidence: enter pixels per degree by hand"; }
        });
    }
    void apply_scale() { double c; { std::lock_guard<std::mutex> l(mu_); c = pending_scale_; pending_scale_ = 0; } if (c > 0) { scale_ = c; det_->set_scale(c); if (fast_) fast_->set_scale(c); sbw_->init(info_.rate_hz, c, info_.n_classes); } }
    void sync_expert() { if (!rec_ || !expert_) return; const int64_t n = rec_->size(); std::vector<uint8_t> out; expert_->render(0, n, true, out); for (int64_t i = 0; i < n; ++i) if (out[size_t(i)] != rec_->expert(i)) rec_->set_expert(i, out[size_t(i)]); }

    // state
    std::mutex ana_mu_; std::condition_variable ana_cv_; bool det_prov_ = false; int64_t slow_hop_ = 5, slow_done_ = 0;
    mutable std::mutex mu_, life_mu_, infos_mu_, swap_mu_, swap_slot_mu_, train_mu_; SessionConfig cfg_; SessionInfo info_; std::atomic<double> scale_{1.0}; bool scale_estimated_ = false; double scale_confidence_ = 0; bool scale_low_ = false; double pending_scale_ = 0; std::string warning_;
    std::unique_ptr<Recorder> rec_; VectorTrace empty_; std::unique_ptr<ISource> src_; std::unique_ptr<ITriggerEngine> trig_; std::unique_ptr<StateBlockWriter> sbw_; std::unique_ptr<IDetector> det_, fast_; std::string active_id_;
    std::unique_ptr<ExpertLabels> expert_; std::unique_ptr<ITrainer> trainer_; TrainProgress progress_;
    std::vector<Event> finals_, provs_; uint64_t n_final_ = 0, n_prov_ = 0, id_counter_ = 0; int64_t last_final_off_ = -1, blink_start_ = -1, scanned_ = 0, analysed_ = 0; std::vector<double> lat_;
    std::atomic<double> arrival_[kArrivalRing]; double rate_measured_ = 0, rate_standard_ = 0; uint64_t gaps_seen_ = 0;
    std::atomic<bool> stop_{false}, stop_ana_{false}, drain_{false}, src_done_{false}, recorder_full_{false}; std::atomic<bool> running_{false}; bool scale_started_ = false;
    std::thread src_thread_, ana_thread_, swap_thread_, scale_thread_; std::unique_ptr<IDetector> swap_det_; std::string swap_id_; bool swap_ready_ = false;
    mutable std::map<std::string, DetectorInfo> info_cache_; mutable double last_cpu_ = 0, cpu_pct_ = 0; mutable Clock::time_point last_cpu_wall_ = Clock::now();
};

}  // namespace

std::unique_ptr<IEngine> make_engine() { return std::make_unique<Engine>(); }
std::unique_ptr<IEngine> make_engine_with_trainer(std::unique_ptr<ITrainer> trainer) { return std::make_unique<Engine>(std::move(trainer)); }

}  // namespace eyelive

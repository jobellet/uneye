// IEngine (task T18): source thread -> recorder (the queue) -> analysis thread (detectors, events, state block, triggers); rate measurement, preallocation, blink events, scale selection, hot swap, prelabel, training glue.
#include "eyelive/candidates.hpp"
#include "eyelive/engine.hpp"
#include "eyelive/engine_hooks.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <mutex>
#include <thread>
#include <map>
#include "eyelive/fast_detector.hpp"
#include "eyelive/hmm_detector.hpp"
#include "eyelive/net_detectors.hpp"
#include "uneye/causal_engine.hpp"
#include "uneye/gaze_engine.hpp"
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
        if (al_job_.valid()) al_job_.wait(); al_job_ = std::future<std::vector<Event>>(); al_committed_ = 0; al_relabel_pending_ = false; al_dirty_ = false; al_training_ = false; al_gen_ = 0; al_labeler_ = al_cfg_.first_labeler; al_spans_.assign(1, {}); al_gen_names_.assign(1, al_cfg_.first_labeler); al_gen_train_.assign(1, AutoGeneration()); al_prev_model_.clear(); al_msg_.clear(); al_history_cache_.clear();
        expert_ = std::make_unique<ExpertLabels>(cfg.n_classes); finals_.clear(); provs_.clear(); prov_open_.clear(); n_final_ = n_prov_ = 0; last_final_off_ = -1; blink_start_ = -1; scanned_ = 0; id_counter_ = 0; lat_.clear();
        rate_measured_ = measured; rate_standard_ = standard; gaps_seen_ = gaps; recorder_full_ = false; src_done_ = false; scale_started_ = false; scale_confidence_ = 0; scale_low_ = false; warning_.clear();
        { std::lock_guard<std::mutex> fl(fc_mu_); gaze_.reset(); fc_ring_.assign(kForecastRing, ForecastSample()); gaze_n_ = 0; fc_from_ = 0; }
        if (std::fabs(rate - 1000.0) < 50.0) {                            // forecast of the eye position: needs the two causal networks of the gaze engine and a 1 kHz stream
            const std::string fast = find_model_file("causal.bin"), refine = find_model_file("tcn_l10.bin");
            if (!fast.empty() && !refine.empty()) try { uneye::gaze::Models m; m.fast = uneye::make_causal_engine(fast); m.refine = uneye::make_causal_engine(refine); m.refine_delay = 10; uneye::gaze::Config gc; gc.fs_hz = rate; std::lock_guard<std::mutex> fl(fc_mu_); gaze_ = std::make_unique<uneye::gaze::GazeEngine>(gc, std::move(m)); } catch (const std::exception&) {}
        }
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
    void events_from(uint64_t first, std::vector<Event>& out) const override { std::lock_guard<std::mutex> l(mu_); out.assign(finals_.begin() + long(std::min<uint64_t>(first, finals_.size())), finals_.end()); out.insert(out.end(), provs_.begin(), provs_.end()); }
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
        sync_expert(); const int64_t n = rec_->size(); std::vector<uint8_t> lab; expert_->render(0, n, true, lab); return start_training_on(cfg_in, lab, n, "no reviewed stretch of at least 0.25 s: mark the reviewed ranges in the Review tab", error);
    }
    // trains on the labelled stretches of `lab` (kIgnoreLabel = not used); positions are multiplied by the scale; one core at low priority while acquiring, all cores after the session
    bool start_training_on(const TrainConfig& cfg_in, const std::vector<uint8_t>& lab, int64_t n, const char* none_msg, std::string* error) {
        TrainConfig cfg = cfg_in; cfg.deg_per_unit = scale_; if (running_) cfg.n_threads = 1; std::vector<TrainTrace> traces;      // a session that records: one core, whatever was asked
        n = std::min<int64_t>(n, int64_t(lab.size()));
        for (int64_t i = 0; i < n;) { if (lab[size_t(i)] == kIgnoreLabel) { ++i; continue; } int64_t e = i; while (e + 1 < n && lab[size_t(e + 1)] != kIgnoreLabel) ++e;
            if (e - i + 1 >= int64_t(0.25 * info_.rate_hz)) { TrainTrace t; t.rate_hz = info_.rate_hz; for (int64_t k = i; k <= e; ++k) { t.x.push_back(float(rec_->x(k) * scale_)); t.y.push_back(float(rec_->y(k) * scale_)); t.labels.push_back(lab[size_t(k)]); } traces.push_back(std::move(t)); } i = e + 1; }
        if (traces.empty()) { if (error) *error = none_msg; return false; }
        { std::lock_guard<std::mutex> l(train_mu_); progress_ = TrainProgress(); } const auto cb = [this](const TrainProgress& p) { std::lock_guard<std::mutex> l(train_mu_); progress_ = p; };
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
    // ================================================================ assisted annotation (AutoLearnConfig in engine.hpp)
    void autolearn_set(const AutoLearnConfig& c) override {
        const bool was = al_cfg_.enabled; al_cfg_ = c; if (c.enabled && !was) { al_labeler_ = c.first_labeler; al_gen_ = 0; al_committed_ = 0; al_relabel_pending_ = false; al_dirty_ = false; al_spans_.assign(1, {}); al_history_.clear(); al_prev_model_.clear(); al_msg_.clear(); al_training_ = false; al_trainings_ = 0; al_gen_names_.assign(1, c.first_labeler); al_gen_train_.assign(1, AutoGeneration()); }
    }
    bool autolearn_train_now(std::string* error) override {
        if (!al_cfg_.enabled || !rec_) { if (error) *error = "assisted annotation is off"; return false; } if (al_training_) { if (error) *error = "a model is already training"; return false; }
        std::string err; const bool ok = al_begin_training(&err); if (!ok && error) *error = err; return ok;
    }
    AutoLearnStatus autolearn_poll(bool edited) override {
        AutoLearnStatus st; st.enabled = al_cfg_.enabled; st.expected_epochs = al_cfg_.expected_epochs; if (!al_cfg_.enabled || !rec_) return st;
        const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); if (edited) { al_last_edit_ = now; al_dirty_ = true; }
        const int64_t n = rec_->size(); const double rate = info_.rate_hz > 0 ? info_.rate_hz : 1000.0;
        // 1. a finished labelling job is applied (on this thread: it owns the expert labels)
        if (al_job_.valid() && al_job_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) al_apply_job(al_job_.get());
        // 2. labelling: new data in blocks, or the windows after the reviewed ones when a new model is ready
        if (!al_job_.valid() && n > 0) {
            const int64_t guard = running_ ? int64_t(0.3 * rate) : 0, chunk = std::max<int64_t>(10, int64_t(al_cfg_.chunk_s * rate)); int64_t from = al_committed_; bool start = false;
            if (al_relabel_pending_) { int64_t rev_end = 0; for (const auto& r : expert_->reviewed()) rev_end = std::max(rev_end, r.second + 1); from = std::max(rev_end, int64_t(0)); start = n - guard - from >= 10; if (!start) al_relabel_pending_ = false; }
            else start = (n - guard - from >= chunk) || (!running_ && n - from >= 10);
            if (start) al_launch_job(from, n, guard);
        }
        // 3. training: after the quiet delay, and the progress of the running one
        const TrainProgress tp = training_progress();
        if (al_training_) { if (tp.finished) al_finish_training(tp); }
        else if (al_dirty_ && now - al_last_edit_ >= al_cfg_.retrain_delay_s && !al_job_.valid()) { std::string err; if (!al_begin_training(&err)) { al_dirty_ = false; al_msg_ = err; } }
        // 4. status
        st.labeler = al_labeler_; st.generation = al_gen_; st.labeling = al_job_.valid(); st.training = al_training_; st.labelled_s = double(al_committed_) / rate; st.message = al_msg_;
        if (al_training_) { st.epoch = tp.epoch; st.train_loss = tp.train_loss; st.val_loss = tp.val_loss; st.val_f1 = tp.val_event_f1; st.progress = std::min(0.99, double(tp.epoch) / std::max(1, al_cfg_.expected_epochs)); }
        else if (al_dirty_) st.retrain_in_s = std::max(0.0, al_cfg_.retrain_delay_s - (now - al_last_edit_));
        if (now - al_stats_time_ > 1.5) { al_stats_time_ = now; al_history_cache_ = al_generation_stats(n); } st.history = al_history_cache_;
        return st;
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
        uint64_t id; { std::lock_guard<std::mutex> l(mu_); finals_.push_back(e); ++n_final_; id = ++id_counter_; provs_.erase(std::remove_if(provs_.begin(), provs_.end(), [&](const Event& p) { return p.onset <= e.offset && p.cls == e.cls; }), provs_.end()); prune_open(); }
        rec_->append_event(e); for (int64_t t = e.onset; t <= e.offset && t - e.onset < 20000; ++t) rec_->set_label(t, e.cls);
        sbw_->publish_event(e, id); trig_->on_event(e, arrival(e.onset)); last_final_off_ = std::max(last_final_off_, e.offset);
    }
    // a provisional event is OPEN until the fast detector closes it (its own final event gives the end and the amplitude) or retracts it: while open its end follows the newest sample, so the label grows with the saccade on screen
    void prune_open() { prov_open_.erase(std::remove_if(prov_open_.begin(), prov_open_.end(), [&](int64_t on) { for (const auto& p : provs_) if (p.onset == on) return false; return true; }), prov_open_.end()); }
    void close_provisional(const Event& f) {
        std::lock_guard<std::mutex> l(mu_); for (size_t k = 0; k < provs_.size(); ++k) if (provs_[k].onset == f.onset) {
            if (f.confidence <= 0.0f) provs_.erase(provs_.begin() + long(k));                                                  // retracted: no real movement followed
            else { provs_[k].offset = f.offset; provs_[k].t_offset = f.t_offset; provs_[k].amplitude_deg = f.amplitude_deg; provs_[k].peak_speed_deg_s = f.peak_speed_deg_s; provs_[k].direction_rad = f.direction_rad; prov_open_.erase(std::remove(prov_open_.begin(), prov_open_.end(), f.onset), prov_open_.end()); }
            break; }
        prune_open();
    }
    void grow_provisionals(int64_t n) {
        std::lock_guard<std::mutex> l(mu_); const int64_t limit = int64_t(0.5 * (info_.rate_hz > 0 ? info_.rate_hz : 1000.0));
        for (size_t k = 0; k < provs_.size();) { const bool open = std::find(prov_open_.begin(), prov_open_.end(), provs_[k].onset) != prov_open_.end(); if (!open) { ++k; continue; }
            if (n - 1 - provs_[k].onset > limit) { provs_.erase(provs_.begin() + long(k)); continue; }                         // never closed within 0.5 s: dropped
            provs_[k].offset = n - 1; provs_[k].t_offset = rec_->t(n - 1); ++k; }
        prune_open();
    }
    void publish_prov_trigger_only(const Event& e) {
        { std::lock_guard<std::mutex> l(mu_); ++n_prov_; } sbw_->publish_event(e, n_prov_); trig_->on_event(e, arrival(e.onset));
    }
    void publish_prov(const Event& e) {
        { std::lock_guard<std::mutex> l(mu_); provs_.push_back(e); prov_open_.push_back(e.onset); if (provs_.size() > 8) provs_.erase(provs_.begin()); ++n_prov_; prune_open(); }
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
                for (auto& e : prov) { measure_latency(n); publish_prov(e); } for (auto& e : fprov) { measure_latency(n); publish_prov_trigger_only(e); }       // the fast detector runs beside the chosen one for the triggers only: what is displayed and recorded is the chosen detector's output
                grow_provisionals(n);
                // blink events from runs of invalid samples
                for (int64_t i = scanned_; i < n; ++i) { const bool bad = !rec_->valid(i) || (rec_->flags(i) & kFlagBlink); if (bad && blink_start_ < 0) blink_start_ = i; else if (!bad && blink_start_ >= 0) { if (i - blink_start_ >= 3) publish_final(make_blink(blink_start_, i - 1)); blink_start_ = -1; } } scanned_ = n;
                feed_gaze(n);
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

    // candidate targets: the landing estimate becomes the chosen candidate (positions in the stream frame, converted to degrees with sc)
    void apply_candidates(const uneye::gaze::Output& o, int64_t i, double sc, ForecastSample& f) const {
        if (!cands_any_.load(std::memory_order_acquire) || o.event_onset < 0) return;
        const int64_t on = i - (o.index - o.event_onset);                                  // recorder index of the onset (the gaze engine counts from its last reset)
        if (on < 1 || on > i) return;
        const double p0x = rec_->x(on - 1) * sc, p0y = rec_->y(on - 1) * sc, cx = rec_->x(i) * sc, cy = rec_->y(i) * sc;
        const double rem = std::hypot(o.forecast.landing.x - cx, o.forecast.landing.y - cy);
        CandidateChoice ch; { std::lock_guard<std::mutex> l(cand_mu_); ch = pick_candidate(p0x, p0y, cx, cy, rem, cands_.data(), cands_.size(), sc, CandidateParams()); if (ch.index >= 0) { const double ms = (rec_->t(i) - rec_->t(on)) * 1000.0; double lx = 0, ly = 0; landing_from_candidate(rec_->x(on - 1), rec_->y(on - 1), cands_[size_t(ch.index)].x, cands_[size_t(ch.index)].y, f.lx, f.ly, ms, lx, ly); f.lx = float(lx); f.ly = float(ly); } }
        if (ch.index >= 0) { f.candidate = ch.index; f.candidate_confidence = float(ch.confidence); }
    }
    void candidates_set(const std::vector<Candidate>& c) override { { std::lock_guard<std::mutex> l(cand_mu_); cands_ = c; } cands_any_.store(!c.empty(), std::memory_order_release);
        if (!running_) { std::lock_guard<std::mutex> l(fc_mu_); gaze_n_ = 0; fc_from_ = INT64_MAX; } }       // a finished session: forecasts already computed are stale (while recording, only the new samples use the new list)
    // pushes sample i through the gaze engine (positions in degrees) and returns its forecast in stream units
    ForecastSample forecast_of(int64_t i, double sc) const {
        uneye::gaze::Sample gs; gs.t_us = rec_->t(i) * 1e6; const bool ok = rec_->valid(i) && std::isfinite(rec_->x(i)) && std::isfinite(rec_->y(i)); gs.x_deg = ok ? rec_->x(i) * sc : std::nan(""); gs.y_deg = ok ? rec_->y(i) * sc : std::nan(""); uneye::gaze::Output o; gaze_->push(gs, o);
        ForecastSample f; if (o.forecast.at10.valid && o.forecast.at20.valid) { f.valid = true; f.ballistic = o.forecast.ballistic; f.x10 = float(o.forecast.at10.x / sc); f.y10 = float(o.forecast.at10.y / sc); f.x20 = float(o.forecast.at20.x / sc); f.y20 = float(o.forecast.at20.y / sc);
            f.sx10 = float(o.forecast.at10.scale_x / sc); f.sy10 = float(o.forecast.at10.scale_y / sc); f.sx20 = float(o.forecast.at20.scale_x / sc); f.sy20 = float(o.forecast.at20.scale_y / sc);
            if (o.forecast.landing.valid) { f.landing = true; f.lx = float(o.forecast.landing.x / sc); f.ly = float(o.forecast.landing.y / sc); apply_candidates(o, i, sc, f); } }
        return f;
    }
    // ---- forecast of the future eye position: every new sample goes through the gaze engine; the forecast is stored (stream units) in a ring read by the GUI
    void feed_gaze(int64_t n) {
        if (!gaze_) return; const double now = std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
        if (now > fc_until_.load()) { gaze_stale_ = true; return; }                                                  // nobody looks at the forecast: no CPU spent on it
        if (gaze_stale_) { std::lock_guard<std::mutex> l(fc_mu_); gaze_->reset(); gaze_n_ = std::max<int64_t>(gaze_n_, n - 3000); fc_from_ = gaze_n_; gaze_stale_ = false; }       // resume on the last 3 s with a fresh state
        if (n <= gaze_n_) return; const double scv = scale_.load(); const double sc = scv > 0 ? scv : 1.0; std::vector<ForecastSample> batch; batch.reserve(size_t(n - gaze_n_)); float lx = 0, ly = 0, lsx = 0, lsy = 0; bool have = false;
        for (int64_t i = gaze_n_; i < n; ++i) { const ForecastSample f = forecast_of(i, sc); if (f.valid) { lx = f.x20; ly = f.y20; lsx = f.sx20; lsy = f.sy20; have = true; } batch.push_back(f); }
        { std::lock_guard<std::mutex> l(fc_mu_); for (size_t k = 0; k < batch.size(); ++k) fc_ring_[size_t(gaze_n_ + int64_t(k)) % kForecastRing] = batch[k]; gaze_n_ = n; }
        if (have && sbw_) sbw_->publish_prediction(float(lx * sc), float(ly * sc), float(lsx * sc), float(lsy * sc), rec_->t(n - 1) + 0.020);       // landing prediction in the state block (degrees like the other fields), valid at t + 20 ms
    }
public:
    void forecast_enable(bool on) override { fc_until_ = on ? std::chrono::duration<double>(Clock::now().time_since_epoch()).count() + 3.0 : 0.0; }
    bool forecast_available() const override { std::lock_guard<std::mutex> l(fc_mu_); return gaze_ != nullptr; }
    size_t forecasts(int64_t from, int64_t to, std::vector<ForecastSample>& out) const override {
        out.assign(size_t(std::max<int64_t>(0, to - from)), ForecastSample()); std::lock_guard<std::mutex> l(fc_mu_); if (!gaze_) return 0; from = std::max<int64_t>(from, 0);
        // a session that is not recording (finished, or reloaded) has no analysis thread feeding the gaze engine: compute the forecasts of the requested range now, once (cached while the range stays inside it)
        if (!running_ && rec_ && to > from && (to > gaze_n_ || from < fc_from_)) { const int64_t N = rec_->size(), hi = std::min<int64_t>(to, N), lo = std::max<int64_t>(0, std::min<int64_t>(from, hi) - 600), lim = std::max<int64_t>(lo, hi - int64_t(kForecastRing) + 1);
            if (hi > lo) { const double scv = scale_.load(), sc = scv > 0 ? scv : 1.0; gaze_->reset(); for (int64_t i = lim; i < hi; ++i) { fc_ring_[size_t(i) % kForecastRing] = forecast_of(i, sc); } gaze_n_ = hi; fc_from_ = lim; } }
        size_t got = 0; for (int64_t i = from; i < to; ++i) if (i >= fc_from_ && i < gaze_n_ && gaze_n_ - i <= int64_t(kForecastRing)) { out[size_t(i - from)] = fc_ring_[size_t(i) % kForecastRing]; got += out[size_t(i - from)].valid; }
        return got;
    }
private:
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

    // ---- assisted annotation helpers (all called on the poll thread)
    void al_launch_job(int64_t from, int64_t n, int64_t guard) {
        const double rate = info_.rate_hz > 0 ? info_.rate_hz : 1000.0; const int64_t ctx = std::min<int64_t>(from, int64_t(1.0 * rate)), a = from - ctx, to = n - guard; if (to - from < 5) return;
        SessionInfo si = info_; si.deg_per_unit = scale_; auto part = std::make_shared<VectorTrace>(si); for (int64_t i = a; i < n; ++i) { Sample sm; sm.t = rec_->t(i); sm.x = rec_->x(i); sm.y = rec_->y(i); sm.flags = rec_->flags(i); part->push(sm); }
        const std::string labeler = al_labeler_; const double sc = scale_; const int gen = al_gen_; al_job_from_ = from; al_job_to_ = to; al_job_gen_ = gen; al_job_relabel_ = al_relabel_pending_;
        al_job_ = std::async(std::launch::async, [part, labeler, sc, a, from, gen, si]() { std::vector<Event> out; std::string err; auto d = DetectorRegistry::instance().create(labeler, &err); if (!d) return out;
            d->reset(si); d->set_scale(sc); std::vector<Event> ev; d->detect_all(*part, ev); ev.erase(std::remove_if(ev.begin(), ev.end(), [](const Event& e) { return e.cls == 2; }), ev.end());
            // blink events (class 2) come from the data, whatever the labeler: runs of >= 3 invalid or blink-flagged samples; a model's own class-2 events are dropped
            int64_t bs = -1; const int64_t np = part->size(); for (int64_t i = 0; i <= np; ++i) { const bool bad = i < np && (!part->valid(i) || (part->flags(i) & kFlagBlink)); if (bad && bs < 0) bs = i; else if (!bad && bs >= 0) { if (i - bs >= 3) { Event b; b.onset = bs; b.offset = i - 1; b.t_onset = part->t(bs); b.t_offset = part->t(i - 1); b.cls = 2; ev.push_back(b); } bs = -1; } }
            for (auto& e : ev) { e.onset += a; e.offset += a; e.provisional = false; e.origin = EventOrigin::Detector; e.model_version = uint32_t(gen); if (e.onset >= from) out.push_back(e); }
            std::sort(out.begin(), out.end(), [](const Event& x, const Event& y) { return x.onset < y.onset; }); return out; });
    }
    void al_apply_job(std::vector<Event> ev) {            // events are the detections whose onset lies in [job_from, job_to): the region replaced is [job_from, max(job_to - 1, last offset)]
        int64_t hi = al_job_to_ - 1; std::vector<Event> keep; for (auto& e : ev) if (e.onset <= hi) { keep.push_back(e); hi = std::max(hi, e.offset); }
        expert_->apply_auto_labels(keep, al_job_from_, hi); if (int(al_spans_.size()) <= al_job_gen_) al_spans_.resize(size_t(al_job_gen_) + 1); al_spans_[size_t(al_job_gen_)].push_back({al_job_from_, hi});
        if (al_job_relabel_) al_relabel_pending_ = false; al_committed_ = std::max(al_committed_, hi + 1); al_stats_time_ = -1e9;
    }
    bool al_begin_training(std::string* err) {
        const int64_t n = std::min<int64_t>(rec_->size(), al_committed_); if (n < int64_t(2.0 * info_.rate_hz)) { if (err) *err = "wait for 2 s of labelled data before training"; return false; }
        std::vector<uint8_t> lab; expert_->render(0, n - 1, false, lab); TrainConfig tc; tc.n_classes = al_cfg_.n_classes; tc.max_seconds_of_data = al_cfg_.max_seconds_of_data; tc.max_epochs = al_cfg_.max_epochs; tc.n_threads = al_cfg_.n_threads; tc.patience = 4; tc.seed = 1 + uint64_t(al_gen_);
        if (!al_prev_model_.empty()) tc.init_model = al_prev_model_; else if (al_cfg_.pretrained_start) { const std::string d = models_dir(); tc.init_model = (d.empty() ? std::string("models") : d) + "/combined.bin"; }
        if (!start_training_on(tc, lab, n, "no labelled data to train on", err)) return false; al_training_ = true; al_dirty_ = false; al_train_t0_ = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); al_msg_.clear(); return true;
    }
    void al_finish_training(const TrainProgress& tp) {
        al_training_ = false; if (tp.failed || tp.cancelled) { al_msg_ = tp.failed ? "training failed: " + tp.message : "training cancelled"; return; }
        const bool adopt = tp.val_event_f1 >= al_cfg_.adopt_min_f1; const std::string name = adopt ? "auto_g" + std::to_string(al_gen_ + 1) : "auto_t" + std::to_string(++al_trainings_); std::string err;
        if (!deploy_trained_model(name, &err)) { al_msg_ = "cannot keep the model: " + err; return; }
        namespace fs = std::filesystem; al_prev_model_ = (fs::absolute(fs::path(cfg_.session_path).parent_path() / "models" / ("user_" + name + ".bin"))).string();      // the next training continues from it, adopted or not
        if (!adopt) { char b[160]; std::snprintf(b, sizeof b, "model not good enough yet (validation F1 %.2f < %.2f): still labelling with %s; it keeps learning from your next corrections", tp.val_event_f1, al_cfg_.adopt_min_f1, al_labeler_.c_str()); al_msg_ = b; return; }
        al_gen_ += 1; al_labeler_ = "user:" + name; al_relabel_pending_ = true; if (int(al_gen_names_.size()) <= al_gen_) al_gen_names_.resize(size_t(al_gen_) + 1); al_gen_names_[size_t(al_gen_)] = al_labeler_;
        if (int(al_gen_train_.size()) <= al_gen_) al_gen_train_.resize(size_t(al_gen_) + 1); AutoGeneration g; g.id = al_gen_; g.labeler = al_labeler_; g.val_f1 = tp.val_event_f1; g.val_kappa = tp.val_kappa; g.epochs = tp.epoch; g.train_s = tp.elapsed_s; al_gen_train_[size_t(al_gen_)] = g;
        if (int(al_spans_.size()) <= al_gen_) al_spans_.resize(size_t(al_gen_) + 1); al_msg_ = "model " + std::to_string(al_gen_) + " ready (" + std::to_string(tp.epoch) + " epochs, validation F1 " + std::to_string(tp.val_event_f1).substr(0, 5) + ")"; al_stats_time_ = -1e9;
    }
    // per labeler: its labels against your final labels, on the windows it labelled AND you reviewed (events and samples are mapped onto the concatenation of those windows)
    std::vector<AutoGeneration> al_generation_stats(int64_t n) const {
        std::vector<AutoGeneration> out; const double rate = info_.rate_hz > 0 ? info_.rate_hz : 1000.0; const auto& rev = expert_->reviewed(); const auto& pre = expert_->prelabels(); const auto& fin = expert_->events();
        for (size_t g = 0; g < al_spans_.size(); ++g) {
            AutoGeneration ag; if (g < al_gen_train_.size()) ag = al_gen_train_[g]; ag.id = int(g); ag.labeler = g < al_gen_names_.size() ? al_gen_names_[g] : std::string();
            std::vector<std::pair<int64_t, int64_t>> R; for (const auto& sp : al_spans_[g]) for (const auto& r : rev) { const int64_t a = std::max(sp.first, r.first), b = std::min(sp.second, r.second); if (a <= b) R.push_back({a, b}); }
            std::sort(R.begin(), R.end()); std::vector<int64_t> base; int64_t tot = 0; for (const auto& r : R) { base.push_back(tot); tot += r.second - r.first + 1; }
            auto remap = [&](const std::vector<Event>& v, bool only_gen) { std::vector<Event> m; for (const auto& e : v) { if (only_gen && e.model_version != uint32_t(g)) continue; for (size_t k = 0; k < R.size(); ++k) if (e.onset >= R[k].first && e.onset <= R[k].second) { Event q = e; q.onset = base[k] + (e.onset - R[k].first); q.offset = std::min(base[k] + (R[k].second - R[k].first), base[k] + (e.offset - R[k].first)); m.push_back(q); break; } } return m; };
            ag.reviewed_s = double(tot) / rate; if (tot > 0) { const auto P = remap(pre, true), E = remap(fin, false); ag.n_events = int(P.size()); const AgreementStats a = agreement(P, E, tot, 1); ag.f1 = a.event_f1; ag.kappa = a.kappa; }
            out.push_back(ag);
        }
        (void)n; return out;
    }

    // state
    AutoLearnConfig al_cfg_; std::string al_labeler_ = "hmm_accel", al_prev_model_, al_msg_; int al_gen_ = 0, al_job_gen_ = 0, al_trainings_ = 0; int64_t al_committed_ = 0, al_job_from_ = 0, al_job_to_ = 0; bool al_relabel_pending_ = false, al_job_relabel_ = false, al_dirty_ = false, al_training_ = false;
    double al_last_edit_ = 0, al_train_t0_ = 0, al_stats_time_ = -1e9; std::future<std::vector<Event>> al_job_; std::vector<std::vector<std::pair<int64_t, int64_t>>> al_spans_; std::vector<std::string> al_gen_names_; std::vector<AutoGeneration> al_gen_train_, al_history_, al_history_cache_;
    mutable std::mutex cand_mu_; std::vector<Candidate> cands_; std::atomic<bool> cands_any_{false};
    static constexpr size_t kForecastRing = 65536; mutable std::mutex fc_mu_; mutable std::unique_ptr<uneye::gaze::GazeEngine> gaze_; mutable std::vector<ForecastSample> fc_ring_; mutable int64_t gaze_n_ = 0, fc_from_ = 0; std::atomic<double> fc_until_{0.0}; bool gaze_stale_ = true;
    std::mutex ana_mu_; std::condition_variable ana_cv_; bool det_prov_ = false; int64_t slow_hop_ = 5, slow_done_ = 0;
    mutable std::mutex mu_, life_mu_, infos_mu_, swap_mu_, swap_slot_mu_, train_mu_; SessionConfig cfg_; SessionInfo info_; std::atomic<double> scale_{1.0}; bool scale_estimated_ = false; double scale_confidence_ = 0; bool scale_low_ = false; double pending_scale_ = 0; std::string warning_;
    std::unique_ptr<Recorder> rec_; VectorTrace empty_; std::unique_ptr<ISource> src_; std::unique_ptr<ITriggerEngine> trig_; std::unique_ptr<StateBlockWriter> sbw_; std::unique_ptr<IDetector> det_, fast_; std::string active_id_;
    std::unique_ptr<ExpertLabels> expert_; std::unique_ptr<ITrainer> trainer_; TrainProgress progress_;
    std::vector<Event> finals_, provs_; std::vector<int64_t> prov_open_; uint64_t n_final_ = 0, n_prov_ = 0, id_counter_ = 0; int64_t last_final_off_ = -1, blink_start_ = -1, scanned_ = 0, analysed_ = 0; std::vector<double> lat_;
    std::atomic<double> arrival_[kArrivalRing]; double rate_measured_ = 0, rate_standard_ = 0; uint64_t gaps_seen_ = 0;
    std::atomic<bool> stop_{false}, stop_ana_{false}, drain_{false}, src_done_{false}, recorder_full_{false}; std::atomic<bool> running_{false}; bool scale_started_ = false;
    std::thread src_thread_, ana_thread_, swap_thread_, scale_thread_; std::unique_ptr<IDetector> swap_det_; std::string swap_id_; bool swap_ready_ = false;
    mutable std::map<std::string, DetectorInfo> info_cache_; mutable double last_cpu_ = 0, cpu_pct_ = 0; mutable Clock::time_point last_cpu_wall_ = Clock::now();
};

}  // namespace

std::unique_ptr<IEngine> make_engine() { return std::make_unique<Engine>(); }
std::unique_ptr<IEngine> make_engine_with_trainer(std::unique_ptr<ITrainer> trainer) { return std::make_unique<Engine>(std::move(trainer)); }

}  // namespace eyelive

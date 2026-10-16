// T18: end-to-end (SimSource -> engine -> events -> session file -> state block), latency of provisional events, 500 / 1000 / 2000 Hz streams, start failures, clean close, detector hot swap, review -> train -> deploy with a stub trainer.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <thread>
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define SANITIZED 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define SANITIZED 1
#endif
#endif
#ifndef SANITIZED
#define SANITIZED 0
#endif
#include "eyelive/engine.hpp"
#include "eyelive/engine_hooks.hpp"
#include "eyelive/frame_mailbox.hpp"
#include "eyelive/net_detectors.hpp"
#include "eyelive/recorder.hpp"
#include "eyelive/sim_source.hpp"
#include "eyelive/state_io.hpp"
#include "eyelive/trigger_clock.hpp"
using namespace eyelive;
namespace fs = std::filesystem;
static int fails = 0;
static const bool kCI = std::getenv("CI") != nullptr;                                  // shared CI runners: timing limits are 5x looser (the measured values are still printed)
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static std::string tmp(const std::string& name) { fs::path d = fs::temp_directory_path() / "eyelive_engine_test"; fs::create_directories(d); fs::path p = d / name; std::error_code ec; fs::remove(p, ec); return p.string(); }
static SessionConfig cfg_sim(const std::string& name, double rate, double speed, const std::string& detector, double duration = 70) {
    SessionConfig c; c.source.kind = "sim"; c.source.rate_hz = rate; c.source.speed = speed; c.source.seed = 1; c.session_path = tmp(name); c.duration_s = duration; c.detector_id = detector; c.rate_hz = 0; c.deg_per_unit = 1.0; return c;
}
static void sleep_ms(double ms) { std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(ms)); }
// wait until `samples` have been recorded and the number of final events has been stable for 400 ms
static void wait_done(IEngine& e, uint64_t samples, double timeout_s = 60) {
    const auto t0 = std::chrono::steady_clock::now(); uint64_t last = ~0ull; auto stable_since = t0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < timeout_s) {
        const auto s = e.stats(); if (s.samples >= samples) { if (s.events_final != last) { last = s.events_final; stable_since = std::chrono::steady_clock::now(); } else if (std::chrono::duration<double>(std::chrono::steady_clock::now() - stable_since).count() > 0.4) return; } sleep_ms(20);
    }
}
static bool found(const std::vector<Event>& ev, const Event& t, uint8_t cls) { for (auto& e : ev) if (e.cls == cls && !e.provisional && e.onset <= t.offset && t.onset <= e.offset) return true; return false; }

// a trainer double: records what it receives, "trains" instantly and writes a copy of an existing model file as the result
class StubTrainer : public ITrainer {
 public:
    explicit StubTrainer(std::string model) : model_(std::move(model)) {}
    void start(std::vector<TrainTrace> traces, const TrainConfig& cfg, ProgressFn progress) override {
        n_traces = traces.size(); for (auto& t : traces) { n_samples += t.x.size(); for (auto l : t.labels) n_ignored += l == kIgnoreLabel; } got = cfg; TrainProgress p; p.epoch = 1; p.finished = true; p.val_event_f1 = 0.9; p.message = "stub"; done_ = true; if (progress) progress(p);
    }
    void cancel() override {}
    bool running() const override { return false; }
    bool save_model(const std::string& path, std::string* error) override {
        if (!done_) { if (error) *error = "stub: nothing trained"; return false; }
        std::error_code ec; fs::copy_file(model_, path, fs::copy_options::overwrite_existing, ec); const std::string js = path.substr(0, path.size() - 4) + ".json"; std::ofstream(js) << "{\"window\": 200, \"classes\": 2, \"deg_per_unit\": 1.0}"; return !ec;
    }
    size_t n_traces = 0, n_samples = 0, n_ignored = 0; TrainConfig got; bool done_ = false;
 private:
    std::string model_;
};

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : "."; set_models_dir(root + "/models");
    // 1. end to end: SimSource (as fast as possible) -> engine (hmm_accel) -> events; the session file reloads; the state block was updated
    {
        auto e = make_engine(); SessionConfig c = cfg_sim("e2e.eyelive", 1000, 0.0, "hmm_accel"); std::string err; CHECK(e->start(c, &err), "start: %s", err.c_str());
        wait_done(*e, 60000); auto st = e->stats(); e->stop(); const auto ev = e->events(); const auto fin = e->stats();
        SimSource sim(c.source); std::string se; sim.open(&se); const auto truth = sim.truth(); int n = 0, hit = 0; for (const auto& t : truth) { if (t.amplitude_deg < 0.3f || t.offset > 60000 - 400) continue; ++n; hit += found(ev, t, 1); }
        std::printf("end to end: %llu samples, rate measured %.1f Hz (standard %.0f), %zu events, recall of the %d truth events >= 0.3 deg: %d (%.1f %%), cpu %.1f %%\n", (unsigned long long)fin.samples, fin.rate_hz_measured, fin.rate_hz_standard, ev.size(), n, hit, n ? 100.0 * hit / n : 0.0, st.cpu_percent);
        CHECK(fin.samples == 60000, "recorded %llu samples", (unsigned long long)fin.samples); CHECK(n > 20 && double(hit) >= 0.95 * n, "recall %d / %d", hit, n);
        auto rd = Recorder::open_readonly(c.session_path, &err); CHECK(rd != nullptr, "reload: %s", err.c_str());
        if (rd) { CHECK(rd->size() == 60000 && rd->event_count() > 20, "reloaded %lld samples, %lld events", (long long)rd->size(), (long long)rd->event_count()); bool ok = true; for (int64_t i = 0; i < rd->size() && ok; ++i) ok = std::fabs(rd->t(i) - double(i) / 1000.0) < 1e-9;
            CHECK(ok, "time stamps changed in the file"); StateBlock sb; CHECK(read_state_block(rd->state_block(), &sb), "state block unreadable"); std::printf("state block: magic %08x, %llu samples, %llu final events, rate %.0f\n", sb.magic, (unsigned long long)sb.n_samples, (unsigned long long)sb.event_counter, sb.rate_hz);
            CHECK(sb.magic == kStateMagic && sb.n_samples == 60000 && sb.event_counter > 20 && std::fabs(sb.rate_hz - 1000.0) < 1, "state block content"); }
        std::ifstream f(c.session_path, std::ios::binary); f.seekg(52); uint32_t flags = 0; f.read(reinterpret_cast<char*>(&flags), 4); CHECK((flags & 1u) == 1u, "file not marked cleanly closed after stop()");
        auto labels = 0; if (rd) for (int64_t i = 0; i < rd->size(); ++i) labels += rd->label(i) == 1; CHECK(labels > 1000, "per-sample detector labels not written (%d)", labels);
    }
    // 2. latency of provisional events: fast_threshold on a paced 1 kHz stream, a rule on every provisional event
    {
        auto e = make_engine(); SessionConfig c = cfg_sim("lat.eyelive", 1000, 1.0, "fast_threshold"); std::string err; CHECK(e->start(c, &err), "start: %s", err.c_str()); TriggerRule r; r.name = "any"; r.any_class = true; r.anchor = TriggerAnchor::OnsetProvisional; r.sink = "null"; e->set_trigger_rules({r});
        sleep_ms(8000); auto st = e->stats(); auto log = e->trigger_log(); e->stop();
        std::printf("provisional events: %llu, triggers fired %zu, detection latency (arrival of the newest sample -> detector output) p50 %.3f ms, p95 %.3f ms, cpu %.1f %%\n", (unsigned long long)st.events_provisional, log.size(), st.detect_latency_ms_p50, st.detect_latency_ms_p95, st.cpu_percent);
        CHECK(st.events_provisional > 3, "only %llu provisional events", (unsigned long long)st.events_provisional); CHECK(!log.empty(), "no trigger fired"); CHECK(st.detect_latency_ms_p95 < (kCI ? 5.0 : 1.0), "p95 detection latency %.3f ms", st.detect_latency_ms_p95);
    }
    // 3. streams at 500 / 1000 / 2000 Hz: no drop (continuous time stamps), cpu
    for (int k = 0; k < 4; ++k) {
        const double rate = k == 0 ? 500.0 : (k == 3 ? 2000.0 : 1000.0); const char* det = k == 1 ? "fast_threshold" : (k == 3 ? "fast_threshold" : "hmm_accel");
        auto e = make_engine(); SessionConfig c = cfg_sim("rate.eyelive", rate, 1.0, det); std::string err; CHECK(e->start(c, &err), "start %g Hz: %s", rate, err.c_str());
        sleep_ms(5000); auto st = e->stats(); e->stop(); auto fin = e->stats(); const auto& tr = e->trace(); bool gap = false; for (int64_t i = 1; i < tr.size() && !gap; ++i) gap = std::fabs((tr.t(i) - tr.t(i - 1)) * rate - 1.0) > 1e-6;
        std::printf("%4.0f Hz %-14s: measured %.1f Hz, %lld samples in 5 s, gaps %s, cpu %.1f %%\n", rate, det, fin.rate_hz_measured, (long long)tr.size(), gap ? "YES" : "none", st.cpu_percent);
        CHECK(!gap, "gap in the recorded stream at %g Hz", rate); CHECK(std::fabs(double(tr.size()) / (5.0 * rate) - 1.0) < 0.2, "%lld samples at %g Hz", (long long)tr.size(), rate); CHECK(SANITIZED || st.cpu_percent < (kCI ? 40.0 : 20.0), "cpu %.1f %% at %g Hz", st.cpu_percent, rate);      // target < 5 % (measured 2-4.5 % on an M1); the hard limit is loose because CI machines are noisy and instrumented builds are slower
    }
    // 4. start failures: a message, no exception, no file left running
    {
        auto e = make_engine(); SessionConfig c = cfg_sim("x.eyelive", 1000, 0.0, "hmm_accel"); c.session_path = "/nonexistent_dir_eyelive/s.eyelive"; std::string err; CHECK(!e->start(c, &err) && !err.empty(), "start in a missing directory must fail"); std::printf("missing directory -> '%s'\n", err.c_str());
        err.clear(); c = cfg_sim("x.eyelive", 1000, 0.0, "no_such_detector"); CHECK(!e->start(c, &err) && !err.empty(), "unknown detector must fail"); std::printf("unknown detector -> '%s'\n", err.c_str());
        err.clear(); c = cfg_sim("x.eyelive", 1000, 0.0, "hmm_accel"); c.source.kind = "udp"; c.source.port = 1; c.rate_hz = 0; c.source.bind_address = "999.1.1.1"; CHECK(!e->start(c, &err) && !err.empty(), "bad udp address must fail"); std::printf("bad udp source -> '%s'\n", err.c_str());
        e->stop();
    }
    // 5. detector hot swap during streaming: no sample and no event lost
    {
        auto e = make_engine(); SessionConfig c = cfg_sim("swap.eyelive", 1000, 4.0, "hmm_accel"); std::string err; CHECK(e->start(c, &err), "start: %s", err.c_str()); sleep_ms(2500); const auto before = e->stats().samples; CHECK(e->set_detector("unet_1+2+3", &err), "swap: %s", err.c_str()); sleep_ms(4500);
        CHECK(!e->set_detector("nope", &err), "swap to an unknown detector must fail"); e->stop(); const auto& tr = e->trace(); const auto ev = e->events(); const int64_t N = tr.size(); bool gap = false; for (int64_t i = 1; i < N && !gap; ++i) gap = std::fabs((tr.t(i) - tr.t(i - 1)) * 1000.0 - 1.0) > 1e-6;
        SimSource sim(c.source); std::string se; sim.open(&se); int n = 0, hit = 0, around = 0, around_hit = 0; for (const auto& t : sim.truth()) { if (t.amplitude_deg < 1.0f || t.offset > N - 1500) continue; ++n; const bool f = found(ev, t, 1); hit += f; if (t.onset > int64_t(before) - 2000 && t.onset < int64_t(before) + 2000) { ++around; around_hit += f; } }
        std::printf("hot swap at sample %llu of %lld: %d of %d truth events >= 1 deg found overall, %d of %d within 2 s of the swap, time gaps %s\n", (unsigned long long)before, (long long)N, hit, n, around_hit, around, gap ? "YES" : "none");
        CHECK(!gap, "samples lost during the swap"); CHECK(n > 5 && double(hit) >= 0.9 * n, "events lost: %d of %d", hit, n); CHECK(around == 0 || double(around_hit) >= 0.8 * around, "events around the swap lost: %d of %d", around_hit, around);
    }
    // 6. review -> train -> deploy with a stub trainer
    {
        auto stub = std::make_unique<StubTrainer>(root + "/models/synthetic.bin"); StubTrainer* sp = stub.get(); auto e = make_engine_with_trainer(std::move(stub)); SessionConfig c = cfg_sim("review.eyelive", 1000, 0.0, "hmm_accel"); std::string err; CHECK(e->start(c, &err), "start: %s", err.c_str());
        wait_done(*e, 60000); e->stop(); const int64_t N = e->trace().size(); CHECK(e->prelabel("hmm_accel", 0, N, &err), "prelabel: %s", err.c_str()); CHECK(!e->expert().events().empty(), "no prelabel loaded");
        CHECK(e->expert().prelabels().size() == e->expert().events().size(), "prelabels differ from events");
        // the expert deletes one event and reviews the first 20 s
        const size_t before = e->expert().events().size(); CHECK(e->expert().remove(0), "remove"); e->expert().mark_reviewed(0, 20000, true); CHECK(e->expert().events().size() == before - 1, "events after remove");
        auto ag = e->agreement_with_expert(1); std::printf("agreement of the detector with the expert on the reviewed 20 s: F1 %.3f kappa %.3f (tp %d fp %d fn %d)\n", ag.event_f1, ag.kappa, ag.tp, ag.fp, ag.fn); CHECK(ag.fp >= 1 && ag.event_f1 > 0.8, "agreement F1 %.3f fp %d", ag.event_f1, ag.fp);
        std::string e2; auto cp = e->fit_calibration_to_expert(1, &e2); (void)cp; CHECK(e2.empty(), "fit_calibration: %s", e2.c_str());
        TrainConfig tc; tc.n_classes = 3; CHECK(e->start_training(tc, &err), "start_training: %s", err.c_str()); CHECK(sp->n_traces >= 1 && sp->n_samples >= 19000, "trainer received %zu traces / %zu samples", sp->n_traces, sp->n_samples);
        CHECK(e->training_progress().finished && e->training_progress().message == "stub", "training progress not forwarded"); CHECK(sp->got.n_threads == 0 || !e->stats().running, "cores");
        CHECK(e->deploy_trained_model("stub_model", &err), "deploy: %s", err.c_str()); bool listed = false; for (auto& d : e->available_detectors()) { listed = listed || d.id == "user:stub_model"; std::printf("  detector %s\n", d.id.c_str()); } CHECK(listed, "deployed model not listed as a detector");
        CHECK(e->prelabel("user:stub_model", 0, 5000, &err), "the deployed model cannot pre-label: %s", err.c_str());
        CHECK(!e->deploy_trained_model("a/b", &err), "bad model name accepted");
        std::printf("review -> train -> deploy: trainer got %zu traces, %zu samples; 'user:stub_model' registered\n", sp->n_traces, sp->n_samples);
    }
    // 7. assisted annotation: the label-free detector labels the data, the expert corrects, a model is trained after the quiet delay and then labels the windows after the reviewed ones
    {
        auto e = make_engine(); SessionConfig c = cfg_sim("auto.eyelive", 1000, 0.0, "hmm_accel"); c.source.seed = 3; std::string err; CHECK(e->start(c, &err), "start: %s", err.c_str()); wait_done(*e, 60000); e->stop(); const int64_t N = e->trace().size();
        AutoLearnConfig ac; ac.enabled = true; ac.retrain_delay_s = 0.3; ac.chunk_s = 2.0; ac.max_epochs = SANITIZED ? 4 : 25; ac.expected_epochs = 25; if (SANITIZED) ac.adopt_min_f1 = 0.0; ac.max_seconds_of_data = SANITIZED ? 8.0 : 30.0; e->autolearn_set(ac);
        auto poll_until = [&](const std::function<bool(const AutoLearnStatus&)>& done, double timeout_s, bool edited_first = false) { AutoLearnStatus st; bool edited = edited_first; const auto t0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < timeout_s) { st = e->autolearn_poll(edited); edited = false; if (done(st)) return st; sleep_ms(20); } return st; };
        auto st = poll_until([&](const AutoLearnStatus& s) { return s.labelled_s >= double(N) / 1000.0 - 0.5; }, 120); std::printf("auto: labelled %.1f s with '%s', %zu events\n", st.labelled_s, st.labeler.c_str(), e->expert().events().size());
        CHECK(st.labeler == "hmm_accel" && st.generation == 0 && e->expert().events().size() > 20, "initial labelling: labeler %s, %zu events", st.labeler.c_str(), e->expert().events().size());
        CHECK(!e->expert().undo(), "automatic labelling must not appear in the undo history");
        // the expert deletes the first two events; their neighbourhood is reviewed (silently, like the Review tab does) and the poll is told about the edit
        const size_t before = e->expert().events().size(); int64_t rev_end = 0; for (int k = 0; k < 2; ++k) { const Event ev = e->expert().events()[0]; rev_end = std::max<int64_t>(rev_end, ev.offset + 500); CHECK(e->expert().remove(0), "remove"); e->expert().mark_reviewed(std::max<int64_t>(0, ev.onset - 500), ev.offset + 500, true, false); }
        CHECK(e->expert().events().size() == before - 2, "events after two deletions");
        bool saw_training = false; int max_epoch = 0; double max_progress = 0; st = poll_until([&](const AutoLearnStatus& s) { saw_training |= s.training; max_epoch = std::max(max_epoch, s.epoch); max_progress = std::max(max_progress, s.progress); return s.generation >= 1 && !s.training; }, SANITIZED ? 400 : 240, true);
        std::printf("auto: after the corrections the labeler is '%s' (generation %d), training seen %d, max epoch %d, progress %.2f, message '%s'\n", st.labeler.c_str(), st.generation, int(saw_training), max_epoch, max_progress, st.message.c_str());
        CHECK(st.generation == 1 && st.labeler == "user:auto_g1", "no model was trained: generation %d, message '%s'", st.generation, st.message.c_str()); CHECK(saw_training && max_epoch >= 1, "the training progress was never visible");
        // the next windows (after the reviewed ones) are now labelled by the model; the deleted events stay deleted; the history lists both labelers
        st = poll_until([&](const AutoLearnStatus& s) { return !s.labeling && s.history.size() >= 2; }, 180); poll_until([&](const AutoLearnStatus& s) { return !s.labeling; }, 120);
        int by_model = 0, by_hmm = 0, deleted_back = 0; for (const auto& p : e->expert().prelabels()) { by_model += p.model_version == 1 && p.onset > rev_end; by_hmm += p.model_version == 0 && p.onset > rev_end; } for (const auto& ev : e->expert().events()) deleted_back += ev.onset < rev_end - 500 && ev.origin == EventOrigin::Detector && ev.model_version == 1;
        std::printf("auto: windows after the reviewed ones: %d labelled by the model, %d still by the label-free detector; history:", by_model, by_hmm); for (auto& g : st.history) std::printf(" [%s reviewed %.1f s, F1 %.2f, %d events]", g.labeler.c_str(), g.reviewed_s, g.f1, g.n_events); std::printf("\n");
        CHECK(by_model > 10 && by_hmm == 0, "windows after the reviewed ones: %d by the model, %d by the detector", by_model, by_hmm); CHECK(deleted_back == 0, "a model re-labelled a reviewed window"); CHECK(e->expert().events().size() > 10, "events after relabelling: %zu", e->expert().events().size());
        CHECK(st.history.size() >= 2 && st.history[0].reviewed_s > 0, "history: %zu generations", st.history.size());
        CHECK(e->expert().undo(), "the expert's last edit can be undone"); 
    }
    // 8. forecast of the eye position: the gaze engine runs on the stream; its +10 / +20 ms forecasts are compared with what the eye really did and with the trivial "stay where you are" forecast
    {
        auto e = make_engine(); e->forecast_enable(true); SessionConfig c = cfg_sim("forecast.eyelive", 1000, 0.0, "hmm_accel"); c.source.seed = 5; std::string err; CHECK(e->start(c, &err), "start: %s", err.c_str()); for (int w = 0; w < 6000 && e->stats().samples < 60000; ++w) { e->forecast_enable(true); sleep_ms(10); } std::vector<ForecastSample> probe; for (int w = 0; w < 3000; ++w) { e->forecast_enable(true); sleep_ms(10); if (e->forecasts(59000, 59001, probe) > 0) break; } e->stop();      // sanitizer builds analyse slowly: keep asking for the forecast until the end of the stream has one
        const auto& tr = e->trace(); const int64_t N = tr.size(); std::vector<ForecastSample> fc; const size_t got = e->forecasts(0, N, fc); CHECK(e->forecast_available(), "forecast not available (models missing?)"); CHECK(got > size_t(0.9 * double(N)), "only %zu of %lld samples have a forecast", got, (long long)N);
        std::vector<double> err10, err20, stay20, err20_sacc, stay20_sacc; for (int64_t i = 500; i + 20 < N; ++i) { const auto& f = fc[size_t(i)]; if (!f.valid) continue; const double e20 = std::hypot(f.x20 - tr.x(i + 20), f.y20 - tr.y(i + 20)), e10 = std::hypot(f.x10 - tr.x(i + 10), f.y10 - tr.y(i + 10)), s20 = std::hypot(tr.x(i) - tr.x(i + 20), tr.y(i) - tr.y(i + 20));
            err10.push_back(e10); err20.push_back(e20); stay20.push_back(s20); if (std::hypot(tr.x(i + 20) - tr.x(i), tr.y(i + 20) - tr.y(i)) > 0.5) { err20_sacc.push_back(e20); stay20_sacc.push_back(s20); } }
        auto med = [](std::vector<double> v) { if (v.empty()) return 0.0; std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end()); return v[v.size() / 2]; }; auto mean = [](const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return v.empty() ? 0.0 : s / v.size(); };
        std::printf("forecast: %zu samples; median error %.4f deg at +10 ms, %.4f deg at +20 ms (stay-where-you-are: %.4f); during large movements (%zu samples) mean error %.3f deg against %.3f for staying\n", err20.size(), med(err10), med(err20), med(stay20), err20_sacc.size(), mean(err20_sacc), mean(stay20_sacc));
        CHECK(med(err20) < 0.2, "median +20 ms error %.3f deg", med(err20)); CHECK(err20_sacc.empty() || mean(err20_sacc) < mean(stay20_sacc), "the forecast is not better than staying during movements (%.3f vs %.3f)", mean(err20_sacc), mean(stay20_sacc));
        auto rd = Recorder::open_readonly(c.session_path, &err); StateBlock sb; if (rd && read_state_block(rd->state_block(), &sb)) { std::printf("state block prediction: valid %d at (%.2f, %.2f) +- (%.3f, %.3f)\n", int(sb.pred_valid), double(sb.pred_x), double(sb.pred_y), double(sb.pred_sigma_x), double(sb.pred_sigma_y)); }
    }
    // 8b. candidate targets: for every detected saccade, 4 candidates (its real landing point and 3 landing points of other saccades); the landing estimate between 10 and 20 ms after the onset must pick the real one
    //     more often than not and be closer to the real landing point than the estimate of the tables alone
    {
        auto e = make_engine(); SessionConfig c = cfg_sim("cand.eyelive", 1000, 0.0, "hmm_accel"); c.source.seed = 5; std::string err; CHECK(e->start(c, &err), "start: %s", err.c_str());
        for (int w = 0; w < 6000 && e->stats().samples < 40000; ++w) sleep_ms(10); e->stop(); const auto& tr = e->trace(); const int64_t N = tr.size(); const auto ev = e->events();
        std::vector<Event> sac; for (const auto& x : ev) if (x.cls == 1 && !x.provisional && x.amplitude_deg > 2.0 && x.onset > 700 && x.offset + 30 < N && x.offset - x.onset > 25) sac.push_back(x);
        std::vector<double> e_tab, e_cand; int chosen = 0, tot = 0; std::vector<ForecastSample> fc;
        for (size_t k = 0; k < sac.size() && k < 60; ++k) {
            const Event& s = sac[k]; const double lx = tr.x(s.offset), ly = tr.y(s.offset);
            for (int pass = 0; pass < 2; ++pass) {
                std::vector<Candidate> cs; if (pass == 1) { Candidate t; t.x = float(lx); t.y = float(ly); cs.push_back(t); for (int j = 1; j <= 3; ++j) { const Event& o = sac[(k + size_t(j) * 7) % sac.size()]; Candidate d; d.x = float(tr.x(o.offset)); d.y = float(tr.y(o.offset)); cs.push_back(d); } }
                e->candidates_set(cs); e->forecast_enable(true); fc.clear(); e->forecasts(s.onset + 10, s.onset + 21, fc);
                for (const auto& f : fc) { if (!f.landing) continue; const double er = std::hypot(f.lx - lx, f.ly - ly); if (pass == 0) e_tab.push_back(er); else { e_cand.push_back(er); ++tot; chosen += f.candidate == 0; } }
            }
        }
        e->candidates_set({});
        auto mean = [](const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return v.empty() ? 0.0 : s / double(v.size()); };
        std::printf("candidates: %zu saccades, true candidate chosen %.1f %% of %d estimates, landing error %.3f deg against %.3f with the tables alone\n", sac.size(), tot ? 100.0 * chosen / tot : 0.0, tot, mean(e_cand), mean(e_tab));
        CHECK(tot > 50, "too few landing estimates (%d)", tot); CHECK(chosen > tot / 2, "the true candidate was chosen only %d of %d times", chosen, tot); CHECK(mean(e_cand) < mean(e_tab), "candidates did not improve the landing estimate (%.3f vs %.3f)", mean(e_cand), mean(e_tab));
    }
    // 8c. stimulus feed: a script shares a frame (saliency peaks become the candidates) and then explicit candidates (priority) through the mailbox
    {
        const std::string fp = (fs::temp_directory_path() / "eyelive_feed_test.frames").string(); FrameMailboxWriter w; std::string err; CHECK(w.create(fp, 64, 64, 3, 3, -20, 15, 20, -15, &err), "mailbox create: %s", err.c_str());
        auto e = make_engine(); StimulusFeedConfig fc; fc.path = fp; fc.saliency = true; CHECK(e->stimulus_feed_open(fc, &err), "feed open: %s", err.c_str());
        std::vector<uint8_t> px(64 * 64 * 3, 110); for (int y = 8; y < 20; ++y) for (int x = 44; x < 56; ++x) { px[(size_t(y) * 64 + size_t(x)) * 3] = 230; px[(size_t(y) * 64 + size_t(x)) * 3 + 1] = 40; px[(size_t(y) * 64 + size_t(x)) * 3 + 2] = 40; }
        for (size_t i = 0; i < px.size(); i += 5) px[i] = uint8_t(px[i] + int((i * 2654435761u >> 11) % 9) - 4); w.push_frame(px.data(), 1.0);
        for (int k = 0; k < 400 && e->candidates().empty(); ++k) sleep_ms(5); auto c = e->candidates(); auto st = e->stimulus_feed_status();
        CHECK(!c.empty() && st.frames >= 1 && st.open, "no saliency candidates from the frame (%zu, frames %llu)", c.size(), (unsigned long long)st.frames);
        if (!c.empty()) { const double ex = -20 + 40.0 * 50 / 64, ey = 15 - 30.0 * 14 / 64; CHECK(std::hypot(c[0].x - ex, c[0].y - ey) < 8.0, "first candidate (%.1f, %.1f), square near (%.1f, %.1f)", c[0].x, c[0].y, ex, ey); }
        w.set_candidates({{1.f, 2.f, 1.f}, {3.f, 4.f, 2.f}, {-5.f, 6.f, 1.f}}, 15.0); for (int k = 0; k < 400 && e->candidates().size() != 3; ++k) sleep_ms(5); c = e->candidates(); st = e->stimulus_feed_status();
        CHECK(c.size() == 3 && c[1].x == 3.f && st.explicit_candidates, "explicit candidates not used (%zu)", c.size()); w.push_frame(px.data(), 2.0); sleep_ms(100); CHECK(e->candidates().size() == 3, "a new frame replaced the explicit candidates");
        std::printf("stimulus feed: %llu frames, %llu candidate updates, last frame age %.3f s\n", (unsigned long long)st.frames, (unsigned long long)st.candidate_updates, st.last_frame_age_s);
        e->stimulus_feed_close(); CHECK(!e->stimulus_feed_status().open, "feed still open"); std::string e2; CHECK(!e->stimulus_feed_open(StimulusFeedConfig{(fs::temp_directory_path() / "nothing_here.frames").string()}, &e2) && !e2.empty(), "missing file accepted"); fs::remove(fp);
    }
    // 9. live labels come from the CHOSEN detector. fast_threshold chosen: an open provisional label whose end follows the newest sample while the saccade happens. hmm_accel chosen: only its own (final) events are listed, 
    //    but the provisional events of the fast detector still reach the triggers (rules anchored on the onset)
    {
        auto e = make_engine(); SessionConfig c = cfg_sim("live.eyelive", 1000, 1.0, "fast_threshold"); std::string err; CHECK(e->start(c, &err), "start: %s", err.c_str()); int64_t max_grown = 0, seen = 0; std::vector<int64_t> onsets; const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 9.0) { const int64_t n = e->trace().size(); for (const auto& ev : e->events()) if (ev.provisional) { if (std::find(onsets.begin(), onsets.end(), ev.onset) == onsets.end()) { onsets.push_back(ev.onset); ++seen; } max_grown = std::max(max_grown, ev.offset - ev.onset); if (ev.offset > n) CHECK(false, "a provisional end beyond the recorded data"); } sleep_ms(2); }
        e->stop(); std::printf("live labels, fast_threshold chosen: %lld provisional events seen while recording, the longest label reached %lld samples before the final event replaced it\n", (long long)seen, (long long)max_grown);
        CHECK(seen >= 3, "only %lld provisional events seen", (long long)seen); CHECK(max_grown >= 8, "the provisional label never grew with the saccade (longest %lld samples)", (long long)max_grown);
    }
    {
        auto e = make_engine(); SessionConfig c = cfg_sim("live2.eyelive", 1000, 1.0, "hmm_accel"); std::string err; CHECK(e->start(c, &err), "start: %s", err.c_str()); TriggerRule r; r.name = "onset"; r.any_class = true; r.anchor = TriggerAnchor::OnsetProvisional; r.sink = "null"; e->set_trigger_rules({r}); size_t prov_listed = 0, finals_listed = 0; const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 9.0) { for (const auto& ev : e->events()) (ev.provisional ? prov_listed : finals_listed) += 1; sleep_ms(5); } const auto log = e->trigger_log(); const auto st = e->stats(); e->stop();
        std::printf("live labels, hmm_accel chosen: provisional entries in the event list %zu, final entries seen %zu; triggers fired on the fast detector's onsets: %zu (%llu provisional events)\n", prov_listed, finals_listed, log.size(), (unsigned long long)st.events_provisional);
        CHECK(prov_listed == 0, "the fast detector's provisional events must not be listed when another detector is chosen (%zu)", prov_listed); CHECK(finals_listed > 0, "no final event listed"); CHECK(!log.empty(), "the onset rule never fired");
    }
    std::error_code ec; fs::remove_all(fs::temp_directory_path() / "eyelive_engine_test", ec);
    std::printf(fails ? "FAILED\n" : "all engine tests passed\n"); return fails ? 1 : 0;
}

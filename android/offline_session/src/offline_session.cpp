// OfflineSession implementation (plan phase 1). Reuses the frozen contracts
// (ExpertLabels, ITrainer, IDetector, ITraceView) and the engine glue patterns
// of cpp/app/engine/src/engine.cpp (blink events from invalid-sample runs,
// reviewed-range training traces with the 0.25 s minimum).
#include "offline_session/offline_session.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <thread>

#include "eyelive/annotation.hpp"         // agreement() is declared in the frozen contract
#include "eyelive/calib.hpp"
#include "eyelive/detector.hpp"
#include "eyelive/hmm_detector.hpp"      // pulls the class so the static registrar survives linking
#include "eyelive/net_detectors.hpp"     // make_user_model_detector (trainer .bin + .json sidecar)
#include "eyelive/trace.hpp"

namespace offline {
namespace {

constexpr double kTargetRate = 1000.0;
constexpr int64_t kReopenSamples = 150;      // a new point mark un-reviews +-150 ms around it
constexpr int64_t kMinReviewedSamples = 250;   // 0.25 s at 1 kHz (engine glue rule)

std::vector<double> resample_linear(const std::vector<double>& in, double from_hz, double to_hz) {
    if (std::fabs(from_hz - to_hz) < 1e-9 || in.empty()) return in;
    const double ratio = from_hz / to_hz;
    const size_t n_out = size_t(std::lround(double(in.size()) / ratio));
    std::vector<double> out(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        const double pos = double(i) * ratio;
        const size_t j = std::min(size_t(pos), in.size() - 1);
        const double frac = pos - double(j);
        const double b = in[std::min(j + 1, in.size() - 1)];
        out[i] = in[j] + frac * (b - in[j]);
    }
    return out;
}

// One loaded trial: positions in degrees at 1 kHz + validity flags.
struct Trial {
    std::vector<float> x, y;
    std::vector<uint8_t> valid;        // 0 = blink / lost sample
    eyelive::ExpertLabels labels{3};
    std::vector<PointMark> points;     // pass-1 marks (never trained on)
};

// Robust per-axis noise scale of the valid samples: median absolute first
// difference / 0.6745 (MAD estimator, immune to the large saccade steps).
double noise_std(const std::vector<float>& v, const std::vector<uint8_t>& valid) {
    std::vector<double> d;
    for (size_t i = 1; i < v.size(); ++i)
        if (valid[i] && valid[i - 1]) d.push_back(std::fabs(double(v[i]) - double(v[i - 1])));
    if (d.size() < 8) return 0.0;
    std::sort(d.begin(), d.end());
    return d[d.size() / 2] / 0.6745;
}

// The trainer's velocity (dataset.cpp) maps any non-finite difference to 0,
// and a constant or linearly interpolated segment reads as a zero-velocity
// block: a blink labelled 2 with such positions collapses 3-class training
// (measured: validation event F1 stays 0.000 whether the blink is labelled 2
// or ignored, while the same data without the blink trains to F1 1.000).
// A real tracker returns noise during a blink, and filling the invalid
// samples of the TRAINING copy with Gaussian noise at 3x the local fixation
// noise makes the blink distinguishable from fixation again (measured:
// val event F1 1.000, kappa 1.000). The raw trace stays untouched for
// display / detection.
void fill_invalid(std::vector<float>& v, const std::vector<uint8_t>& valid, double std_, std::mt19937_64& rng) {
    std::normal_distribution<double> nd(0.0, std_);
    const size_t n = v.size();
    size_t i = 0;
    while (i < n) {
        if (valid[i]) { ++i; continue; }
        size_t j = i;
        while (j < n && !valid[j]) ++j;
        const double c = i > 0 ? double(v[i - 1]) : (j < n ? double(v[j]) : 0.0);
        for (size_t k = i; k < j; ++k) v[k] = float(c + nd(rng));
        i = j;
    }
}

Mark to_mark(const eyelive::Event& e) {
    Mark m; m.onset = e.onset; m.offset = e.offset; m.cls = e.cls;
    m.from_prelabel = e.origin == eyelive::EventOrigin::Detector;
    m.confidence = e.confidence;
    return m;
}

// Blink events (class 2) from runs of >= 3 invalid samples — same rule as
// engine.cpp::blink_events / make_blink (detector_id 3 = the guard).
std::vector<eyelive::Event> blink_events(const Trial& t) {
    std::vector<eyelive::Event> v;
    const int64_t n = int64_t(t.x.size());
    int64_t start = -1;
    for (int64_t i = 0; i <= n; ++i) {
        const bool bad = i < n && !t.valid[size_t(i)];
        if (bad && start < 0) start = i;
        else if (!bad && start >= 0) {
            if (i - start >= 3) {
                eyelive::Event e; e.onset = start; e.offset = i - 1;
                e.t_onset = double(start) / kTargetRate; e.t_offset = double(i - 1) / kTargetRate;
                e.cls = 2; e.origin = eyelive::EventOrigin::Detector; e.detector_id = 3;
                v.push_back(e);
            }
            start = -1;
        }
    }
    return v;
}

}  // namespace

struct OfflineSession::Impl {
    mutable std::mutex mu;
    std::vector<Trial> trials;
    double deg_per_unit = 1.0;
    bool scale_estimated = false;
    Prelabeler prelabeler = Prelabeler::HMM;
    std::string model_bin;                  // ShippedModel / LastTrained
    std::string last_trained;               // set by save_trained_model
    std::vector<int64_t> test_set;
    std::unique_ptr<eyelive::ITrainer> trainer = eyelive::make_trainer();
    std::function<void(const TrainEvent&)> progress_cb;
    std::mutex progress_mu;

    eyelive::SessionInfo session_info() const {
        eyelive::SessionInfo si; si.rate_hz = kTargetRate; si.deg_per_unit = deg_per_unit; si.n_classes = 3;
        si.scale_estimated = scale_estimated;
        return si;
    }

    eyelive::VectorTrace trace_view(const Trial& t) const {
        eyelive::VectorTrace tr(session_info());
        const size_t n = t.x.size();
        for (size_t i = 0; i < n; ++i) {
            eyelive::Sample s; s.t = double(i) / kTargetRate; s.x = t.x[i]; s.y = t.y[i];
            s.flags = t.valid[i] ? eyelive::kFlagValid : eyelive::kFlagBlink;
            tr.push(s);
        }
        return tr;
    }

    std::unique_ptr<eyelive::IDetector> make_prelabeler(std::string* error) const {
        switch (prelabeler) {
            case Prelabeler::None: return nullptr;
            case Prelabeler::HMM: return std::make_unique<eyelive::HmmDetector>();
            case Prelabeler::ShippedModel:
            case Prelabeler::LastTrained: {
                const std::string& bin = prelabeler == Prelabeler::LastTrained && !last_trained.empty() ? last_trained : model_bin;
                if (bin.empty()) { if (error) *error = "no model file given"; return nullptr; }
                auto d = eyelive::make_user_model_detector(bin, error);
                // The app's labels are 0 fixation / 1 saccade / 2 blink and a
                // model's own class-2 events are dropped (blinks come from the
                // data): force the identity class map so a 3-class model's
                // outputs are not remapped by the default3 rule (which is for
                // 5-class models).
                if (d) d->set_option("class_map", "identity");
                return d;
            }
        }
        if (error) *error = "unknown pre-labeler";
        return nullptr;
    }

    // Batch pre-labeling of one trial: detector events + blink events, loaded
    // into ExpertLabels::load_prelabels (provenance kept). Mirrors
    // engine.cpp::prelabel.
    bool prelabel(Trial& t, std::string* error) {
        if (prelabeler == Prelabeler::None) return true;
        std::string err;
        auto det = make_prelabeler(&err);
        if (!det) { if (error) *error = err; return false; }
        const eyelive::SessionInfo si = session_info();
        eyelive::VectorTrace tr = trace_view(t);
        det->reset(si);
        det->set_scale(deg_per_unit);
        std::vector<eyelive::Event> ev;
        det->detect_all(tr, ev);
        for (auto& e : ev) { e.provisional = false; e.origin = eyelive::EventOrigin::Detector; }
        // a model's own class-2 events are dropped; blinks come from the data (engine rule)
        ev.erase(std::remove_if(ev.begin(), ev.end(), [](const eyelive::Event& e) { return e.cls == 2; }), ev.end());
        auto blinks = blink_events(t);
        ev.insert(ev.end(), blinks.begin(), blinks.end());
        std::sort(ev.begin(), ev.end(), [](const eyelive::Event& a, const eyelive::Event& b) { return a.onset < b.onset; });
        const int64_t n = int64_t(t.x.size());
        t.labels.load_prelabels(ev, 0, n - 1);
        return true;
    }

    // Re-runs the current pre-labeler after a training round (phase 3 swap).
    // Uses ExpertLabels::apply_auto_labels: the reviewed ranges and every
    // expert event are never touched, so the user's work is preserved; only
    // the automatic (detector) labels of the unreviewed parts are replaced.
    // The original pre-labels stay in prelabels() (provenance).
    bool re_prelabel(Trial& t, std::string* error) {
        if (prelabeler == Prelabeler::None) return true;
        std::string err;
        auto det = make_prelabeler(&err);
        if (!det) { if (error) *error = err; return false; }
        const eyelive::SessionInfo si = session_info();
        eyelive::VectorTrace tr = trace_view(t);
        det->reset(si);
        det->set_scale(deg_per_unit);
        std::vector<eyelive::Event> ev;
        det->detect_all(tr, ev);
        for (auto& e : ev) { e.provisional = false; e.origin = eyelive::EventOrigin::Detector; }
        ev.erase(std::remove_if(ev.begin(), ev.end(), [](const eyelive::Event& e) { return e.cls == 2; }), ev.end());
        auto blinks = blink_events(t);
        ev.insert(ev.end(), blinks.begin(), blinks.end());
        std::sort(ev.begin(), ev.end(), [](const eyelive::Event& a, const eyelive::Event& b) { return a.onset < b.onset; });
        const int64_t n = int64_t(t.x.size());
        t.labels.apply_auto_labels(ev, 0, n - 1);
        return true;
    }

    // Training traces from the reviewed ranges of the given trials (the engine
    // glue: render(only_reviewed=true), stretches >= 0.25 s, most recent first).
    // Blink / lost samples are filled with noise at 3x the local fixation
    // noise (see fill_invalid); blink samples keep label 2. Blink events at
    // inference come from the invalid-sample runs (engine rule), but the model
    // still needs a learnable class-2 signal or the 3-class problem collapses.
    std::vector<eyelive::TrainTrace> training_traces(const std::vector<int64_t>& trial_idx, double max_seconds) const {
        std::vector<eyelive::TrainTrace> out;
        double used = 0;
        for (auto it = trial_idx.rbegin(); it != trial_idx.rend() && used < max_seconds; ++it) {
            const Trial& t = trials[size_t(*it)];
            std::vector<float> xf = t.x, yf = t.y;
            std::mt19937_64 rng(size_t(*it) * 2 + 1);   // deterministic per trial
            const double sx = std::max(3.0 * noise_std(xf, t.valid), 1e-3);
            const double sy = std::max(3.0 * noise_std(yf, t.valid), 1e-3);
            fill_invalid(xf, t.valid, sx, rng);
            fill_invalid(yf, t.valid, sy, rng);
            std::vector<uint8_t> lab;
            t.labels.render(0, int64_t(t.x.size()) - 1, true, lab);
            for (const auto& r : t.labels.reviewed()) {
                const int64_t b = std::max<int64_t>(0, r.first);
                const int64_t e = std::min<int64_t>(int64_t(t.x.size()) - 1, r.second);
                if (e - b + 1 < kMinReviewedSamples) continue;   // engine glue drops < 0.25 s
                const double secs = double(e - b + 1) / kTargetRate;
                if (secs <= 0 || used >= max_seconds) break;
                const double take = std::min(secs, max_seconds - used);
                const int64_t n = std::max<int64_t>(1, int64_t(take * kTargetRate));
                eyelive::TrainTrace tr; tr.rate_hz = kTargetRate;
                for (int64_t i = b; i < b + n; ++i) {
                    tr.x.push_back(xf[size_t(i)]);
                    tr.y.push_back(yf[size_t(i)]);
                    tr.labels.push_back(lab[size_t(i)]);
                }
                out.push_back(std::move(tr));
                used += double(n) / kTargetRate;
            }
        }
        return out;
    }
};

OfflineSession::OfflineSession() : impl_(std::make_unique<Impl>()) {}
OfflineSession::~OfflineSession() = default;

LoadResult OfflineSession::load(const std::vector<std::vector<double>>& X,
                                const std::vector<std::vector<double>>& Y,
                                const LoadOptions& opt) {
    LoadResult res;
    std::lock_guard<std::mutex> l(impl_->mu);
    if (X.empty() || X.size() != Y.size()) { res.error = "X/Y row counts differ"; return res; }
    const double rate = opt.rate_hz > 0 ? opt.rate_hz : 1000.0;
    res.rate_hz = kTargetRate;
    res.hmm_rate_warning = std::fabs(rate - 500.0) > 25.0 && std::fabs(rate - 1000.0) > 50.0;

    impl_->prelabeler = opt.prelabeler;
    impl_->model_bin = opt.model_bin;

    // cut into 1 s trials at the declared rate, then resample to 1 kHz
    const size_t win = std::max<size_t>(1, size_t(rate));
    impl_->trials.clear();
    for (size_t k = 0; k < X.size(); ++k) {
        if (X[k].size() != Y[k].size()) { res.error = "ragged input matrix"; return res; }
        for (size_t s = 0; s + win <= X[k].size(); s += win) {
            Trial t;
            std::vector<double> xr = resample_linear(
                std::vector<double>(X[k].begin() + s, X[k].begin() + s + win), rate, kTargetRate);
            std::vector<double> yr = resample_linear(
                std::vector<double>(Y[k].begin() + s, Y[k].begin() + s + win), rate, kTargetRate);
            const double scale = opt.deg_per_unit;
            const size_t n = std::min(xr.size(), yr.size());
            t.x.resize(n); t.y.resize(n); t.valid.assign(n, 1);
            for (size_t i = 0; i < n; ++i) {
                t.x[i] = float(xr[i] * scale);
                t.y[i] = float(yr[i] * scale);
                if (!std::isfinite(xr[i]) || !std::isfinite(yr[i])) t.valid[i] = 0;
            }
            impl_->trials.push_back(std::move(t));
        }
    }
    if (impl_->trials.empty()) { res.error = "no complete 1 s trial in the input"; return res; }
    impl_->deg_per_unit = opt.deg_per_unit > 0 ? opt.deg_per_unit : 1.0;
    res.deg_per_unit = impl_->deg_per_unit;

    // optional scale estimation: network (shipped model) vs the unit-free HMM
    if (opt.estimate_scale && !opt.model_bin.empty()) {
        std::string err;
        auto net = eyelive::make_user_model_detector(opt.model_bin, &err);
        if (net) {
            net->set_option("class_map", "identity");
            eyelive::HmmDetector hmm;
            const Trial& t0 = impl_->trials.front();
            eyelive::VectorTrace tr = impl_->trace_view(t0);
            hmm.reset(impl_->session_info());
            std::vector<eyelive::Event> ref;
            hmm.detect_all(tr, ref);
            auto sc = eyelive::select_scale(tr, *net, ref);
            const double old_scale = impl_->deg_per_unit;
            impl_->deg_per_unit = sc.deg_per_unit;
            res.deg_per_unit = sc.deg_per_unit;
            res.scale_estimated = true;
            impl_->scale_estimated = true;
            // re-apply the scale to every trial
            for (auto& t : impl_->trials)
                for (size_t i = 0; i < t.x.size(); ++i) {
                    t.x[i] = float(double(t.x[i]) / old_scale * sc.deg_per_unit);
                    t.y[i] = float(double(t.y[i]) / old_scale * sc.deg_per_unit);
                }
        }
    }

    // pre-label every trial
    int64_t total = 0;
    for (auto& t : impl_->trials) {
        std::string err;
        if (!impl_->prelabel(t, &err)) { res.prelabeler_error = err; break; }
        total += int64_t(t.x.size());
        for (const auto& e : t.labels.prelabels()) res.prelabels.push_back(to_mark(e));
    }
    res.n_samples = total;
    res.ok = true;
    return res;
}

bool OfflineSession::is_loaded() const { std::lock_guard<std::mutex> l(impl_->mu); return !impl_->trials.empty(); }
int64_t OfflineSession::n_samples() const { std::lock_guard<std::mutex> l(impl_->mu); int64_t n = 0; for (auto& t : impl_->trials) n += int64_t(t.x.size()); return n; }
int64_t OfflineSession::n_trials() const { std::lock_guard<std::mutex> l(impl_->mu); return int64_t(impl_->trials.size()); }
double OfflineSession::rate_hz() const { return kTargetRate; }
double OfflineSession::deg_per_unit() const { std::lock_guard<std::mutex> l(impl_->mu); return impl_->deg_per_unit; }

bool OfflineSession::save_labels(const std::string& path, std::string* error) const {
    std::lock_guard<std::mutex> l(impl_->mu);
    if (impl_->trials.empty()) { if (error) *error = "no recording loaded"; return false; }
    // ExpertLabels::save is per instance; the app saves one file per trial.
    for (size_t k = 0; k < impl_->trials.size(); ++k) {
        const std::string p = impl_->trials.size() == 1 ? path : path + ".trial" + std::to_string(k);
        if (!impl_->trials[k].labels.save(p, error)) return false;
    }
    return true;
}

bool OfflineSession::load_labels(const std::string& path, std::string* error) {
    std::lock_guard<std::mutex> l(impl_->mu);
    if (impl_->trials.empty()) { if (error) *error = "no recording loaded"; return false; }
    for (size_t k = 0; k < impl_->trials.size(); ++k) {
        const std::string p = impl_->trials.size() == 1 ? path : path + ".trial" + std::to_string(k);
        if (!impl_->trials[k].labels.load(p, error)) return false;
    }
    return true;
}

void OfflineSession::trace(int64_t trial, int64_t from, int64_t to, std::vector<float>* x, std::vector<float>* y) const {
    std::lock_guard<std::mutex> l(impl_->mu);
    x->clear(); y->clear();
    if (trial < 0 || size_t(trial) >= impl_->trials.size()) return;
    const Trial& t = impl_->trials[size_t(trial)];
    const int64_t n = int64_t(t.x.size());
    from = std::max<int64_t>(0, from); to = std::min<int64_t>(n - 1, to);
    for (int64_t i = from; i <= to; ++i) { x->push_back(t.x[size_t(i)]); y->push_back(t.y[size_t(i)]); }
}

void OfflineSession::labels(int64_t trial, int64_t from, int64_t to, bool only_reviewed, std::vector<uint8_t>* out) const {
    std::lock_guard<std::mutex> l(impl_->mu);
    out->clear();
    if (trial < 0 || size_t(trial) >= impl_->trials.size()) return;
    const Trial& t = impl_->trials[size_t(trial)];
    const int64_t n = int64_t(t.x.size());
    t.labels.render(from, std::min<int64_t>(n - 1, to), only_reviewed, *out);
}

void OfflineSession::add_point_mark(int64_t trial, int64_t at, uint8_t cls) {
    std::lock_guard<std::mutex> l(impl_->mu);
    if (trial < 0 || size_t(trial) >= impl_->trials.size() || cls == 0 || cls >= eyelive::kMaxClasses) return;
    Trial& t = impl_->trials[size_t(trial)];
    if (at < 0 || at >= int64_t(t.x.size())) return;
    t.points.push_back({at, cls});
    // a new mark in an already reviewed stretch: that stretch has to be refined again before it can train
    t.labels.mark_reviewed(std::max<int64_t>(0, at - kReopenSamples), std::min<int64_t>(int64_t(t.x.size()) - 1, at + kReopenSamples), false, false);
}

bool OfflineSession::remove_point_mark(int64_t trial, int64_t at) {
    std::lock_guard<std::mutex> l(impl_->mu);
    if (trial < 0 || size_t(trial) >= impl_->trials.size()) return false;
    Trial& t = impl_->trials[size_t(trial)];
    int64_t best = -1; int64_t bestd = 6;
    for (size_t i = 0; i < t.points.size(); ++i) {
        const int64_t d = std::llabs(t.points[i].at - at);
        if (d < bestd) { bestd = d; best = int64_t(i); }
    }
    if (best < 0) return false;
    t.points.erase(t.points.begin() + best);
    return true;
}

void OfflineSession::clear_point_marks(int64_t trial) {
    std::lock_guard<std::mutex> l(impl_->mu);
    if (trial < 0 || size_t(trial) >= impl_->trials.size()) return;
    impl_->trials[size_t(trial)].points.clear();
}

std::vector<PointMark> OfflineSession::point_marks(int64_t trial) const {
    std::lock_guard<std::mutex> l(impl_->mu);
    if (trial < 0 || size_t(trial) >= impl_->trials.size()) return {};
    return impl_->trials[size_t(trial)].points;
}

int64_t OfflineSession::snap_to_prelabel(int64_t trial, int64_t at, int64_t tolerance) const {
    std::lock_guard<std::mutex> l(impl_->mu);
    if (trial < 0 || size_t(trial) >= impl_->trials.size()) return -1;
    const Trial& t = impl_->trials[size_t(trial)];
    int64_t best = -1, bestd = tolerance + 1;
    for (const auto& e : t.labels.prelabels()) {
        const int64_t d = std::llabs(e.onset - at);
        if (d <= tolerance && d < bestd) { bestd = d; best = e.onset; }
    }
    return best;
}

bool OfflineSession::refine_mark(int64_t trial, int64_t at, int64_t onset, int64_t offset, uint8_t cls) {
    std::lock_guard<std::mutex> l(impl_->mu);
    if (trial < 0 || size_t(trial) >= impl_->trials.size()) return false;
    Trial& t = impl_->trials[size_t(trial)];
    const int64_t n = int64_t(t.x.size());
    onset = std::max<int64_t>(0, onset); offset = std::min<int64_t>(n - 1, offset);
    if (cls == 0 || cls >= eyelive::kMaxClasses || onset > offset) return false;
    // remove the expert event covering `at` (the mark being refined), then add the refined one
    for (size_t i = 0; i < t.labels.events().size(); ++i) {
        const auto& e = t.labels.events()[i];
        if (e.onset <= at && at <= e.offset) { t.labels.remove(i); break; }
    }
    if (!t.labels.add(onset, offset, cls)) return false;
    // the point mark is consumed by the refinement
    int64_t best = -1, bestd = 6;
    for (size_t i = 0; i < t.points.size(); ++i) {
        const int64_t d = std::llabs(t.points[i].at - at);
        if (d < bestd) { bestd = d; best = int64_t(i); }
    }
    if (best >= 0) t.points.erase(t.points.begin() + best);
    return true;
}

bool OfflineSession::delete_mark(int64_t trial, int64_t at) {
    std::lock_guard<std::mutex> l(impl_->mu);
    if (trial < 0 || size_t(trial) >= impl_->trials.size()) return false;
    Trial& t = impl_->trials[size_t(trial)];
    for (size_t i = 0; i < t.labels.events().size(); ++i) {
        const auto& e = t.labels.events()[i];
        if (e.onset <= at && at <= e.offset) return t.labels.remove(i);
    }
    return remove_point_mark(trial, at);
}

void OfflineSession::mark_epoch_reviewed(int64_t trial, int64_t from, int64_t to) {
    std::lock_guard<std::mutex> l(impl_->mu);
    if (trial < 0 || size_t(trial) >= impl_->trials.size()) return;
    Trial& t = impl_->trials[size_t(trial)];
    const int64_t n = int64_t(t.x.size());
    // reviewed only when scanned AND all its marks are refined (plan rule)
    bool all_refined = true;
    for (const auto& p : t.points) if (p.at >= from && p.at <= to) { all_refined = false; break; }
    if (!all_refined) return;
    // Automatic (detector) events the human did not confirm in a stretch he has seen are not ground truth: drop them,
    // so that a reviewed stretch holds only his own events (a stretch without event is a true "no event" stretch).
    {
        const int64_t f = std::max<int64_t>(0, from), e = std::min<int64_t>(n - 1, to);
        for (size_t i = t.labels.events().size(); i-- > 0;) {
            const auto& ev = t.labels.events()[i];
            if (ev.origin == eyelive::EventOrigin::Detector && ev.onset >= f && ev.offset <= e) t.labels.remove(i);
        }
    }
    t.labels.mark_reviewed(std::max<int64_t>(0, from), std::min<int64_t>(n - 1, to), true, false);
}

eyelive::ExpertLabels& OfflineSession::expert() {
    // The UI edits one trial at a time; the facade exposes the trial being
    // labeled. Kotlin passes the trial index through the JNI edit calls below,
    // which operate on trials[trial] directly.
    std::lock_guard<std::mutex> l(impl_->mu);
    return impl_->trials.front().labels;
}
const eyelive::ExpertLabels& OfflineSession::expert() const {
    std::lock_guard<std::mutex> l(impl_->mu);
    return impl_->trials.front().labels;
}

std::vector<Mark> OfflineSession::events(int64_t trial) const {
    std::lock_guard<std::mutex> l(impl_->mu);
    if (trial < 0 || size_t(trial) >= impl_->trials.size()) return {};
    std::vector<Mark> out;
    for (const auto& e : impl_->trials[size_t(trial)].labels.events()) out.push_back(to_mark(e));
    return out;
}

std::vector<Mark> OfflineSession::prelabels(int64_t trial) const {
    std::lock_guard<std::mutex> l(impl_->mu);
    if (trial < 0 || size_t(trial) >= impl_->trials.size()) return {};
    std::vector<Mark> out;
    for (const auto& e : impl_->trials[size_t(trial)].labels.prelabels()) out.push_back(to_mark(e));
    return out;
}

bool OfflineSession::undo(int64_t trial) {
    std::lock_guard<std::mutex> l(impl_->mu);
    if (trial < 0 || size_t(trial) >= impl_->trials.size()) return false;
    return impl_->trials[size_t(trial)].labels.undo();
}

bool OfflineSession::redo(int64_t trial) {
    std::lock_guard<std::mutex> l(impl_->mu);
    if (trial < 0 || size_t(trial) >= impl_->trials.size()) return false;
    return impl_->trials[size_t(trial)].labels.redo();
}

double OfflineSession::reviewed_seconds() const {
    std::lock_guard<std::mutex> l(impl_->mu);
    double s = 0;
    for (const auto& t : impl_->trials) s += t.labels.reviewed_seconds(kTargetRate);
    return s;
}

bool OfflineSession::start_training(const TrainRequest& req, std::function<void(const TrainEvent&)> progress, std::string* error) {
    std::vector<eyelive::TrainTrace> traces;
    std::vector<int64_t> train_idx;
    {
        std::lock_guard<std::mutex> l(impl_->mu);
        if (impl_->trials.empty()) { if (error) *error = "no recording loaded"; return false; }
        if (impl_->trainer->running()) { if (error) *error = "training already running"; return false; }
        for (int64_t k = 0; k < int64_t(impl_->trials.size()); ++k)
            if (!std::binary_search(impl_->test_set.begin(), impl_->test_set.end(), k)) train_idx.push_back(k);
        traces = impl_->training_traces(train_idx, req.max_seconds);
        if (traces.empty()) { if (error) *error = "no reviewed stretch of at least 0.25 s: mark the reviewed ranges first"; return false; }
    }
    eyelive::TrainConfig cfg;
    cfg.n_classes = 3;
    cfg.max_seconds_of_data = req.max_seconds;
    cfg.rotation_aug = req.rotation_aug;
    cfg.max_epochs = req.max_epochs;
    cfg.patience = req.patience;
    cfg.n_threads = req.n_threads;
    cfg.deg_per_unit = impl_->deg_per_unit;
    if (!req.init_model.empty()) cfg.init_model = req.init_model;
    else if (req.use_last_model_as_init && !impl_->last_trained.empty()) cfg.init_model = impl_->last_trained;
    {
        std::lock_guard<std::mutex> l(impl_->progress_mu);
        impl_->progress_cb = std::move(progress);
    }
    impl_->trainer->start(std::move(traces), cfg, [this](const eyelive::TrainProgress& p) {
        TrainEvent e;
        e.epoch = p.epoch; e.train_loss = p.train_loss; e.val_loss = p.val_loss;
        e.val_event_f1 = p.val_event_f1; e.val_kappa = p.val_kappa; e.elapsed_s = p.elapsed_s;
        e.finished = p.finished; e.cancelled = p.cancelled; e.failed = p.failed; e.message = p.message;
        std::lock_guard<std::mutex> l(impl_->progress_mu);
        if (impl_->progress_cb) impl_->progress_cb(e);
    });
    return true;
}

void OfflineSession::cancel_training() { impl_->trainer->cancel(); }
bool OfflineSession::training_running() const { return impl_->trainer->running(); }

void OfflineSession::wait_training_done() const {
    while (impl_->trainer->running()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

bool OfflineSession::save_trained_model(const std::string& path, std::string* error) {
    if (!impl_->trainer->save_model(path, error)) return false;
    std::lock_guard<std::mutex> l(impl_->mu);
    impl_->last_trained = path;
    return true;
}

std::string OfflineSession::last_trained_model() const { std::lock_guard<std::mutex> l(impl_->mu); return impl_->last_trained; }

void OfflineSession::choose_test_set(int64_t every_nth, int64_t min_trials) {
    std::lock_guard<std::mutex> l(impl_->mu);
    impl_->test_set.clear();
    if (every_nth <= 0) return;
    for (int64_t k = 0; k < int64_t(impl_->trials.size()); k += every_nth) impl_->test_set.push_back(k);
    // guarantee a minimum size when there are enough trials
    if (min_trials > 0 && int64_t(impl_->test_set.size()) < min_trials) {
        for (int64_t k = 0; k < int64_t(impl_->trials.size()) && int64_t(impl_->test_set.size()) < min_trials; ++k)
            if (!std::binary_search(impl_->test_set.begin(), impl_->test_set.end(), k)) impl_->test_set.push_back(k);
        std::sort(impl_->test_set.begin(), impl_->test_set.end());
    }
}

std::vector<int64_t> OfflineSession::test_trials() const { std::lock_guard<std::mutex> l(impl_->mu); return impl_->test_set; }

TestScores OfflineSession::score_test_set(const std::string& model_bin, std::string* error) {
    TestScores out;
    std::lock_guard<std::mutex> l(impl_->mu);
    if (impl_->test_set.empty()) { out.error = "no test set chosen"; return out; }
    std::string err;
    auto det = eyelive::make_user_model_detector(model_bin, &err);
    if (!det) { out.error = err; if (error) *error = err; return out; }
    det->set_option("class_map", "identity");   // the app's classes are the model's classes
    const eyelive::SessionInfo si = impl_->session_info();
    for (const int64_t k : impl_->test_set) {
        const Trial& t = impl_->trials[size_t(k)];
        eyelive::VectorTrace tr = impl_->trace_view(t);
        det->reset(si);
        det->set_scale(impl_->deg_per_unit);
        std::vector<eyelive::Event> detected;
        det->detect_all(tr, detected);
        detected.erase(std::remove_if(detected.begin(), detected.end(), [](const eyelive::Event& e) { return e.cls == 2; }), detected.end());
        auto blinks = blink_events(t);
        detected.insert(detected.end(), blinks.begin(), blinks.end());
        const int64_t n = int64_t(t.x.size());
        std::vector<uint8_t> rev;
        t.labels.render(0, n - 1, true, rev);
        for (uint8_t cls = 1; cls < eyelive::kMaxClasses; ++cls) {
            std::vector<eyelive::Event> ex;
            for (const auto& e : t.labels.events())
                if (e.cls == cls && e.onset < n && rev[size_t(e.onset)] != eyelive::kIgnoreLabel) { ex.push_back(e); ++out.n_events[cls]; }
            if (ex.empty()) continue;   // agreement() is undefined without expert events of the class
            auto a = eyelive::agreement(detected, ex, n, cls);
            // accumulate as sums of per-trial stats weighted by samples; the caller pools
            out.event_f1[cls] += a.event_f1 * double(n);
            out.kappa[cls] += a.kappa * double(n);
        }
        out.n_samples += n;
    }
    if (out.n_samples > 0)
        for (uint8_t cls = 1; cls < eyelive::kMaxClasses; ++cls) { out.event_f1[cls] /= double(out.n_samples); out.kappa[cls] /= double(out.n_samples); }
    out.ok = true;
    return out;
}

bool OfflineSession::prelabel_trial(int64_t trial, std::string* error) {
    std::lock_guard<std::mutex> l(impl_->mu);
    if (trial < 0 || size_t(trial) >= impl_->trials.size()) { if (error) *error = "trial index out of range"; return false; }
    return impl_->prelabel(impl_->trials[size_t(trial)], error);
}

bool OfflineSession::re_prelabel_all(std::string* error) {
    std::lock_guard<std::mutex> l(impl_->mu);
    if (impl_->trials.empty()) { if (error) *error = "no recording loaded"; return false; }
    for (auto& t : impl_->trials) {
        std::string err;
        if (!impl_->re_prelabel(t, &err)) { if (error) *error = err; return false; }
    }
    return true;
}

void OfflineSession::set_prelabeler(Prelabeler p, const std::string& model_bin) {
    std::lock_guard<std::mutex> l(impl_->mu);
    impl_->prelabeler = p;
    if (!model_bin.empty()) impl_->model_bin = model_bin;
}

Prelabeler OfflineSession::prelabeler() const { std::lock_guard<std::mutex> l(impl_->mu); return impl_->prelabeler; }

}  // namespace offline

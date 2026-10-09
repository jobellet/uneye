// fault_injection: Step 5 of docs/ROADMAP.md. Perturbs real recordings (set B = test data, human labels) in controlled ways and compares
//   nn     the causal network alone (GazeEngine in NetworkOnly mode, fast TCN only): what a bare model would output
//   engine the full engine (fast TCN + 10 ms TCN + U'n'Eye window network, physics layer and network watchdog), WITHOUT the safety guard
//   guard  the same engine followed by the deterministic SafetyGuard (only the events the guard emits)
//
// Metrics per (perturbation, level, system):
//   false_per_min   emitted events that overlap no human-labeled saccade (+-10 samples), per minute of data
//   recall          share of human-labeled saccades overlapped by an emitted event
//   dangerous       emitted events that are physiologically impossible when measured on the CLEAN (unperturbed) signal: the eye did not
//                   move (peak speed < 10 deg/s), or amplitude > 40 deg, or duration > 150 ms. Measuring on the clean signal is what makes
//                   this independent of the guard's own checks (which only see the perturbed signal).
//   degraded, safe  share of samples the guard spent in DEGRADED / SAFE (guard only)
//
// Deterministic: fixed seeds, no clock (the guard gets "compute time unknown", except in the deadline scenario where late samples are injected).
// Usage (from cpp/): build/fault_injection [--trials 120] [--out ../docs/slides/data/fault_injection.csv] [--trace ../docs/slides/data/trace.csv]
//                    [--trace-kind frozen] [--only <perturbation>]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "uneye/causal_engine.hpp"
#include "uneye/engine.hpp"
#include "uneye/gaze_engine.hpp"
#include "uneye/safety_guard.hpp"

using namespace uneye;
using gaze::Sample;
using safety::GuardOutput;
using safety::GuardState;
using safety::SafetyGuard;

namespace {

struct Rng { uint64_t s; explicit Rng(uint64_t seed) : s(seed * 2862933555777941757ULL + 3037000493ULL) {}
    double u() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<double>(s >> 11) / 9007199254740992.0; }
    double n() { double a = u(), b = u(); return std::sqrt(-2 * std::log(a + 1e-12)) * std::cos(6.283185307179586 * b); } };

std::vector<std::vector<double>> read_csv(const std::string& path, int max_rows) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    std::vector<std::vector<double>> rows; std::string line;
    while (static_cast<int>(rows.size()) < max_rows && std::getline(f, line)) {
        std::vector<double> r; std::stringstream ss(line); std::string tok;
        while (std::getline(ss, tok, ',')) r.push_back(tok.empty() || tok == "nan" ? NAN : std::stod(tok));
        if (!r.empty()) rows.push_back(std::move(r));
    }
    return rows;
}

// one continuous stream: the trials one after the other, separated by 100 samples of NaN (looks like a blink, has no label)
struct Stream { std::vector<double> x, y; std::vector<int> lab; };   // lab: 1 saccade, 0 fixation, -1 no label (gap)

Stream load_stream(const std::string& root, const std::string& d, const std::string& f, int trials, const std::string& set) {
    const std::string p = root + "/data/" + d + "/" + f;
    auto X = read_csv(p + "_X_set" + set + ".csv", trials), Y = read_csv(p + "_Y_set" + set + ".csv", trials), L = read_csv(p + "_Labels_set" + set + ".csv", trials);
    Stream s;
    for (size_t k = 0; k < X.size(); ++k) {
        for (size_t t = 0; t < X[k].size(); ++t) { s.x.push_back(X[k][t]); s.y.push_back(Y[k][t]); s.lab.push_back(L[k][t] > 0 ? 1 : 0); }
        for (int g = 0; g < 100; ++g) { s.x.push_back(NAN); s.y.push_back(NAN); s.lab.push_back(-1); }
    }
    return s;
}

// ---------------------------------------------------------------- perturbations (applied to a copy of the stream)
struct Perturbed { std::vector<Sample> samples; std::vector<long> src; std::vector<double> late_us; };   // src: index in the clean stream

Perturbed perturb(const Stream& s, const std::string& kind, double level, uint64_t seed) {
    Rng r(seed);
    Perturbed p;
    const size_t n = s.x.size();
    double drift = 0;
    long burst_left = 0;
    for (size_t i = 0; i < n; ++i) {
        double x = s.x[i], y = s.y[i];
        if (kind == "rate_wrong" || kind == "rate_correct") {          // 500 Hz data: every 2nd sample. Fed with 1 ms (wrong) or 2 ms (correct) steps
            if (i % 2) continue;
        }
        if (std::isfinite(x)) {
            if (kind == "noise") { x += level * r.n(); y += level * r.n(); }
            else if (kind == "burst") {                                    // 50 ms of 200 Hz interference, starting at random about once a second
                if (burst_left == 0 && r.u() < 0.001) burst_left = 50;
                if (burst_left > 0) { const double ph = 6.283185307179586 * 200.0 * static_cast<double>(i) / 1000.0; x += level * std::sin(ph); y += level * std::cos(ph); --burst_left; }
            }
            else if (kind == "dropout") { if (burst_left == 0 && r.u() < level) burst_left = 150; if (burst_left > 0) { x = y = NAN; --burst_left; } }
            else if (kind == "saturation") { x = std::min(std::max(x, -level), level); y = std::min(std::max(y, -level), level); }
            else if (kind == "drift") { drift += level / 1000.0; x += drift; }
            else if (kind == "scale") { x *= level; y *= level; }
            else if (kind == "spikes") { if (r.u() < level) { x += (r.u() < 0.5 ? -1 : 1) * 20.0; } }
            else if (kind == "frozen") { if (burst_left == 0 && r.u() < level) burst_left = 200; if (burst_left > 0) { x = p.samples.empty() ? x : p.samples.back().x_deg; y = p.samples.empty() ? y : p.samples.back().y_deg; --burst_left; } }
        }
        Sample sm;
        const double step_us = kind == "rate_correct" ? 2000.0 : 1000.0;
        sm.t_us = static_cast<double>(p.samples.size()) * step_us;
        sm.x_deg = x; sm.y_deg = y;
        p.samples.push_back(sm); p.src.push_back(static_cast<long>(i));
        p.late_us.push_back(kind == "deadline" && r.u() < level ? 2000.0 : -1.0);
    }
    return p;
}

// ---------------------------------------------------------------- events and metrics
struct Ev { long on, off; };   // indices in the CLEAN stream

std::vector<Ev> runs_of(const std::vector<int>& lab) {
    std::vector<Ev> v; long on = -1;
    for (size_t i = 0; i <= lab.size(); ++i) {
        const bool s = i < lab.size() && lab[i] == 1;
        if (s && on < 0) on = static_cast<long>(i);
        if (!s && on >= 0) { v.push_back({on, static_cast<long>(i) - 1}); on = -1; }
    }
    return v;
}

// events from a per-sample saccade label (systems without the guard): runs merged over gaps of <= 2 samples, at least 3 samples
std::vector<Ev> events_from_labels(const std::vector<int>& sac, const std::vector<long>& src) {
    std::vector<Ev> v; long on = -1, last = -1; int gap = 0;
    for (size_t i = 0; i <= sac.size(); ++i) {
        const bool s = i < sac.size() && sac[i];
        if (s) { if (on < 0) on = static_cast<long>(i); last = static_cast<long>(i); gap = 0; }
        else if (on >= 0 && i < sac.size() && ++gap <= 2) continue;
        else if (on >= 0) { if (last - on + 1 >= 3) v.push_back({src[static_cast<size_t>(on)], src[static_cast<size_t>(last)]}); on = -1; gap = 0; }
    }
    return v;
}

struct Metrics { double false_per_min = 0, recall = 0; long dangerous = 0, emitted = 0; double degraded = 0, safe = 0; };

Metrics score(const Stream& s, const std::vector<Ev>& det, double minutes) {
    Metrics m; m.emitted = static_cast<long>(det.size());
    const auto truth = runs_of(s.lab);
    std::vector<char> hit(truth.size(), 0);
    long fp = 0;
    for (const auto& e : det) {
        bool any = false;
        for (size_t k = 0; k < truth.size(); ++k)
            if (e.on <= truth[k].off + 10 && e.off >= truth[k].on - 10) { any = true; hit[k] = 1; }
        if (!any) ++fp;
        // physiology on the CLEAN signal (3-sample mean velocity, as in the engine)
        double peak = 0;
        for (long i = std::max(e.on, 3L); i <= e.off && i < static_cast<long>(s.x.size()); ++i) {
            const double dx = (s.x[static_cast<size_t>(i)] - s.x[static_cast<size_t>(i - 3)]) / 3.0 * 1000.0, dy = (s.y[static_cast<size_t>(i)] - s.y[static_cast<size_t>(i - 3)]) / 3.0 * 1000.0;
            if (std::isfinite(dx) && std::isfinite(dy)) peak = std::max(peak, std::hypot(dx, dy));
        }
        const long a = std::max(e.on - 1, 0L), b = std::min(e.off + 1, static_cast<long>(s.x.size()) - 1);
        const double amp = std::hypot(s.x[static_cast<size_t>(b)] - s.x[static_cast<size_t>(a)], s.y[static_cast<size_t>(b)] - s.y[static_cast<size_t>(a)]);
        const double dur = static_cast<double>(e.off - e.on + 1);
        if (peak < 10.0 || (std::isfinite(amp) && amp > 40.0) || dur > 150.0) ++m.dangerous;
    }
    long nh = 0; for (char h : hit) nh += h;
    m.recall = truth.empty() ? 0 : static_cast<double>(nh) / static_cast<double>(truth.size());
    m.false_per_min = static_cast<double>(fp) / minutes;
    return m;
}

gaze::Models make_models(const std::string& root, bool full) {
    gaze::Models m;
    m.fast = make_causal_engine(root + "/models/causal.bin");
    if (full) { m.refine = make_causal_engine(root + "/models/tcn_l10.bin"); m.window = make_native_engine(root + "/models/combined.bin", 200); }
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    std::string root = ".", repo = "..", outp = "fault_injection.csv", tracep = "";
    int trials = 120;
    std::string only = "", trace_kind = "frozen";        // --only <kind>: run one perturbation; --trace-kind: which one is traced (dataset 1, first level)
    std::string set = "B";                               // --set A: training set, used ONLY to calibrate guard thresholds (cpp/scripts/calibrate_ood.py)
    double ood = -1;                                     // --ood <deg/s>: guard out-of-distribution noise limit (default: GuardConfig)
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--trials") trials = std::atoi(v.c_str()); else if (a == "--out") outp = v; else if (a == "--trace") tracep = v;
        else if (a == "--root") root = v; else if (a == "--repo") repo = v;
        else if (a == "--only") only = v; else if (a == "--trace-kind") trace_kind = v;
        else if (a == "--set") set = v; else if (a == "--ood") ood = std::atof(v.c_str());
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    struct Scn { std::string kind; std::vector<double> levels; };
    const std::vector<Scn> scenarios = {
        {"none", {0}},
        {"noise", {0.02, 0.05, 0.1, 0.2}},                  // deg, gaussian per sample
        {"burst", {0.1, 0.5, 2.0}},                         // deg, 200 Hz interference, 50 ms bursts
        {"dropout", {0.002, 0.01}},                         // probability per sample of a 150 ms dropout
        {"spikes", {0.001, 0.01}},                          // probability per sample of a 20 deg spike
        {"saturation", {5.0, 2.0}},                         // clipping at +-level deg (below the guard's 30 deg rail: NOT expected to be caught)
        {"frozen", {0.002}},                                // probability per sample of a 200 ms frozen signal
        {"drift", {5.0, 50.0}},                             // deg/s
        {"scale", {0.1, 10.0}},                             // amplitude x level
        {"rate_wrong", {1}}, {"rate_correct", {1}},         // 500 Hz data fed with 1 ms / 2 ms timestamps
        {"deadline", {0.0005, 0.01}},                       // probability per sample that the computation was late (guard input only)
        {"net_stuck", {1}},                                 // network outputs p(saccade) = 1 forever (fault inside the model)
    };
    const std::vector<std::pair<std::string, std::string>> sets = {{"dataset1", "dataset1_1000hz"}, {"dataset2", "dataset2_1000hz"}};
    std::vector<Stream> streams;
    for (const auto& s : sets) streams.push_back(load_stream(repo, s.first, s.second, trials, set));
    safety::GuardConfig gcfg; if (ood > 0) gcfg.ood_sigma_hi_deg_s = ood;

    std::FILE* out = std::fopen(outp.c_str(), "w");
    if (!out) { std::fprintf(stderr, "cannot write %s\n", outp.c_str()); return 1; }
    std::fprintf(out, "perturbation,level,system,dataset,minutes,emitted,false_per_min,recall,dangerous,degraded,safe\n");
    std::FILE* tr = tracep.empty() ? nullptr : std::fopen(tracep.c_str(), "w");
    if (tr) std::fprintf(tr, "i,x_clean,x_in,human,p_nn,label_nn,state_guard,label_guard,reason\n");

    struct StuckNet : StepEngine { int classes() const override { return 2; } int receptive_field() const override { return 1; } void reset() override {}
                                   void step(float, float, float* p) override { p[0] = 0.0f; p[1] = 1.0f; } };

    for (const auto& sc : scenarios) {
        if (!only.empty() && sc.kind != only) continue;
        for (double level : sc.levels)
            for (size_t d = 0; d < streams.size(); ++d) {
                const Stream& s = streams[d];
                const Perturbed p = perturb(s, sc.kind, level, 1000 + d);
                const double minutes = static_cast<double>(p.samples.size()) / 1000.0 / 60.0;   // as seen by the engine (1 ms per sample)
                const bool stuck = sc.kind == "net_stuck";
                // ---- nn: network alone
                gaze::Config cn; cn.mode = gaze::Mode::NetworkOnly;
                gaze::Models mn = make_models(root, false); if (stuck) mn.fast = std::make_unique<StuckNet>();
                gaze::GazeEngine en(cn, std::move(mn));
                // ---- engine + guard
                gaze::Models mf = make_models(root, true); if (stuck) mf.fast = std::make_unique<StuckNet>();
                gaze::GazeEngine ef(gaze::Config{}, std::move(mf));
                SafetyGuard guard(gcfg);
                std::vector<int> sac_nn(p.samples.size()), sac_eng(p.samples.size());
                std::vector<Ev> ev_guard;
                long n_deg = 0, n_safe = 0;
                gaze::Output on, of; GuardOutput go;
                const bool trace_this = tr && sc.kind == trace_kind && level == sc.levels.front() && d == 0;
                for (size_t i = 0; i < p.samples.size(); ++i) {
                    en.push(p.samples[i], on); ef.push(p.samples[i], of);
                    guard.step(p.samples[i], of, p.late_us[i], go);
                    sac_nn[i] = on.state == gaze::State::Saccade; sac_eng[i] = of.state == gaze::State::Saccade;
                    n_deg += go.state == GuardState::Degraded; n_safe += go.state == GuardState::Safe;
                    if (go.event_ended && go.event.accepted)
                        ev_guard.push_back({p.src[static_cast<size_t>(go.event.onset)], p.src[static_cast<size_t>(go.event.offset)]});
                    if (trace_this && i < 60000)
                        std::fprintf(tr, "%zu,%.5f,%.5f,%d,%.4f,%d,%d,%d,%s\n", i, s.x[static_cast<size_t>(p.src[i])], p.samples[i].x_deg, s.lab[static_cast<size_t>(p.src[i])],
                                     on.p_saccade, sac_nn[i], static_cast<int>(go.state), static_cast<int>(go.label), safety::reason_name(go.reason));
                }
                const Metrics a = score(s, events_from_labels(sac_nn, p.src), minutes);
                const Metrics b = score(s, events_from_labels(sac_eng, p.src), minutes);
                Metrics c = score(s, ev_guard, minutes);
                c.degraded = static_cast<double>(n_deg) / static_cast<double>(p.samples.size()); c.safe = static_cast<double>(n_safe) / static_cast<double>(p.samples.size());
                const Metrics* ms[3] = {&a, &b, &c}; const char* names[3] = {"nn", "engine", "guard"};
                for (int k = 0; k < 3; ++k)
                    std::fprintf(out, "%s,%g,%s,%s,%.3f,%ld,%.3f,%.4f,%ld,%.4f,%.4f\n", sc.kind.c_str(), level, names[k], sets[d].first.c_str(), minutes,
                                 ms[k]->emitted, ms[k]->false_per_min, ms[k]->recall, ms[k]->dangerous, ms[k]->degraded, ms[k]->safe);
                std::fflush(out);
                std::printf("%-12s %-7g %s | false/min nn %6.1f engine %6.1f guard %6.1f | dangerous nn %4ld engine %4ld guard %4ld | recall nn %.2f engine %.2f guard %.2f | guard DEG %.2f SAFE %.2f\n",
                            sc.kind.c_str(), level, sets[d].first.c_str(), a.false_per_min, b.false_per_min, c.false_per_min, a.dangerous, b.dangerous, c.dangerous,
                            a.recall, b.recall, c.recall, c.degraded, c.safe);
                // every reason the guard gave in this run, with its count (what drove DEGRADED / SAFE)
                const auto& br = guard.stats().by_reason; std::printf("             guard reasons:");
                for (int k = 1; k < static_cast<int>(safety::Reason::Count); ++k)
                    if (br[static_cast<size_t>(k)]) std::printf(" %s %ld", safety::reason_name(static_cast<safety::Reason>(k)), static_cast<long>(br[static_cast<size_t>(k)]));
                std::printf("\n");
                std::fflush(stdout);
            }
    }
    std::fclose(out);
    if (tr) std::fclose(tr);
    return 0;
}

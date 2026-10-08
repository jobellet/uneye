// gaze_replay: stream recorded trials through the GazeEngine like a live tracker, with optional fault injection, and write every output.
//   gaze_replay --x X.csv --y Y.csv --fast models/causal.bin --refine models/tcn_l10.bin --out out.bin [--mode fused|nn|heur]
//               [--fault none|nan|const1|stuck|random|badsum] [--fault-start 3000] [--loss 0.01] [--spike 0.001] [--trials 100]
// Output: float32 matrix, one row per processed sample, kCols columns (see the Python scorer online/gaze_score.py).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "uneye/gaze_engine.hpp"

using namespace uneye;
using namespace uneye::gaze;

namespace {

constexpr int kCols = 24;

// ---- deterministic pseudo random numbers for fault injection (no <random>, results identical everywhere)
struct Rng { uint64_t s; explicit Rng(uint64_t seed) : s(seed * 2862933555777941757ULL + 3037000493ULL) {}
    double u() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<double>(s >> 11) / 9007199254740992.0; } };

enum class Fault { None, Nan, Const1, Const0, Stuck, Random, BadSum };

class FaultyEngine : public StepEngine {                   // wraps a real network and breaks it after `start` steps
public:
    FaultyEngine(std::unique_ptr<StepEngine> in, Fault f, long start) : in_(std::move(in)), f_(f), start_(start), rng_(7) {}
    int classes() const override { return 2; }
    int receptive_field() const override { return in_->receptive_field(); }
    void reset() override { in_->reset(); steps_ = 0; }
    void step(float dx, float dy, float* p) override {
        in_->step(dx, dy, p);
        if (f_ == Fault::None || steps_++ < start_) return;
        switch (f_) {
            case Fault::Nan: p[0] = p[1] = NAN; break;
            case Fault::Const1: p[0] = 0.0f; p[1] = 1.0f; break;
            case Fault::Const0: p[0] = 1.0f; p[1] = 0.0f; break;
            case Fault::Stuck: p[0] = 0.3f; p[1] = 0.7f; break;
            case Fault::Random: { const float u = static_cast<float>(rng_.u()); p[1] = u; p[0] = 1.0f - u; break; }
            case Fault::BadSum: p[0] = 0.9f; p[1] = 0.9f; break;
            default: break;
        }
    }
private:
    std::unique_ptr<StepEngine> in_; Fault f_; long start_; long steps_ = 0; Rng rng_;
};

std::vector<std::vector<double>> read_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    std::vector<std::vector<double>> rows; std::string line;
    while (std::getline(f, line)) {
        std::vector<double> r; std::stringstream ss(line); std::string tok;
        while (std::getline(ss, tok, ',')) r.push_back(tok.empty() || tok == "nan" ? NAN : std::stod(tok));
        if (!r.empty()) rows.push_back(std::move(r));
    }
    return rows;
}

double pct(std::vector<double>& v, double q) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[static_cast<size_t>(q * (v.size() - 1))]; }

}  // namespace

int main(int argc, char** argv) {
    std::string xf, yf, fastp, refp, outp = "gaze_out.bin", mode = "fused", fault = "none";
    long fault_start = 3000; int trials = 1 << 30; double loss = 0, spike = 0; bool continuous = false;
    Config cfg;
    auto need = [&](int& i) -> const char* { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); std::exit(2); } return argv[++i]; };
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--x") xf = need(i); else if (s == "--y") yf = need(i);
        else if (s == "--fast") fastp = need(i); else if (s == "--refine") refp = need(i);
        else if (s == "--out") outp = need(i); else if (s == "--mode") mode = need(i);
        else if (s == "--fault") fault = need(i); else if (s == "--fault-start") fault_start = std::atol(need(i));
        else if (s == "--loss") loss = std::atof(need(i)); else if (s == "--spike") spike = std::atof(need(i));
        else if (s == "--trials") trials = std::atoi(need(i));
        else if (s == "--continuous") continuous = true;
        else if (s == "--lambda") cfg.lambda = std::atof(need(i)); else if (s == "--lambda-weak") cfg.lambda_weak = std::atof(need(i));
        else if (s == "--lambda-strong") cfg.lambda_strong = std::atof(need(i)); else if (s == "--floor") cfg.floor_deg_s = std::atof(need(i));
        else if (s == "--floor-weak") cfg.floor_weak_deg_s = std::atof(need(i)); else if (s == "--floor-strong") cfg.floor_strong_deg_s = std::atof(need(i));
        else if (s == "--min-run") cfg.min_run = std::atoi(need(i)); else if (s == "--gate-window") cfg.gate_half_window = std::atoi(need(i));
        else { std::fprintf(stderr, "unknown option %s\n", s.c_str()); return 2; }
    }
    if (xf.empty() || yf.empty()) { std::fprintf(stderr, "usage: gaze_replay --x X.csv --y Y.csv [--fast f.bin --refine r.bin] --out out.bin\n"); return 2; }
    Fault fk = fault == "nan" ? Fault::Nan : fault == "const1" ? Fault::Const1 : fault == "const0" ? Fault::Const0 : fault == "stuck" ? Fault::Stuck : fault == "random" ? Fault::Random : fault == "badsum" ? Fault::BadSum : Fault::None;

    cfg.mode = mode == "nn" ? Mode::NetworkOnly : mode == "heur" ? Mode::HeuristicOnly : Mode::Fused;
    Models m;
    if (!fastp.empty()) m.fast = std::make_unique<FaultyEngine>(make_causal_engine(fastp), fk, fault_start);
    if (!refp.empty()) { m.refine = std::make_unique<FaultyEngine>(make_causal_engine(refp), fk, fault_start); m.refine_delay = 10; }
    GazeEngine eng(cfg, std::move(m));

    auto X = read_csv(xf), Y = read_csv(yf);
    const int n_trials = std::min<int>(trials, static_cast<int>(X.size()));
    std::vector<float> out; out.reserve(static_cast<size_t>(n_trials) * 1000 * kCols);
    std::vector<double> ns; ns.reserve(static_cast<size_t>(n_trials) * 1000);
    Rng rng(42);
    Output o; Snapshot snap;
    long unhealthy = 0, total = 0; long g_clock = 0;
    for (int k = 0; k < n_trials; ++k) {
        if (!continuous || k == 0) eng.reset();                 // --continuous: one uninterrupted stream (trial boundaries look like blinks)
        const size_t T = X[static_cast<size_t>(k)].size();
        std::vector<std::vector<float>> rows;                   // one row per processed sample of this trial (index = sample number)
        long n_in = 0; if (!continuous) g_clock = 0;
        for (size_t t = 0; t < T; ++t) {
            if (loss > 0 && rng.u() < loss) { ++n_in; ++g_clock; continue; }          // dropped packet: the timestamp jumps
            Sample s; s.t_us = static_cast<double>(g_clock++) * 1000.0; ++n_in; s.x_deg = X[static_cast<size_t>(k)][t]; s.y_deg = Y[static_cast<size_t>(k)][t];
            if (spike > 0 && rng.u() < spike) { const double r = rng.u(); s.x_deg = r < 0.33 ? NAN : r < 0.66 ? 1e9 : -s.x_deg * 1000.0; }
            const auto t0 = std::chrono::steady_clock::now();
            eng.push(s, o);
            ns.push_back(static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()));
            ++total; if (o.health != Health::Ok) ++unhealthy;
            const float amp = 0;
            eng.snapshot(snap);
            // age-30 bin = the label after the delayed network and the blink margins have had their say
            float st30 = -1;
            if (snap.n_bins > 30) st30 = static_cast<float>(snap.bins[static_cast<size_t>(snap.n_bins - 1 - 30)].state);
            float ev_cls = -1, ev_pred = 0;
            for (int e = 0; e < snap.n_events; ++e) if (snap.events[static_cast<size_t>(e)].ongoing) { ev_cls = static_cast<float>(snap.events[static_cast<size_t>(e)].cls); ev_pred = snap.events[static_cast<size_t>(e)].predicted_amplitude_deg; }
            const float row[kCols] = {static_cast<float>(k), static_cast<float>(o.index), static_cast<float>(t), static_cast<float>(o.state), o.p_saccade, static_cast<float>(o.source),
                static_cast<float>(o.flags), static_cast<float>(o.health), st30,
                static_cast<float>(o.forecast.at10.x), static_cast<float>(o.forecast.at10.y), static_cast<float>(o.forecast.at10.scale_x), static_cast<float>(o.forecast.at10.scale_y),
                static_cast<float>(o.forecast.at20.x), static_cast<float>(o.forecast.at20.y), static_cast<float>(o.forecast.at20.scale_x), static_cast<float>(o.forecast.at20.scale_y),
                static_cast<float>(o.event_onset), o.forecast.ballistic ? 1.0f : 0.0f, ev_pred, ev_cls, static_cast<float>(o.revised_first), static_cast<float>(o.revised_last), amp};
            out.insert(out.end(), row, row + kCols);
        }
    }
    std::ofstream f(outp, std::ios::binary);
    const int32_t hdr[2] = {static_cast<int32_t>(out.size() / kCols), kCols};
    f.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
    f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
    const auto st = eng.stats();
    std::printf("%ld samples, %.3f%% not Ok | push() time: median %.0f ns, p99 %.0f ns, p99.99 %.0f ns, max %.0f ns  (snapshot per sample excluded)\n",
                total, 100.0 * unhealthy / std::max(total, 1L), pct(ns, 0.5), pct(ns, 0.99), pct(ns, 0.9999), pct(ns, 1.0));
    std::printf("network watchdog: trust dropped %lld x (invalid output %lld, stuck %lld, veto-rate %lld, missed clear saccades %lld, flicker %lld)\n", (long long)(st.trust_drops_invalid + st.trust_drops_stuck + st.trust_drops_veto + st.trust_drops_miss + st.trust_drops_flicker), (long long)st.trust_drops_invalid, (long long)st.trust_drops_stuck, (long long)st.trust_drops_veto, (long long)st.trust_drops_miss, (long long)st.trust_drops_flicker);
    std::printf("clear fast movements (>100 deg/s): %lld, not seen by the network: %lld\n", (long long)st.clear_events, (long long)st.clear_missed);
    std::printf("stats of the last trial: samples %lld invalid %lld vetoed %lld overridden %lld network-invalid %lld resets %lld\n", (long long)st.samples, (long long)st.invalid, (long long)st.vetoed, (long long)st.overridden, (long long)st.nn_invalid, (long long)st.resets);
    return 0;
}

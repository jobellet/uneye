// uneye_rt: real-time saccade/microsaccade labeling with U'n'Eye.
//
//   uneye_rt --model models/combined.bin --sim                      simulated eye tracker, scored vs ground truth
//   uneye_rt --model models/dataset1.onnx --replay X.csv Y.csv L.csv  replay recorded trials as a live stream, scored
//   some_gaze_source | uneye_rt --model models/combined.bin --stdin   live: "x y" per line (deg), events on stdout
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "uneye/detector.hpp"
#include "uneye/simulator.hpp"

using namespace uneye;
using Clock = std::chrono::steady_clock;

namespace {

struct Args {
    std::string model = "models/combined.bin", mode, xf, yf, lf, out_labels;
    double window_ms = 200;  // time bin seen by the network
    int window = 200;
    Config cfg;
    SimConfig sim;
    bool realtime = false, sweep = false, per_sample = false, fast = false;
};

bool ends_with(const std::string& s, const std::string& e) { return s.size() >= e.size() && s.compare(s.size() - e.size(), e.size(), e) == 0; }

std::unique_ptr<Engine> make_engine(const Args& a);

std::unique_ptr<StreamingDetector> make_detector(const Args& a) {
    if (is_causal_file(a.model)) return std::make_unique<StreamingDetector>(make_causal_engine(a.model), a.cfg);
    return std::make_unique<StreamingDetector>(make_engine(a), a.cfg);
}

std::unique_ptr<Engine> make_engine(const Args& a) {
    if (ends_with(a.model, ".onnx")) {
        // classes from sidecar json is overkill: probe via file name convention -> use 2 unless "andersson"
        const int classes = a.model.find("andersson") != std::string::npos ? 5 : 2;
        return make_onnx_engine(a.model, a.window, classes);
    }
    return make_native_engine(a.model, a.window);
}

std::vector<std::vector<double>> read_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        std::vector<double> r;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) r.push_back(tok.empty() || tok == "nan" ? NAN : std::stod(tok));
        if (!r.empty()) rows.push_back(std::move(r));
    }
    return rows;
}

// ---------------- scoring ----------------
struct Score {
    int64_t tp = 0, fp = 0, fn = 0, tn = 0;
    int64_t ev_true[2] = {0, 0}, ev_hit[2] = {0, 0}, ev_det = 0, ev_det_ok = 0;  // [0]=micro(<1deg) [1]=saccade
    std::vector<double> lat_on, lat_on_micro;  // onset-known latency (ms)
    std::vector<double> lat;                    // detection latency (ms) of true events: confirmed - onset
    std::vector<double> lat_micro, onset_err;   // ms
    std::vector<double> infer_us;
};

struct Run { std::vector<int> fpred; std::vector<Event> fevents; std::vector<std::pair<int64_t,int64_t>> fonsets; std::vector<std::pair<int64_t,int64_t>> onsets; std::vector<int> pred; std::vector<Event> events; std::vector<int64_t> infer_ns; };

void score_trace(Score& S, const std::vector<int>& truth, const std::vector<int>& pred,
                 const std::vector<Event>& dets, const std::vector<std::pair<int64_t,int64_t>>& onsets, const std::vector<double>& x, const std::vector<double>& y,
                 double fs) {
    const size_t n = truth.size();
    for (size_t i = 0; i < n; ++i) {
        const bool t = truth[i] != 0, p = pred[i] != 0;
        (t ? (p ? S.tp : S.fn) : (p ? S.fp : S.tn))++;
    }
    // true events = runs of truth != 0
    std::vector<std::pair<int64_t, int64_t>> te;
    for (size_t i = 0; i < n;) {
        if (!truth[i]) { ++i; continue; }
        size_t j = i; while (j + 1 < n && truth[j + 1]) ++j;
        te.push_back({int64_t(i), int64_t(j)}); i = j + 1;
    }
    const double ms = 1000.0 / fs;
    std::vector<char> used(dets.size(), 0);
    for (auto& e : te) {
        const int64_t a = std::max<int64_t>(e.first - 1, 0);
        const double amp = std::hypot(x[e.second] - x[a], y[e.second] - y[a]);
        const int k = amp >= 1.0 ? 1 : 0;
        S.ev_true[k]++;
        for (size_t d = 0; d < dets.size(); ++d)
            if (!used[d] && dets[d].onset <= e.second && dets[d].offset >= e.first) {
                used[d] = 1; S.ev_hit[k]++;
                const double l = (dets[d].confirmed - 1 - e.first) * ms;
                S.lat.push_back(l); if (k == 0) S.lat_micro.push_back(l);
                for (auto& o : onsets) if (o.first == dets[d].onset) {
                    const double lo = (o.second - e.first) * ms;
                    S.lat_on.push_back(lo); if (k == 0) S.lat_on_micro.push_back(lo);
                    break;
                }
                S.onset_err.push_back((dets[d].onset - e.first) * ms);
                break;
            }
    }
    S.ev_det += int64_t(dets.size());
    for (char u : used) S.ev_det_ok += u;
}

double pct(std::vector<double> v, double q) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    return v[size_t(q * (v.size() - 1))];
}

void report(const Args& a, const Score& S, int lookahead, const char* tag = "committed") {
    const double P = double(S.tp) / std::max<int64_t>(S.tp + S.fp, 1), R = double(S.tp) / std::max<int64_t>(S.tp + S.fn, 1);
    const double F1 = 2 * P * R / std::max(P + R, 1e-12);
    const double N = double(S.tp + S.fp + S.fn + S.tn);
    const double po = (S.tp + S.tn) / N;
    const double pe = ((S.tp + S.fp) * double(S.tp + S.fn) + (S.fn + S.tn) * double(S.fp + S.tn)) / (N * N);
    const double kappa = (po - pe) / std::max(1 - pe, 1e-12);
    const int64_t nt = S.ev_true[0] + S.ev_true[1], nh = S.ev_hit[0] + S.ev_hit[1];
    std::printf("%-9s lookahead=%3d  F1=%.3f kappa=%.3f | events: recall=%.3f (micro %.3f, sacc %.3f) precision=%.3f | onset-known latency ms: median=%.0f p95=%.0f (micro %.0f) | event-end latency median=%.0f | onset pos err median=%.0f ms",
                tag, lookahead, F1, kappa, double(nh) / std::max<int64_t>(nt, 1),
                double(S.ev_hit[0]) / std::max<int64_t>(S.ev_true[0], 1), double(S.ev_hit[1]) / std::max<int64_t>(S.ev_true[1], 1),
                double(S.ev_det_ok) / std::max<int64_t>(S.ev_det, 1), pct(S.lat_on, 0.5), pct(S.lat_on, 0.95), pct(S.lat_on_micro, 0.5), pct(S.lat, 0.5), pct(S.onset_err, 0.5));
    if (!a.sweep) {
        std::printf("\n  true events: %lld (micro %lld)  detections: %lld\n", (long long)nt, (long long)S.ev_true[0], (long long)S.ev_det);
        std::printf("  network call: median=%.0f us  p99=%.0f us  max=%.0f us  (budget per sample @%.0f Hz = %.0f us, hop=%d -> %.0f us per call)\n",
                    pct(S.infer_us, 0.5), pct(S.infer_us, 0.99), pct(S.infer_us, 1.0), a.cfg.fs, 1e6 / a.cfg.fs, a.cfg.hop, 1e6 * a.cfg.hop / a.cfg.fs);
    }
    std::printf("\n");
}

// Stream one trace through a fresh detector exactly as a live tracker would.
Run stream_trace(StreamingDetector& det, const Args& a, const std::vector<double>& x, const std::vector<double>& y, bool pace) {
    det.reset();
    Run r; r.pred.assign(x.size(), 0);
    det.on_label = [&](const Label& l) { if (l.index >= 0 && size_t(l.index) < r.pred.size()) r.pred[l.index] = l.cls; };
    det.on_event = [&](const Event& e) { r.events.push_back(e); };
    if (a.fast) {
        r.fpred.assign(x.size(), 0);
        det.on_fast = [&](const Fast& f) { if (f.index >= 0 && size_t(f.index) < r.fpred.size()) r.fpred[f.index] = f.cls; };
    }
    det.on_onset = [&](int64_t st) { r.onsets.push_back({st, det.samples() - 1}); };
    const auto t0 = Clock::now();
    for (size_t i = 0; i < x.size(); ++i) {
        if (pace) std::this_thread::sleep_until(t0 + std::chrono::nanoseconds(int64_t(1e9 * i / a.cfg.fs)));
        det.push(x[i], y[i]);
        if (det.last_infer_ns() && (det.samples() % a.cfg.hop) == 0 && det.samples() >= det.config().hop) r.infer_ns.push_back(det.last_infer_ns());
    }
    det.finish();
    if (a.fast) {  // events from runs of immediate labels; onset known fast_confirm samples after it starts
        const int cf = a.cfg.fast_confirm;
        for (size_t i = 0; i < r.fpred.size();) {
            if (!r.fpred[i]) { ++i; continue; }
            size_t j = i; while (j + 1 < r.fpred.size() && r.fpred[j + 1]) ++j;
            if (int(j - i + 1) >= cf) { r.fevents.push_back({1, int64_t(i), int64_t(j), int64_t(j) + 2}); r.fonsets.push_back({int64_t(i), int64_t(i) + cf - 1}); }
            i = j + 1;
        }
    }
    return r;
}

void run_live(const Args& a) {
    auto detp = make_detector(a); auto& det = *detp;
    det.on_event = [&](const Event& e) {
        std::printf("EVENT cls=%d onset=%lld offset=%lld dur_ms=%.1f known_at=%lld\n", e.cls, (long long)e.onset, (long long)e.offset,
                    (e.offset - e.onset + 1) * 1000.0 / a.cfg.fs, (long long)(e.confirmed - 1));
        std::fflush(stdout);
    };
    if (a.per_sample)
        det.on_label = [&](const Label& l) { std::printf("LABEL %lld %d %.3f\n", (long long)l.index, l.cls, l.p_sacc); };
    std::string line;
    while (std::getline(std::cin, line)) {
        for (char& c : line) if (c == ',' || c == ';' || c == '\t') c = ' ';
        std::istringstream ss(line);
        double x, y;
        if (!(ss >> x >> y)) continue;  // skip headers / junk
        det.push(x, y);
    }
    det.finish();
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    auto need = [&](int& i) -> const char* { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); std::exit(2); } return argv[++i]; };
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--model") a.model = need(i);
        else if (s == "--window-ms") a.window_ms = std::atof(need(i));
        else if (s == "--fast") { a.fast = true; a.cfg.hop = 1; }
        else if (s == "--hop") a.cfg.hop = std::atoi(need(i));
        else if (s == "--lookahead") a.cfg.lookahead = std::atoi(need(i));
        else if (s == "--fs") { a.cfg.fs = a.sim.fs = std::atof(need(i)); }
        else if (s == "--threshold") a.cfg.threshold = std::atof(need(i));
        else if (s == "--min-dur") a.cfg.min_sacc_dur_ms = std::atof(need(i));
        else if (s == "--min-dist") a.cfg.min_sacc_dist_ms = std::atof(need(i));
        else if (s == "--scale") a.cfg.input_scale = std::atof(need(i));
        else if (s == "--sim") a.mode = "sim";
        else if (s == "--stdin") a.mode = "stdin";
        else if (s == "--replay") { a.mode = "replay"; a.xf = need(i); a.yf = need(i); if (i + 1 < argc && argv[i + 1][0] != '-') a.lf = argv[++i]; }
        else if (s == "--duration") a.sim.duration_s = std::atof(need(i));
        else if (s == "--seed") a.sim.seed = std::atoi(need(i));
        else if (s == "--noise") a.sim.noise_sd = std::atof(need(i));
        else if (s == "--rate") a.sim.rate_hz = std::atof(need(i));
        else if (s == "--realtime") a.realtime = true;
        else if (s == "--sweep-lookahead") a.sweep = true;
        else if (s == "--per-sample") a.per_sample = true;
        else { std::fprintf(stderr, "unknown option %s\n", s.c_str()); return 2; }
    }
    a.cfg.window_ms = a.window_ms;
    a.window = window_samples(a.cfg.fs, a.window_ms);
    if (a.mode.empty()) { std::fprintf(stderr, "usage: uneye_rt --model M (--sim | --replay X.csv Y.csv [Labels.csv] | --stdin) [--window-ms 200 --hop 1 --fast --lookahead 25 --fs 1000 --realtime --sweep-lookahead]\n"); return 2; }
    if (a.mode == "stdin") { run_live(a); return 0; }

    // gather traces
    std::vector<std::vector<double>> X, Y;
    std::vector<std::vector<int>> L;
    if (a.mode == "sim") {
        const auto tr = simulate(a.sim);
        X.push_back(tr.x); Y.push_back(tr.y); L.push_back(tr.label);
        std::printf("simulated %.0f s @ %.0f Hz, %zu saccades (%.0f%% micro), noise %.3f deg\n", a.sim.duration_s, a.sim.fs, tr.events.size(),
                    100.0 * std::count_if(tr.events.begin(), tr.events.end(), [](auto& e) { return e.amp < 1; }) / std::max<size_t>(tr.events.size(), 1), a.sim.noise_sd);
    } else {
        X = read_csv(a.xf); Y = read_csv(a.yf);
        if (!a.lf.empty()) {
            for (auto& r : read_csv(a.lf)) { std::vector<int> l(r.size()); for (size_t i = 0; i < r.size(); ++i) l[i] = int(r[i]); L.push_back(l); }
        }
        std::printf("replaying %zu trials of %zu samples @ %.0f Hz, time bin %.0f ms = %d samples, hop %d\n", X.size(), X[0].size(), a.cfg.fs, a.window_ms, a.window, a.cfg.hop);
    }
    std::vector<int> las = a.sweep ? std::vector<int>{0, 5, 10, 15, 25, 40, 60, 80} : std::vector<int>{a.cfg.lookahead};
    for (int la : las) {
        a.cfg.lookahead = la;
        Score S, SF;
        auto detp = make_detector(a); auto& det = *detp;
        for (size_t k = 0; k < X.size(); ++k) {
            Run r = stream_trace(det, a, X[k], Y[k], a.realtime);
            for (auto ns : r.infer_ns) { S.infer_us.push_back(ns / 1000.0); SF.infer_us.push_back(ns / 1000.0); }
            if (a.fast && k < L.size()) score_trace(SF, L[k], r.fpred, r.fevents, r.fonsets, X[k], Y[k], a.cfg.fs);
            if (k < L.size()) score_trace(S, L[k], r.pred, r.events, r.onsets, X[k], Y[k], a.cfg.fs);
            else for (auto& e : r.events) std::printf("EVENT onset=%lld offset=%lld\n", (long long)e.onset, (long long)e.offset);
        }
        if (!L.empty()) { report(a, S, la); if (a.fast) report(a, SF, 0, "fast"); }
    }
    return 0;
}

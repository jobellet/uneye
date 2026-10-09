// bench_latency: Step 3 of docs/ROADMAP.md. Time of each GazeEngine::push() (+ SafetyGuard::step()) over >= 1e6 samples.
// The input is generated BEFORE the measurement (fixed seed), the engine is warmed up on 2000 samples, then every push is timed with
// std::chrono::steady_clock calls OUTSIDE the engine. Raw latencies (float32 ns) are written to a binary file for the figure.
// This is NOT a worst-case execution time: a laptop OS has jitter, the observed maximum depends on what else runs.
// Usage (from cpp/): build/bench_latency [--samples 1000000] [--no-window] [--out ../docs/slides/data/latency.bin]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "uneye/causal_engine.hpp"
#include "uneye/engine.hpp"
#include "uneye/gaze_engine.hpp"
#include "uneye/safety_guard.hpp"

using namespace uneye;

namespace {
struct Rng { uint64_t s; explicit Rng(uint64_t x) : s(x * 2862933555777941757ULL + 3037000493ULL) {}
    double u() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<double>(s >> 11) / 9007199254740992.0; }
    double n() { double a = u(), b = u(); return std::sqrt(-2 * std::log(a + 1e-12)) * std::cos(6.283185307179586 * b); } };

std::vector<gaze::Sample> signal(long n) {          // fixation noise 0.01 deg + a main-sequence saccade every 250-850 ms
    Rng r(2026); std::vector<gaze::Sample> v(static_cast<size_t>(n));
    double x = 0, y = 0; long next = 300; int left = 0, len = 0; double ax = 0, ay = 0;
    for (long i = 0; i < n; ++i) {
        if (i == next) { const double amp = 0.2 + 7.0 * std::pow(r.u(), 2), th = 6.283185307 * r.u(); len = static_cast<int>(2.2 * amp + 21); left = len; ax = amp * std::cos(th); ay = amp * std::sin(th); next += 250 + static_cast<long>(600 * r.u()); }
        if (left > 0) { const double u0 = 1.0 - static_cast<double>(left) / len, u1 = u0 + 1.0 / len; const double a = 0.5 * (1 - std::cos(3.141592653589793 * u1)) - 0.5 * (1 - std::cos(3.141592653589793 * u0)); x += ax * a; y += ay * a; --left; }
        // keep the gaze within +-20 deg (a long random walk of saccades would leave the screen)
        if (std::fabs(x) > 20) x *= 0.5;
        if (std::fabs(y) > 20) y *= 0.5;
        v[static_cast<size_t>(i)] = gaze::Sample{static_cast<double>(i) * 1000.0, x + 0.01 * r.n(), y + 0.01 * r.n()};
    }
    return v;
}
double pct(std::vector<float> v, double q) { std::sort(v.begin(), v.end()); return v[static_cast<size_t>(q * static_cast<double>(v.size() - 1))]; }
}  // namespace

int main(int argc, char** argv) {
    long n = 1000000; bool window = true; std::string outp = "latency.bin", root = ".";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--samples" && i + 1 < argc) n = std::atol(argv[++i]);
        else if (a == "--no-window") window = false;
        else if (a == "--out" && i + 1 < argc) outp = argv[++i];
        else if (a == "--root" && i + 1 < argc) root = argv[++i];
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    gaze::Models m;
    m.fast = make_causal_engine(root + "/models/causal.bin");
    m.refine = make_causal_engine(root + "/models/tcn_l10.bin");
    if (window) m.window = make_native_engine(root + "/models/combined.bin", 200);
    gaze::GazeEngine eng(gaze::Config{}, std::move(m));
    safety::SafetyGuard guard; gaze::Output o; safety::GuardOutput go;
    const auto in = signal(n + 2000);
    for (long i = 0; i < 2000; ++i) { eng.push(in[static_cast<size_t>(i)], o); guard.step(in[static_cast<size_t>(i)], o, -1.0, go); }   // warm-up (caches)
    std::vector<float> ns(static_cast<size_t>(n));
    for (long i = 0; i < n; ++i) {
        const auto& s = in[static_cast<size_t>(i + 2000)];
        const auto t0 = std::chrono::steady_clock::now();
        eng.push(s, o);
        guard.step(s, o, -1.0, go);
        const auto t1 = std::chrono::steady_clock::now();
        ns[static_cast<size_t>(i)] = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }
    std::FILE* f = std::fopen(outp.c_str(), "wb");
    if (f) { std::fwrite(ns.data(), sizeof(float), ns.size(), f); std::fclose(f); }
    long over = 0; for (float v : ns) over += v > 1e6f;
    std::printf("samples %ld, window network %s | push+guard ns: median %.0f  p99 %.0f  p99.9 %.0f  max %.0f | over 1 ms: %ld\n", n, window ? "on" : "off",
                pct(ns, 0.5), pct(ns, 0.99), pct(ns, 0.999), pct(ns, 1.0), over);
    return 0;
}

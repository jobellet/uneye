// Engineering evidence for the GazeEngine: no allocation in steady state, bit-exact determinism, survival of hostile input (run this
// binary under -fsanitize=address,undefined and, separately, -fsanitize=thread), thread-safe hand-off, behaviour of the guards.
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "uneye/gaze_engine.hpp"
#include "uneye/spsc_queue.hpp"

// ---------------------------------------------------------------- global allocation counter
static std::atomic<long> g_allocs{0};
void* operator new(std::size_t n) { g_allocs.fetch_add(1, std::memory_order_relaxed); if (void* p = std::malloc(n ? n : 1)) return p; throw std::bad_alloc(); }
void* operator new[](std::size_t n) { g_allocs.fetch_add(1, std::memory_order_relaxed); if (void* p = std::malloc(n ? n : 1)) return p; throw std::bad_alloc(); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

using namespace uneye;
using namespace uneye::gaze;

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); ++g_fail; } } while (0)
static void section(const char* s) { std::printf("%s\n", s); }

// ---------------------------------------------------------------- synthetic gaze: fixation noise + saccades + blinks (deterministic)
struct Rng { uint64_t s; explicit Rng(uint64_t x) : s(x * 2862933555777941757ULL + 3037000493ULL) {}
    double u() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<double>(s >> 11) / 9007199254740992.0; }
    double n() { double a = u(), b = u(); return std::sqrt(-2 * std::log(a + 1e-12)) * std::cos(6.283185307179586 * b); } };

static std::vector<Sample> synth(int n, uint64_t seed, bool blinks) {
    Rng r(seed); std::vector<Sample> v(static_cast<size_t>(n));
    double x = 0, y = 0; int next = 300; int sac_left = 0, sac_len = 0; double ax = 0, ay = 0, px = 0, py = 0; int blink_left = 0;
    for (int i = 0; i < n; ++i) {
        if (i == next) { const double amp = 0.2 + 7.0 * std::pow(r.u(), 2), th = 6.283185307 * r.u(); sac_len = static_cast<int>(2.2 * amp + 21); sac_left = sac_len; ax = amp * std::cos(th); ay = amp * std::sin(th); px = x; py = y; next += 250 + static_cast<int>(600 * r.u()); }
        if (sac_left > 0) { const double u0 = 1.0 - static_cast<double>(sac_left) / sac_len, u1 = u0 + 1.0 / sac_len; const double a = 0.5 * (1 - std::cos(3.141592653589793 * u1)) - 0.5 * (1 - std::cos(3.141592653589793 * u0)); x += ax * a; y += ay * a; --sac_left; }
        (void)px; (void)py;
        Sample s; s.t_us = i * 1000.0; s.x_deg = x + 0.01 * r.n(); s.y_deg = y + 0.01 * r.n();
        if (blinks && blink_left == 0 && r.u() < 0.0008) blink_left = 60 + static_cast<int>(40 * r.u());
        if (blink_left > 0) { --blink_left; s.x_deg = std::nan(""); s.y_deg = std::nan(""); }
        v[static_cast<size_t>(i)] = s;
    }
    return v;
}

static Models load_models(const std::string& root, bool with_window = true) {
    Models m;
    m.fast = make_causal_engine(root + "/models/causal.bin");
    m.refine = make_causal_engine(root + "/models/tcn_l10.bin");
    m.refine_delay = 10;
    if (with_window) m.window = make_native_engine(root + "/models/combined.bin", 200);   // the original U'n'Eye as the third (finalizing) stage
    return m;
}

static bool same(const Output& a, const Output& b) {
    auto fe = [](double x, double y) { return std::memcmp(&x, &y, sizeof x) == 0; };
    auto ff = [](float x, float y) { return std::memcmp(&x, &y, sizeof x) == 0; };
    auto ie = [&](const Interval& p, const Interval& q) { return p.valid == q.valid && fe(p.x, q.x) && fe(p.y, q.y) && fe(p.scale_x, q.scale_x) && fe(p.scale_y, q.scale_y); };
    return a.index == b.index && a.state == b.state && ff(a.p_saccade, b.p_saccade) && a.source == b.source && a.flags == b.flags && a.health == b.health &&
           a.revised_first == b.revised_first && a.revised_last == b.revised_last && a.event_onset == b.event_onset && ie(a.forecast.at10, b.forecast.at10) &&
           ie(a.forecast.at20, b.forecast.at20) && a.forecast.ballistic == b.forecast.ballistic;
}

static bool sane(const Output& o) {
    if (!(o.p_saccade >= 0.0f && o.p_saccade <= 1.0f)) return false;
    for (const Interval* iv : {&o.forecast.at10, &o.forecast.at20})
        if (iv->valid && !(std::isfinite(iv->x) && std::isfinite(iv->y) && std::isfinite(iv->scale_x) && std::isfinite(iv->scale_y) && iv->scale_x >= 0 && iv->scale_y >= 0)) return false;
    return true;
}

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : ".";
    const auto data = synth(60000, 1, true);

    section("1. no heap allocation in steady state (push + snapshot), with all three networks");
    {
        GazeEngine eng(Config{}, load_models(root));
        Output o; Snapshot snap;
        for (int i = 0; i < 3000; ++i) { eng.push(data[static_cast<size_t>(i)], o); eng.snapshot(snap); }   // warm-up
        const long before = g_allocs.load();
        for (size_t i = 3000; i < data.size(); ++i) { eng.push(data[i], o); if (i % 7 == 0) eng.snapshot(snap); }
        const long used = g_allocs.load() - before;
        std::printf("  %zu pushes, allocations: %ld\n", data.size() - 3000, used);
        CHECK(used == 0, "steady-state push/snapshot must not allocate");
    }

    section("2. determinism: two engines, same input -> bit-identical output; reset() -> identical replay");
    {
        GazeEngine a(Config{}, load_models(root)), b(Config{}, load_models(root));
        Output oa, ob; bool ok = true;
        std::vector<Output> first;
        for (size_t i = 0; i < 20000; ++i) { a.push(data[i], oa); b.push(data[i], ob); ok = ok && same(oa, ob); first.push_back(oa); }
        CHECK(ok, "two engines diverged");
        a.reset(); bool ok2 = true;
        for (size_t i = 0; i < 20000; ++i) { a.push(data[i], oa); ok2 = ok2 && same(oa, first[i]); }
        CHECK(ok2, "replay after reset() differs from the first run");
        std::printf("  20000 samples compared bit by bit: %s\n", ok && ok2 ? "identical" : "DIFFERENT");
    }

    section("3. hostile input (NaN, inf, +-1e308, denormals, absurd / backwards / NaN timestamps, bursts): no crash, outputs sane");
    {
        GazeEngine eng(Config{}, load_models(root));
        Rng r(99); Output o; Snapshot snap; long bad = 0, resets = 0;
        const double evil[] = {std::nan(""), std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), 1e308, -1e308, 1e-310, 0.0, -0.0, 91.0, -91.0, 1e9, 5e-324};
        for (int i = 0; i < 400000; ++i) {
            Sample s = data[static_cast<size_t>(i % static_cast<int>(data.size()))];
            const double u = r.u();
            if (u < 0.02) s.x_deg = evil[static_cast<size_t>(r.u() * 12) % 12];
            else if (u < 0.04) s.y_deg = evil[static_cast<size_t>(r.u() * 12) % 12];
            else if (u < 0.05) s.t_us = evil[static_cast<size_t>(r.u() * 12) % 12];
            else if (u < 0.06) s.t_us = -5;
            else if (u < 0.065) s.t_us = s.t_us + 1e7 * r.u();                   // a long gap -> reset
            else if (u < 0.07) s.t_us = s.t_us - 5e4;                             // backwards
            else if (u < 0.0702) { for (int k = 0; k < 400; ++k) { Sample z; z.t_us = -1; z.x_deg = std::nan(""); z.y_deg = std::nan(""); eng.push(z, o); } }   // a long dropout burst
            eng.push(s, o);
            if (!sane(o)) ++bad;
            if (o.flags & flag::Reset) ++resets;
            if (i % 11 == 0) { eng.snapshot(snap); if (snap.n_bins > kMaxBins || snap.n_events > kMaxEvents) ++bad; for (int e = 0; e < snap.n_events; ++e) if (snap.events[static_cast<size_t>(e)].offset < snap.events[static_cast<size_t>(e)].onset) ++bad; }
        }
        std::printf("  400000 hostile pushes, insane outputs: %ld, resets triggered: %ld\n", bad, resets);
        CHECK(bad == 0, "an output violated its invariants");
        CHECK(resets > 0, "long gaps must reset the engine");
    }

    section("4. guards: blink, glitch, out of range, backwards time, unsupported rate");
    {
        Config c; GazeEngine eng(c, load_models(root)); Output o;
        auto sm = synth(2000, 5, false);
        for (int i = 0; i < 1000; ++i) eng.push(sm[static_cast<size_t>(i)], o);
        CHECK(o.state == State::Fixation || o.state == State::Saccade, "valid data must give a valid state");
        for (int i = 1000; i < 1060; ++i) { Sample s = sm[static_cast<size_t>(i)]; s.x_deg = std::nan(""); eng.push(s, o); }
        CHECK(o.state == State::Blink, "60 missing samples must be a blink");
        Snapshot snap; eng.snapshot(snap);
        int margin = 0; for (int k = 0; k < snap.n_bins; ++k) if (snap.bins[static_cast<size_t>(k)].index < 1000 && snap.bins[static_cast<size_t>(k)].index >= 970 && snap.bins[static_cast<size_t>(k)].state == State::Blink) ++margin;
        CHECK(margin >= 25, "the 30 ms before a blink must be revised to blink");
        for (int i = 1060; i < 1100; ++i) eng.push(sm[static_cast<size_t>(i)], o);
        CHECK(o.state != State::Blink, "after the margin the engine must recover");
        Sample g = sm[1100]; g.x_deg = 50.0; eng.push(g, o);                      // 1 sample 50 deg away: a glitch, not a blink
        CHECK(o.state == State::Invalid && (o.flags & flag::TooFast), "an impossible jump is an invalid sample");
        Sample ob = sm[1101]; ob.x_deg = 120.0; eng.push(ob, o);
        CHECK(o.flags & flag::OutOfRange, "out-of-range positions are flagged");
        Sample back = sm[1102]; back.t_us = 5.0; eng.push(back, o);
        CHECK(o.flags & flag::TimeBackwards, "a backwards timestamp is flagged");
        Config c2; c2.fs_hz = 250.0; GazeEngine slow(c2, load_models(root));
        slow.push(sm[0], o);
        CHECK(o.health == Health::Failed && (o.flags & flag::UnsupportedRate), "an unsupported sampling rate must fail loudly");
    }

    section("5. guard rail: a broken network cannot drive the output (no network -> heuristic only; NaN network -> fallback)");
    {
        struct NanNet : StepEngine { int classes() const override { return 2; } int receptive_field() const override { return 1; } void reset() override {} void step(float, float, float* p) override { p[0] = p[1] = NAN; } };
        struct OneNet : StepEngine { int classes() const override { return 2; } int receptive_field() const override { return 1; } void reset() override {} void step(float, float, float* p) override { p[0] = 0; p[1] = 1; } };
        auto quiet = synth(6000, 3, false); for (auto& s : quiet) { s.x_deg = 0.01 * std::sin(s.t_us / 5e4); s.y_deg = 0.0; }   // no movement at all
        for (int variant = 0; variant < 3; ++variant) {
            Models m; if (variant == 1) { m.fast = std::make_unique<NanNet>(); m.refine = std::make_unique<NanNet>(); } if (variant == 2) { m.fast = std::make_unique<OneNet>(); m.refine = std::make_unique<OneNet>(); }
            GazeEngine eng(Config{}, std::move(m)); Output o; long sacc = 0;
            for (size_t i = 0; i < quiet.size(); ++i) { eng.push(quiet[i], o); if (o.state == State::Saccade) ++sacc; }
            std::printf("  variant %d (%s): saccade labels on a motionless eye: %ld of %zu\n", variant, variant == 0 ? "no network" : variant == 1 ? "NaN network" : "always-saccade network", sacc, quiet.size());
            CHECK(sacc == 0, "no saccade may be declared when the data show no movement");
            if (variant == 1) CHECK(o.health == Health::Degraded, "a NaN network must make the engine Degraded");
        }
    }

    section("6. lock-free hand-off between a reader thread and the engine thread gives the same result as a single thread");
    {
        const size_t N = 30000;
        std::vector<Output> ref(N); { GazeEngine e(Config{}, load_models(root)); for (size_t i = 0; i < N; ++i) e.push(data[i], ref[i]); }
        static SpscQueue<Sample, 1024> q; std::vector<Output> got(N); std::atomic<bool> done{false};
        std::thread consumer([&] { GazeEngine e(Config{}, load_models(root)); size_t n = 0; Sample s;
            while (n < N) { if (q.try_pop(s)) e.push(s, got[n++]); else std::this_thread::yield(); } done = true; });
        for (size_t i = 0; i < N; ++i) while (!q.try_push(data[i])) std::this_thread::yield();
        consumer.join();
        bool ok = true; for (size_t i = 0; i < N; ++i) ok = ok && same(ref[i], got[i]);
        CHECK(ok, "threaded run differs from the single-threaded run");
        std::printf("  %zu samples through the queue: %s\n", N, ok ? "identical" : "DIFFERENT");
    }

    section("7. window stage (U'n'Eye finalizes bins at age 80): bins reach stage 2, a broken window network changes nothing");
    {
        auto sm = synth(4000, 11, false);
        GazeEngine eng(Config{}, load_models(root)); Output o; Snapshot snap;
        for (size_t i = 0; i < sm.size(); ++i) eng.push(sm[i], o);
        eng.snapshot(snap);
        int st2 = 0, st1 = 0, final_n = 0;
        for (int k = 0; k < snap.n_bins; ++k) { const auto& b = snap.bins[static_cast<size_t>(k)]; if (b.stage == 2) ++st2; if (b.stage == 1) ++st1; if (b.final) ++final_n; }
        std::printf("  last 100 bins: stage 2 (finalized by the window network): %d, stage 1: %d, final: %d\n", st2, st1, final_n);
        CHECK(st2 >= 5 && st2 <= 100, "old bins must be finalized by the window network");
        CHECK(snap.bins[static_cast<size_t>(snap.n_bins - 1)].stage == 0, "the newest bin has only the fast label");
        struct NanWin : Engine { int window() const override { return 200; } int classes() const override { return 2; } void infer(const float*, float* p) override { for (int i = 0; i < 400; ++i) p[i] = NAN; } };
        Models m = load_models(root, false); m.window = std::make_unique<NanWin>();
        GazeEngine bad(Config{}, std::move(m)); GazeEngine ref(Config{}, load_models(root, false));
        Output ob, orf; bool ok = true;
        for (size_t i = 0; i < sm.size(); ++i) { bad.push(sm[i], ob); ref.push(sm[i], orf); ok = ok && ob.state == orf.state && ob.flags == orf.flags; }
        CHECK(ok, "a window network that outputs NaN must not change any label");
        CHECK(bad.stats().window_invalid > 0 && bad.stats().window_runs == bad.stats().window_invalid, "every NaN window run must be counted as invalid");
        std::printf("  NaN window network: %lld runs, all discarded; labels identical to the engine without a window network: %s\n", (long long)bad.stats().window_runs, ok ? "yes" : "NO");
    }

    std::printf("\n%s (%d failed checks)\n", g_fail == 0 ? "ALL CHECKS PASSED" : "FAILURES", g_fail);
    return g_fail == 0 ? 0 : 1;
}

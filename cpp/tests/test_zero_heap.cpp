// Proof test: after the engine is built, processing samples performs ZERO dynamic allocations.
//
// How it works: this executable replaces the global allocation functions. Inside the measured region every call to operator new / new[] /
// malloc / calloc / realloc / aligned allocation (and every free / delete) increments a counter. The region starts when the engine is
// fully built (no warm-up: the very first window-network run is inside the region) and ends after 100000 samples.
// A control block allocates on purpose and the test fails if the hooks did NOT see it, so a "0" cannot come from a broken hook.
//
// Limits: only the allocation functions of this process are seen. Stack use is not measured. Under a sanitizer the malloc hook is switched off
// (the sanitizer owns malloc) and only operator new / delete are counted; the test prints which hooks are active.
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

#include "uneye/gaze_engine.hpp"

// This file IS an allocator replacement (new -> malloc, delete -> free); GCC cannot see that and warns about "mismatched" pairs under sanitizers.
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"

using namespace uneye;
using namespace uneye::gaze;

// ---------------------------------------------------------------- allocation counters (only while armed)
static std::atomic<bool> g_armed{false};
static std::atomic<long> g_new{0}, g_malloc{0}, g_free{0};
static inline void count(std::atomic<long>& c) { if (g_armed.load(std::memory_order_relaxed)) c.fetch_add(1, std::memory_order_relaxed); }

#ifndef UNEYE_NO_MALLOC_HOOK
extern "C" {
void* __libc_malloc(size_t); void* __libc_calloc(size_t, size_t); void* __libc_realloc(void*, size_t); void __libc_free(void*); void* __libc_memalign(size_t, size_t);
void* malloc(size_t n) { count(g_malloc); return __libc_malloc(n); }
void* calloc(size_t a, size_t b) { count(g_malloc); return __libc_calloc(a, b); }
void* realloc(void* p, size_t n) { count(g_malloc); return __libc_realloc(p, n); }
void free(void* p) { if (p) count(g_free); __libc_free(p); }
void* memalign(size_t al, size_t n) { count(g_malloc); return __libc_memalign(al, n); }
void* aligned_alloc(size_t al, size_t n) { count(g_malloc); return __libc_memalign(al, n); }
int posix_memalign(void** out, size_t al, size_t n) { count(g_malloc); *out = __libc_memalign(al, n); return *out ? 0 : 12; }
}
static const bool kMallocHook = true;
#else
static const bool kMallocHook = false;
#endif

void* operator new(std::size_t n) { count(g_new); if (void* p = std::malloc(n ? n : 1)) return p; throw std::bad_alloc(); }
void* operator new[](std::size_t n) { count(g_new); if (void* p = std::malloc(n ? n : 1)) return p; throw std::bad_alloc(); }
void* operator new(std::size_t n, std::align_val_t al) { count(g_new); void* p = nullptr; if (posix_memalign(&p, static_cast<size_t>(al), n ? n : 1) != 0) throw std::bad_alloc(); return p; }
void* operator new[](std::size_t n, std::align_val_t al) { count(g_new); void* p = nullptr; if (posix_memalign(&p, static_cast<size_t>(al), n ? n : 1) != 0) throw std::bad_alloc(); return p; }
void operator delete(void* p) noexcept { if (p) count(g_free); std::free(p); }
void operator delete[](void* p) noexcept { if (p) count(g_free); std::free(p); }
void operator delete(void* p, std::size_t) noexcept { if (p) count(g_free); std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { if (p) count(g_free); std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { if (p) count(g_free); std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { if (p) count(g_free); std::free(p); }

struct Counts { long n, m, f; };
static Counts snap() { return {g_new.load(), g_malloc.load(), g_free.load()}; }
static long total(const Counts& a, const Counts& b) { return (b.n - a.n) + (b.m - a.m); }   // operator new calls malloc, so this counts a "new" twice: fine for a ZERO test

// ---------------------------------------------------------------- deterministic test signal (fixed seed, no <random> so it is the same everywhere)
static uint32_t lcg(uint32_t& s) { s = s * 1664525u + 1013904223u; return s >> 8; }
static double unif(uint32_t& s) { return lcg(s) / 16777216.0; }

static std::vector<Sample> make_signal(int n, uint32_t seed) {
    std::vector<Sample> v; v.reserve(static_cast<size_t>(n));
    uint32_t s = seed; double x = 0, y = 0, tx = 0, ty = 0; int sacc_left = 0; double vx = 0, vy = 0; double t = 0;
    for (int i = 0; i < n; ++i) {
        if (sacc_left == 0 && unif(s) < 0.004) {                       // start a saccade: 20 samples, amplitude 0.3 .. 8 deg
            const double amp = 0.3 + 7.7 * unif(s), ang = 6.2831853 * unif(s);
            vx = amp * std::cos(ang) / 20.0; vy = amp * std::sin(ang) / 20.0; sacc_left = 20;
        }
        if (sacc_left > 0) { x += vx; y += vy; --sacc_left; }
        tx = x + 0.01 * (unif(s) - 0.5); ty = y + 0.01 * (unif(s) - 0.5);
        t += 1000.0;
        Sample sm; sm.t_us = t; sm.x_deg = tx; sm.y_deg = ty;
        // hostile bits, on purpose: NaN burst, huge value, backwards time, a gap of missing samples, a long dropout (blink)
        if (i % 9000 == 4000) sm.x_deg = std::nan("");
        if (i % 9000 == 4001) sm.y_deg = 1e9;
        if (i % 9000 == 5000) sm.t_us = t - 5000.0;
        if (i % 9000 == 6000) t += 7000.0;
        if (i % 9000 >= 7000 && i % 9000 < 7040) { sm.x_deg = std::nan(""); sm.y_deg = std::nan(""); }
        v.push_back(sm);
    }
    return v;
}

static int failed = 0;
#define CHECK(c, msg) do { if (!(c)) { std::printf("  FAIL: %s\n", msg); ++failed; } } while (0)

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : ".";
    std::printf("hooks: operator new/delete + %s\n", kMallocHook ? "malloc/calloc/realloc/free/aligned (glibc __libc_*)" : "(malloc hook OFF: sanitizer build)");

    // ---- control: the hooks must see allocations (otherwise a result of 0 means nothing)
    {
        const Counts a = (g_armed = true, snap());
        int* p = new int[16]; p[0] = 1;
        std::vector<int> v; v.push_back(p[0]);
        void* q = std::malloc(64);
        const Counts b = snap(); g_armed = false;
        std::free(q); delete[] p;
        std::printf("control: deliberate allocations seen by the hooks: new=%ld malloc=%ld\n", b.n - a.n, b.m - a.m);
        CHECK(b.n - a.n >= 2, "hook does not see operator new");
        if (kMallocHook) CHECK(b.m - a.m >= 3, "hook does not see malloc");
    }

    // ---- the window network ALONE: infer() right after construction (no hidden "call it once first" rule), 1000 runs
    {
        auto net = make_native_engine(root + "/models/combined.bin", 200);
        std::vector<float> in(2 * 200), prob(2 * 200);
        uint32_t sd = 7; for (auto& v : in) v = static_cast<float>(unif(sd) - 0.5) * 0.05f;
        const Counts a = (g_armed = true, snap());
        for (int i = 0; i < 1000; ++i) net->infer(in.data(), prob.data());
        const Counts b = snap(); g_armed = false;
        std::printf("native U-Net alone, 1000 infer() calls from the first one: operator new = %ld, malloc family = %ld\n", b.n - a.n, b.m - a.m);
        CHECK(total(a, b) == 0, "NativeEngine::infer allocated");
    }

    // ---- everything that may allocate happens here: models, engine, output objects, the test signal
    Models m;
    m.fast = make_causal_engine(root + "/models/causal.bin");
    m.refine = make_causal_engine(root + "/models/tcn_l10.bin");
    m.refine_delay = 10;
    m.window = make_native_engine(root + "/models/combined.bin", 200);
    GazeEngine eng(Config{}, std::move(m));
    Output out; Snapshot snapshot;
    const auto data = make_signal(100000, 12345u);

    // ---- measured region: no warm-up. The first window-network run is inside.
    const Counts before = (g_armed = true, snap());
    long window_runs = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        eng.push(data[i], out);
        if (i % 7 == 0) eng.snapshot(snapshot);
    }
    const Counts after = snap(); g_armed = false;
    window_runs = static_cast<long>(eng.stats().window_runs);

    std::printf("%zu pushes (+%zu snapshots), window-network runs: %ld, resets: %ld, invalid samples: %ld\n", data.size(), data.size() / 7 + 1, window_runs,
                static_cast<long>(eng.stats().resets), static_cast<long>(eng.stats().invalid));
    std::printf("allocations in the measured region: operator new = %ld, malloc family = %ld   (frees: delete/free = %ld)\n",
                after.n - before.n, after.m - before.m, after.f - before.f);
    CHECK(window_runs > 1000, "the window network did not run: the test would not cover the allocating path");
    CHECK(total(before, after) == 0, "the hot path allocated");
    std::printf(failed ? "FAILED\n" : "PASSED: zero allocations after init\n");
    return failed ? 1 : 0;
}

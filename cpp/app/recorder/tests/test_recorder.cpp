// T01: sizing, crash safety (forked child on POSIX), concurrent reader / writer, full file, error paths, CSV and label / expert / event round trips, push throughput.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include "eyelive/recorder.hpp"
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif
using namespace eyelive;
namespace fs = std::filesystem;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static std::string tmp(const std::string& name) { fs::path d = fs::temp_directory_path() / "eyelive_recorder_test"; fs::create_directories(d); fs::path p = d / name; std::error_code ec; fs::remove(p, ec); return p.string(); }
static RecorderConfig cfg_for(const std::string& path, double rate, double dur, bool reserve = false) { RecorderConfig c; c.path = path; c.rate_hz = rate; c.duration_s = dur; c.reserve_disk = reserve; c.lock_memory = false; return c; }
static Sample sample(int64_t i) { Sample s; s.t = double(i) * 0.001; s.x = float(i); s.y = float(-i); s.flags = (i % 97 == 0) ? (kFlagBlink) : kFlagValid; return s; }

int main() {
    std::string err;
    // 1. sizing: 19 bytes per sample + 512 header + events table
    {
        const auto sz = recorder_sizing(1000, 3600); const double expect = 19.0 * double(sz.capacity_samples) + 512 + 100000.0 * sizeof(Event) + 8;
        std::printf("sizing 1 kHz x 1 h: %llu samples, %.2f MB (formula %.2f MB)\n", (unsigned long long)sz.capacity_samples, sz.bytes / 1e6, expect / 1e6);
        CHECK(std::fabs(double(sz.bytes) - expect) / expect < 0.01, "sizing off by more than 1 %%"); CHECK(sz.capacity_samples == uint64_t(std::ceil(1000 * 3600 * 1.05)) + 1024, "capacity formula");
    }
    // 2. create errors return a message, no exception
    {
        auto r = Recorder::create(cfg_for("/nonexistent_dir_eyelive/x.eyelive", 1000, 1), &err); CHECK(!r && !err.empty(), "missing directory must fail with a message"); std::printf("missing directory -> '%s'\n", err.c_str());
        err.clear(); r = Recorder::create(cfg_for(tmp("huge.eyelive"), 1e6, 1e9), &err); CHECK(!r && !err.empty(), "impossible size must fail"); std::printf("impossible size -> '%s'\n", err.c_str());
        err.clear(); r = Recorder::create(cfg_for(tmp("neg.eyelive"), -5, 10), &err); CHECK(!r && !err.empty(), "invalid rate must fail");
        const std::string p = tmp("dup.eyelive"); auto a = Recorder::create(cfg_for(p, 1000, 1), &err); CHECK(a != nullptr, "create: %s", err.c_str()); err.clear();
        auto b = Recorder::create(cfg_for(p, 1000, 1), &err); CHECK(!b && !err.empty(), "creating over an existing session must fail"); std::printf("existing file -> '%s'\n", err.c_str());
        err.clear(); auto bad = tmp("bad.eyelive"); { std::ofstream f(bad, std::ios::binary); f << "this is not a session file at all, it is just text padded to be long enough for a header check" << std::string(600, 'x'); }
        auto o = Recorder::open_readonly(bad, &err); CHECK(!o && !err.empty(), "bad magic must fail"); err.clear(); o = Recorder::open_readonly("/nonexistent_eyelive_file", &err); CHECK(!o && !err.empty(), "missing file must fail");
    }
    // 3. push returns false exactly when full; count, capacity
    {
        auto r = Recorder::create(cfg_for(tmp("full.eyelive"), 10, 1), &err); CHECK(r != nullptr, "create: %s", err.c_str());
        if (r) { const int64_t cap = r->capacity(); CHECK(cap == int64_t(std::ceil(10 * 1 * 1.05)) + 1024, "capacity %lld", (long long)cap);
            int64_t ok = 0; for (int64_t i = 0; i < cap + 5; ++i) { const bool p = r->push(sample(i)); if (p) ++ok; CHECK(p == (i < cap), "push(%lld) returned %d", (long long)i, int(p)); }
            CHECK(ok == cap && r->size() == cap, "size %lld of %lld", (long long)r->size(), (long long)cap); }
    }
    // 4. label / expert / event / state block / CSV round trips, session info
    {
        RecorderConfig c = cfg_for(tmp("rt.eyelive"), 500, 4); c.session.deg_per_unit = 0.0123; c.session.scale_estimated = true; c.session.n_classes = 4; c.session.class_names = {"fixation", "saccade", "blink", "pursuit"}; c.session.rate_hz = 500;
        auto r = Recorder::create(c, &err); CHECK(r != nullptr, "create: %s", err.c_str());
        if (r) {
            for (int64_t i = 0; i < 1000; ++i) r->push(sample(i));
            r->set_label(10, 1); r->set_label(11, 3); r->set_expert(10, 2); r->set_expert(5000, 1);               // the last one is beyond size(): ignored
            CHECK(r->label(10) == 1 && r->label(11) == 3 && r->label(12) == 0, "labels"); CHECK(r->expert(10) == 2 && r->expert(11) == 255, "expert labels (255 = none)");
            Event e; e.onset = 100; e.offset = 130; e.t_onset = 0.1; e.t_offset = 0.13; e.amplitude_deg = 4.5f; e.peak_speed_deg_s = 300.f; e.cls = 1; e.origin = EventOrigin::Expert; e.confidence = 0.75f;
            CHECK(r->append_event(e), "append_event"); e.onset = 400; CHECK(r->append_event(e), "append_event 2");
            CHECK(r->event_count() == 2 && r->event(0).onset == 100 && r->event(1).onset == 400 && r->event(0).amplitude_deg == 4.5f && r->event(0).origin == EventOrigin::Expert && r->event(0).confidence == 0.75f, "event round trip");
            auto* sb = static_cast<unsigned char*>(r->state_block()); CHECK(sb != nullptr, "state block"); for (int i = 0; i < 256; ++i) sb[i] = (unsigned char)i;          // writable, 256 bytes, does not disturb the data
            CHECK(r->x(999) == 999.f && r->label(10) == 1, "state block write corrupted the data");
            const std::string csv = tmp("rt.csv"); CHECK(r->export_csv(csv, &err), "export_csv: %s", err.c_str());
            std::ifstream f(csv); std::string line; std::getline(f, line); CHECK(line == "index,t,x,y,flags,label,expert", "csv header '%s'", line.c_str());
            bool exact = true; for (int64_t i = 0; i < 1000 && std::getline(f, line); ++i) { std::istringstream is(line); long long idx; double tt; float xx, yy; unsigned fl, lb, ex; char comma;
                is >> idx >> comma >> tt >> comma >> xx >> comma >> yy >> comma >> fl >> comma >> lb >> comma >> ex; const Sample s = sample(i); if (idx != i || tt != s.t || xx != s.x || yy != s.y || fl != (s.flags & 0xFF) || lb != r->label(i) || ex != r->expert(i)) { exact = false; break; } }
            CHECK(exact, "CSV round trip differs"); CHECK(r->close(&err), "close: %s", err.c_str());
            auto o = Recorder::open_readonly(c.path, &err); CHECK(o != nullptr, "open_readonly: %s", err.c_str());
            if (o) { CHECK(o->size() == 1000 && o->x(500) == 500.f && o->t(500) == 0.5 && o->label(11) == 3 && o->expert(10) == 2 && o->event_count() == 2, "reopened data differ");
                     CHECK(std::fabs(o->info().deg_per_unit - 0.0123) < 1e-12 && o->info().scale_estimated && o->info().n_classes == 4 && o->info().class_names.size() == 4 && o->info().class_names[3] == "pursuit" && o->info().rate_hz == 500, "session info lost");
                     CHECK(!o->push(sample(1)), "a read-only recorder must not accept samples"); }
        }
    }
    // 5. crash safety: a child process writes 1e6 samples at full speed and dies without close(); the parent reopens the file
#ifndef _WIN32
    {
        const std::string p = tmp("crash.eyelive"); const int64_t N = 1000000; const pid_t pid = fork();
        if (pid == 0) { std::string e2; auto r = Recorder::create(cfg_for(p, 1000, 1100), &e2); if (!r) _exit(2); for (int64_t i = 0; i < N; ++i) if (!r->push(sample(i))) _exit(3); Event e; e.onset = 7; r->append_event(e); r->set_label(123, 1); _exit(0); }   // no close(), no destructor
        int status = 0; waitpid(pid, &status, 0); CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0, "child failed with status %d", status);
        auto o = Recorder::open_readonly(p, &err); CHECK(o != nullptr, "reopen after crash: %s", err.c_str());
        if (o) { CHECK(o->size() == N, "count %lld != %lld", (long long)o->size(), (long long)N); bool ok = true; for (int64_t i = 0; i < N && ok; ++i) ok = (o->x(i) == float(i) && o->y(i) == float(-i) && o->t(i) == double(i) * 0.001 && o->flags(i) == sample(i).flags); CHECK(ok, "a sample is corrupted after the crash");
                 CHECK(o->event_count() == 1 && o->event(0).onset == 7 && o->label(123) == 1, "event / label lost after the crash"); std::printf("crash test: %lld samples intact\n", (long long)o->size()); }
    }
#endif
    // 6. a writer thread pushes 2e6 samples while a reader thread checks x(i) == i for every i < size()
    {
        auto r = Recorder::create(cfg_for(tmp("race.eyelive"), 1000, 2100), &err); CHECK(r != nullptr, "create: %s", err.c_str());
        if (r) {
            const int64_t N = 2000000; std::atomic<bool> done{false}, torn{false}; std::atomic<int64_t> checked{0};
            std::thread reader([&] { int64_t next = 0; while (!done.load(std::memory_order_acquire) || next < r->size()) { const int64_t n = r->size(); for (; next < n; ++next) { if (r->x(next) != float(next) || r->y(next) != float(-next) || r->t(next) != double(next) * 0.001) { torn = true; return; } (void)r->label(next); } } checked = next; });
            std::thread labeller([&] { while (!done.load(std::memory_order_acquire)) { const int64_t n = r->size(); if (n > 100) r->set_label(n - 100, 1); } });
            for (int64_t i = 0; i < N; ++i) r->push(sample(i));
            done = true; reader.join(); labeller.join();
            CHECK(!torn, "torn read detected"); CHECK(checked == N, "reader checked %lld of %lld", (long long)checked.load(), (long long)N); std::printf("concurrent test: %lld samples checked, no torn read\n", (long long)checked.load());
        }
    }
    // 7. throughput of push
    {
        const int64_t N = 5000000; auto r = Recorder::create(cfg_for(tmp("speed.eyelive"), 1000, 5100), &err); CHECK(r != nullptr, "create: %s", err.c_str());
        if (r) { Sample s; const auto t0 = std::chrono::steady_clock::now(); for (int64_t i = 0; i < N; ++i) { s.t = double(i); s.x = float(i); r->push(s); } const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("push throughput: %.1f million samples/s (single thread)\n", N / dt / 1e6); CHECK(N / dt > 1e6, "throughput below 1e6 samples/s"); }
    }
    // 8. disk reservation (small file): the file has exactly the size given by recorder_sizing
    {
        auto r = Recorder::create(cfg_for(tmp("reserve.eyelive"), 100, 10, true), &err); CHECK(r != nullptr, "create with reserve_disk: %s", err.c_str());
        if (r) { r->push(sample(0)); CHECK(fs::file_size(r->path()) == recorder_sizing(100, 10).bytes, "file size differs from recorder_sizing"); }
    }
    std::error_code ec; fs::remove_all(fs::temp_directory_path() / "eyelive_recorder_test", ec);
    std::printf(fails ? "FAILED\n" : "all recorder tests passed\n"); return fails ? 1 : 0;
}

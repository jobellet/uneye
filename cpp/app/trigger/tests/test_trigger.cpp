// T19: trigger engine. Timing of 200 delayed triggers, rule matching (class, amplitude, anchors, refractory, enable / disable, rule changes), late flag, stop() dropping pending triggers, UDP sink with placeholders, no allocation in on_event.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <new>
#include <thread>
#include "eyelive/trigger.hpp"
#include "eyelive/trigger_clock.hpp"
#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
using namespace eyelive;

static std::atomic<long> g_allocs{0}; static std::atomic<bool> g_count{false};
void* operator new(size_t n) { if (g_count) ++g_allocs; if (void* p = std::malloc(n ? n : 1)) return p; throw std::bad_alloc(); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
static void sleep_ms(double ms) { std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(ms)); }
static Event ev(double t_onset, uint8_t cls, float amp, bool provisional, double dur_s = 0.03) { Event e; e.t_onset = t_onset; e.t_offset = t_onset + dur_s; e.cls = cls; e.amplitude_deg = provisional ? 0.f : amp; e.provisional = provisional; e.confidence = 1.0f; return e; }
static TriggerRule rule(const char* name, uint8_t cls, double delay_ms = 0, TriggerAnchor a = TriggerAnchor::OnsetProvisional) { TriggerRule r; r.name = name; r.cls = cls; r.delay_ms = delay_ms; r.anchor = a; r.sink = "null"; return r; }
static size_t count_of(ITriggerEngine& t, const std::string& name) { size_t n = 0; for (auto& e : t.log_snapshot()) n += e.rule == name; return n; }

int main() {
    // 1. timing: 200 triggers 10 ms after a synthetic onset (the sample "arrives" now), events every 4 ms so that several are pending at once
    {
        auto t = make_trigger_engine(); TriggerRule r = rule("timing", 1, 10.0); t->set_rules({r}); t->start(); const int N = 200;
        for (int i = 0; i < N; ++i) { t->on_event(ev(0.001 * i, 1, 0, true), trigger_host_now_s()); sleep_ms(4); }
        sleep_ms(50); auto log = t->log_snapshot(); t->stop(); CHECK(int(log.size()) == N, "%zu of %d triggers fired", log.size(), N);
        std::vector<double> err; for (auto& e : log) { err.push_back((e.fired_host_s - e.requested_host_s) * 1000.0); CHECK(!e.late, "a trigger was marked late"); } std::sort(err.begin(), err.end());
        const double med = err[err.size() / 2], p95 = err[size_t(0.95 * (err.size() - 1))], mx = err.back(), mn = err.front();
        std::printf("200 triggers at +10 ms: fired - requested: min %.3f ms, median %.3f ms, p95 %.3f ms, max %.3f ms\n", mn, med, p95, mx);
        CHECK(std::fabs(med) < 1.0, "median error %.3f ms", med); CHECK(p95 < 3.0, "p95 error %.3f ms", p95);
    }
    // 2. matching
    {
        auto t = make_trigger_engine(); std::vector<TriggerRule> rs;
        TriggerRule a = rule("sacc", 1); TriggerRule b = rule("any", 0); b.any_class = true; TriggerRule c = rule("micro", 1); c.amp_max_deg = 1.0f; c.anchor = TriggerAnchor::OnsetFinal; TriggerRule d = rule("big", 1); d.amp_min_deg = 3.0f; d.anchor = TriggerAnchor::OnsetFinal;
        TriggerRule e = rule("blink", 2); TriggerRule f = rule("refr", 1); f.refractory_ms = 100.0; TriggerRule g = rule("off", 1, 0, TriggerAnchor::Offset); TriggerRule h = rule("disabled", 1); h.enabled = false; TriggerRule k = rule("amp_prov", 1); k.amp_max_deg = 1.0f;
        t->set_rules({a, b, c, d, e, f, g, h, k}); t->start(); const double now = trigger_host_now_s() - 5.0;            // arrival times in the past: every trigger is already due (marked late) and fires at once; refractory periods compare the anchor times
        t->on_event(ev(1.0, 1, 0, true), now);                    // provisional saccade: sacc, any, refr
        t->on_event(ev(1.0, 1, 0.5f, false), now);                // final, 0.5 deg: micro, off
        t->on_event(ev(2.0, 1, 5.0f, false), now + 1.0);          // final, 5 deg: big, off (refractory does not apply to onset-final rules)
        t->on_event(ev(3.0, 2, 0, true), now + 2.0);              // provisional blink: any, blink
        t->on_event(ev(1.05, 1, 0, true), now + 0.05);            // 50 ms after the first: refr is in its refractory period
        t->on_event(ev(1.30, 1, 0, true), now + 0.30);            // 300 ms later: refr fires again
        Event retract = ev(1.0, 1, 0, false); retract.confidence = 0.0f; t->on_event(retract, now + 0.1);   // retraction: nothing
        sleep_ms(100); auto S = [&](const char* n) { return count_of(*t, n); };
        std::printf("matching: sacc %zu, any %zu, micro %zu, big %zu, blink %zu, refr %zu, off %zu, disabled %zu, amp_prov %zu\n", S("sacc"), S("any"), S("micro"), S("big"), S("blink"), S("refr"), S("off"), S("disabled"), S("amp_prov"));
        CHECK(S("sacc") == 3, "sacc %zu (3 provisional class-1 events)", S("sacc")); CHECK(S("any") == 4, "any %zu (4 provisional events)", S("any")); CHECK(S("micro") == 1, "micro %zu", S("micro")); CHECK(S("big") == 1, "big %zu", S("big"));
        CHECK(S("blink") == 1, "blink %zu (class 2 provisional only)", S("blink")); CHECK(S("refr") == 2, "refr %zu (refractory 100 ms)", S("refr")); CHECK(S("off") == 2, "off %zu (2 final class-1 events)", S("off"));
        CHECK(S("disabled") == 0, "disabled rule fired"); CHECK(S("amp_prov") == 0, "a rule with an amplitude range cannot fire on a provisional event (no amplitude yet)");
        // rule changes while running: enable the disabled rule, remove the others
        h.enabled = true; t->set_rules({h}); t->on_event(ev(5.0, 1, 0, true), trigger_host_now_s()); sleep_ms(50); CHECK(S("disabled") == 1, "enabled rule did not fire"); CHECK(S("sacc") == 3, "removed rule still fires"); t->stop();
    }
    // 3. offset anchor timing and the late flag
    {
        auto t = make_trigger_engine(); TriggerRule r = rule("off", 1, 20.0, TriggerAnchor::Offset); TriggerRule l = rule("late", 1, 10.0); t->set_rules({r, l}); t->start(); const double now = trigger_host_now_s();
        t->on_event(ev(10.0, 1, 2.0f, false, 0.040), now);                                   // offset 40 ms after the onset arrival, +20 ms delay: requested now + 0.060
        t->on_event(ev(10.0, 1, 0, true), now - 0.050);                                      // the onset arrived 50 ms ago, delay 10 ms: already late -> fires at once, marked
        sleep_ms(120); auto log = t->log_snapshot(); t->stop(); CHECK(log.size() == 2, "%zu entries", log.size());
        for (auto& e : log) { if (e.rule == "late") { CHECK(e.late, "late flag missing"); CHECK(e.fired_host_s - now < 0.02, "late trigger fired %.1f ms after the call", (e.fired_host_s - now) * 1000); }
            else { CHECK(!e.late, "offset trigger marked late"); CHECK(std::fabs(e.requested_host_s - (now + 0.060)) < 1e-3, "offset requested %.4f vs %.4f", e.requested_host_s - now, 0.060); CHECK(e.fired_host_s >= e.requested_host_s - 1e-4, "fired before requested"); } }
    }
    // 4. stop() drops triggers whose delay has not elapsed
    { auto t = make_trigger_engine(); t->set_rules({rule("slow", 1, 500.0)}); t->start(); t->on_event(ev(1.0, 1, 0, true), trigger_host_now_s()); sleep_ms(20); t->stop(); sleep_ms(20); CHECK(t->log_snapshot().empty(), "a pending trigger fired after stop()"); t->start(); t->stop(); }
    // 5. UDP sink delivers the payload with the placeholders substituted
#ifndef _WIN32
    {
        const int rx = ::socket(AF_INET, SOCK_DGRAM, 0); sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = 0; CHECK(::bind(rx, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0, "bind");
        socklen_t len = sizeof a; ::getsockname(rx, reinterpret_cast<sockaddr*>(&a), &len); const int port = ntohs(a.sin_port); timeval tv{1, 0}; ::setsockopt(rx, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        auto t = make_trigger_engine(); TriggerRule r = rule("udp", 1); r.sink = "udp:127.0.0.1:" + std::to_string(port); r.payload = "T t={t} amp={amp} cls={cls} {unknown}"; t->set_rules({r}); t->start();
        t->on_event(ev(1.5, 1, 2.25f, false), trigger_host_now_s()); TriggerRule r2 = r; r2.anchor = TriggerAnchor::OnsetFinal; t->set_rules({r2}); t->on_event(ev(1.5, 1, 2.25f, false), trigger_host_now_s());
        char buf[512] = {0}; const long n = ::recvfrom(rx, buf, sizeof buf - 1, 0, nullptr, nullptr); t->stop(); ::close(rx);
        std::printf("udp payload received: '%.*s'\n", int(std::max(0L, n)), buf); CHECK(n > 0 && std::string(buf, size_t(n)) == "T t=1.500000 amp=2.250 cls=1 {unknown}", "UDP payload '%s'", buf);
        // a sink to a closed port, a bad sink spec: the engine must not block, crash or throw
        auto t2 = make_trigger_engine(); TriggerRule bad = rule("bad", 1); bad.sink = "udp:127.0.0.1:9"; TriggerRule bad2 = rule("bad2", 1); bad2.sink = "serial:COM1"; t2->set_rules({bad, bad2}); t2->start(); t2->on_event(ev(1.0, 1, 0, true), trigger_host_now_s()); sleep_ms(30); t2->stop();
    }
#endif
    // 6. no allocation in on_event after the first call
    {
        auto t = make_trigger_engine(); TriggerRule r = rule("alloc", 1, 1.0); r.payload = "T {t} {amp} {cls}"; TriggerRule r2 = rule("any", 0); r2.any_class = true; r2.anchor = TriggerAnchor::OnsetFinal; t->set_rules({r, r2}); t->start();
        t->on_event(ev(1.0, 1, 0, true), trigger_host_now_s()); sleep_ms(10); g_allocs = 0; g_count = true;
        for (int i = 0; i < 1000; ++i) { t->on_event(ev(2.0 + i, 1, 1.0f, i % 2 == 0), trigger_host_now_s()); }
        g_count = false; const long n = g_allocs; sleep_ms(30); t->stop(); std::printf("heap allocations during 1000 on_event calls: %ld\n", n); CHECK(n == 0, "%ld allocations in on_event", n);
    }
    std::printf(fails ? "FAILED\n" : "all trigger tests passed\n"); return fails ? 1 : 0;
}

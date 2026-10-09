// Tests of the deterministic safety guard (include/uneye/safety_guard.hpp): one test per check, the state machine and its hysteresis,
// "the network is ignored in SAFE", determinism, and the guard behind the real engine on clean synthetic gaze (no false alarm).
// Most tests feed the guard a hand-made engine Output, so that each rule is tested alone.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "uneye/causal_engine.hpp"
#include "uneye/engine.hpp"
#include "uneye/gaze_engine.hpp"
#include "uneye/safety_guard.hpp"

using namespace uneye;
using namespace uneye::safety;
using gaze::Sample;

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); ++g_fail; } } while (0)
static void section(const char* s) { std::printf("%s\n", s); }
static bool has(const GuardOutput& o, Reason r) { return (o.reasons >> static_cast<unsigned>(r)) & 1u; }

// ---------------------------------------------------------------- deterministic signals (fixed seed, no <random>)
struct Rng { uint64_t s; explicit Rng(uint64_t x) : s(x * 2862933555777941757ULL + 3037000493ULL) {}
    double u() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<double>(s >> 11) / 9007199254740992.0; }
    double n() { double a = u(), b = u(); return std::sqrt(-2 * std::log(a + 1e-12)) * std::cos(6.283185307179586 * b); } };

// fixation noise 0.01 deg + main-sequence saccades every 250-850 ms (same generator as test_gaze_engine.cpp, no blinks)
static std::vector<Sample> synth(int n, uint64_t seed) {
    Rng r(seed); std::vector<Sample> v(static_cast<size_t>(n));
    double x = 0, y = 0; int next = 300, sac_left = 0, sac_len = 0; double ax = 0, ay = 0;
    for (int i = 0; i < n; ++i) {
        if (i == next) { const double amp = 0.2 + 7.0 * std::pow(r.u(), 2), th = 6.283185307 * r.u(); sac_len = static_cast<int>(2.2 * amp + 21); sac_left = sac_len; ax = amp * std::cos(th); ay = amp * std::sin(th); next += 250 + static_cast<int>(600 * r.u()); }
        if (sac_left > 0) { const double u0 = 1.0 - static_cast<double>(sac_left) / sac_len, u1 = u0 + 1.0 / sac_len; const double a = 0.5 * (1 - std::cos(3.141592653589793 * u1)) - 0.5 * (1 - std::cos(3.141592653589793 * u0)); x += ax * a; y += ay * a; --sac_left; }
        Sample s; s.t_us = i * 1000.0; s.x_deg = x + 0.01 * r.n(); s.y_deg = y + 0.01 * r.n();
        v[static_cast<size_t>(i)] = s;
    }
    return v;
}

static gaze::Output engine_says(gaze::State st, float p = 0.0f, gaze::Health h = gaze::Health::Ok) {
    gaze::Output o; o.state = st; o.p_saccade = p; o.source = gaze::Source::Network; o.health = h; return o;
}
static const gaze::Output kFix = engine_says(gaze::State::Fixation, 0.02f);

// runs the guard over a signal with a fixed engine output; returns the outputs
static std::vector<GuardOutput> run(SafetyGuard& g, const std::vector<Sample>& v, const gaze::Output& eo, double compute_us = 50.0) {
    std::vector<GuardOutput> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) g.step(v[i], eo, compute_us, out[i]);
    return out;
}

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : ".";
    const std::vector<Sample> clean = synth(20000, 7);

    section("[1] clean signal, engine says fixation: stays NORMAL, no reason other than None");
    {
        SafetyGuard g; auto o = run(g, clean, kFix);
        int not_normal = 0, reasons = 0; for (auto& x : o) { not_normal += x.state != GuardState::Normal; reasons += x.reasons != 0; }
        CHECK(not_normal == 0, "clean input must stay NORMAL");
        CHECK(reasons == 0, "clean input must give no reason");
    }

    section("[2] (a) NaN / out of range / spike / backwards time: that sample is Invalid, the state stays NORMAL");
    {
        SafetyGuard g; std::vector<Sample> v(clean.begin(), clean.begin() + 3000);
        v[1000].x_deg = std::nan(""); v[1500].y_deg = 1e9; v[2000].x_deg += 5.0; v[2500].t_us = v[2499].t_us - 1.0;
        auto o = run(g, v, kFix);
        CHECK(o[1000].label == Label::Invalid && has(o[1000], Reason::NonFinite), "NaN -> Invalid + NonFinite");
        CHECK(o[1500].label == Label::Invalid && has(o[1500], Reason::OutOfRange), "1e9 -> Invalid + OutOfRange");
        CHECK(o[2000].label == Label::Invalid && has(o[2000], Reason::Spike), "5 deg in 1 ms -> Invalid + Spike");
        CHECK(o[2500].label == Label::Invalid && has(o[2500], Reason::TimeBackwards), "backwards time -> Invalid + TimeBackwards");
        bool all_normal = true; for (auto& x : o) all_normal &= x.state == GuardState::Normal;
        CHECK(all_normal, "single bad samples must not change the state");
    }

    section("[3] (a) saturation at 35 deg -> SAFE (Saturated)");
    {
        SafetyGuard g; std::vector<Sample> v(clean.begin(), clean.begin() + 2000);
        for (int i = 1000; i < 1100; ++i) v[static_cast<size_t>(i)].x_deg = 35.0;
        auto o = run(g, v, kFix);
        CHECK(o[1010].state == GuardState::Safe && has(o[1010], Reason::Saturated), "held at 35 deg -> SAFE");
        CHECK(o[1010].label == Label::Invalid, "SAFE -> Invalid");
    }

    section("[4] (a) frozen signal (both axes identical) -> SAFE, then SAFE -> DEGRADED -> NORMAL with hysteresis, never SAFE -> NORMAL");
    {
        GuardConfig c; SafetyGuard g(c); std::vector<Sample> v(clean.begin(), clean.begin() + 4000);
        for (int i = 1000; i < 1200; ++i) { v[static_cast<size_t>(i)].x_deg = v[999].x_deg; v[static_cast<size_t>(i)].y_deg = v[999].y_deg; }
        auto o = run(g, v, kFix);
        CHECK(o[1000 + c.flat_samples].state == GuardState::Safe && has(o[1000 + c.flat_samples], Reason::Flat), "frozen 50 samples -> SAFE (Flat)");
        CHECK(o[1000 + c.flat_samples - 2].state == GuardState::Normal, "not before flat_samples");
        int first_degraded = -1, first_normal = -1; bool jumped = false;
        for (int i = 1200; i < 4000; ++i) {
            if (first_degraded < 0 && o[static_cast<size_t>(i)].state == GuardState::Degraded) first_degraded = i;
            if (first_normal < 0 && o[static_cast<size_t>(i)].state == GuardState::Normal) first_normal = i;
            if (o[static_cast<size_t>(i)].state == GuardState::Normal && o[static_cast<size_t>(i - 1)].state == GuardState::Safe) jumped = true;
        }
        CHECK(!jumped, "never SAFE -> NORMAL in one step");
        CHECK(first_degraded == 1200 + c.recover_samples - 1, "SAFE -> DEGRADED after exactly recover_samples clean samples");
        CHECK(first_normal == first_degraded + c.recover_samples, "DEGRADED -> NORMAL after another recover_samples");
        CHECK(has(o[1300], Reason::Hysteresis), "while recovering the reason is Hysteresis");
        std::printf("  SAFE until %d, DEGRADED from %d, NORMAL from %d\n", 1200, first_degraded, first_normal);
    }

    section("[5] (a) wrong sampling rate (500 Hz timestamps on a 1 kHz guard) -> SAFE (RateMismatch)");
    {
        SafetyGuard g; std::vector<Sample> v(clean.begin(), clean.begin() + 1000);
        for (size_t i = 0; i < v.size(); ++i) v[i].t_us = static_cast<double>(i) * 2000.0;
        auto o = run(g, v, kFix);
        CHECK(o.back().state == GuardState::Safe && has(o.back(), Reason::RateMismatch), "2 ms steps -> SAFE");
    }

    section("[6] (a) long dropout -> SAFE (LongInvalid); a blink of 100 ms does not");
    {
        SafetyGuard g; std::vector<Sample> v(clean.begin(), clean.begin() + 3000);
        for (int i = 500; i < 600; ++i) v[static_cast<size_t>(i)].x_deg = std::nan("");
        for (int i = 1000; i < 1700; ++i) v[static_cast<size_t>(i)].x_deg = std::nan("");
        auto o = run(g, v, kFix);
        bool blink_safe = false; for (int i = 500; i < 700; ++i) blink_safe |= o[static_cast<size_t>(i)].state == GuardState::Safe;
        CHECK(!blink_safe, "a 100 ms blink must not trigger SAFE");
        CHECK(o[1600].state == GuardState::Safe && has(o[1600], Reason::LongInvalid), "600 ms invalid -> SAFE");
    }

    section("[7] engine health: Failed -> SAFE, Degraded -> DEGRADED");
    {
        SafetyGuard g1; auto o1 = run(g1, std::vector<Sample>(clean.begin(), clean.begin() + 100), engine_says(gaze::State::Fixation, 0, gaze::Health::Failed));
        CHECK(o1.back().state == GuardState::Safe && o1.back().reason == Reason::EngineFailed, "Failed -> SAFE");
        SafetyGuard g2; auto o2 = run(g2, std::vector<Sample>(clean.begin(), clean.begin() + 100), engine_says(gaze::State::Fixation, 0, gaze::Health::Degraded));
        CHECK(o2.back().state == GuardState::Degraded && o2.back().reason == Reason::NetworkUntrusted, "Degraded -> DEGRADED");
    }

    section("[8] (c) network output always near 0.5 -> DEGRADED (LowConfidence)");
    {
        SafetyGuard g; auto o = run(g, std::vector<Sample>(clean.begin(), clean.begin() + 2000), engine_says(gaze::State::Fixation, 0.45f));
        int first = -1; for (size_t i = 0; i < o.size(); ++i) if (first < 0 && has(o[i], Reason::LowConfidence)) first = static_cast<int>(i);
        CHECK(first > 0 && o.back().state == GuardState::Degraded, "unsure network -> DEGRADED");
        std::printf("  LowConfidence from sample %d (moving average over ~1 s)\n", first);
    }

    section("[9] (c) input noise far above the training range -> DEGRADED (OutOfDistribution)");
    {
        SafetyGuard g; std::vector<Sample> v(clean.begin(), clean.begin() + 2000); Rng r(3);
        for (auto& s : v) { s.x_deg += 0.1 * r.n(); s.y_deg += 0.1 * r.n(); }       // ~ 47 deg/s noise of the 3-sample velocity (limit 23.1)
        auto o = run(g, v, kFix);
        CHECK(o.back().state == GuardState::Degraded && has(o.back(), Reason::OutOfDistribution), "noisy input -> DEGRADED");
    }

    section("[10] (d) deadline: a late sample is Invalid; 1 miss -> DEGRADED, 10 misses in 1 s -> SAFE");
    {
        SafetyGuard g; std::vector<Sample> v(clean.begin(), clean.begin() + 3000); std::vector<GuardOutput> o(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            const bool late = i == 500 || (i >= 2000 && i < 2010);
            g.step(v[i], kFix, late ? 1500.0 : 50.0, o[i]);
        }
        CHECK(o[500].label == Label::Invalid && has(o[500], Reason::DeadlineMiss), "late sample -> Invalid");
        CHECK(o[501].state == GuardState::Degraded, "one miss -> DEGRADED");
        CHECK(o[2009].state == GuardState::Safe, "10 misses -> SAFE");
        CHECK(o[1499].state != GuardState::Normal, "the miss stays in the 1000-sample window");
    }

    section("[11] (b) event physiology: too long, too large, too short, refractory, too fast (lower limit to reach it)");
    {
        // engine labels are made by hand; the positions move so that amplitude and speed are known
        auto drive = [&](SafetyGuard& g, int on, int len, double step_deg, int n) {
            std::vector<GuardOutput> o(static_cast<size_t>(n)); double x = 0; Rng r(11);
            for (int i = 0; i < n; ++i) {
                const bool sac = i >= on && i < on + len;
                if (sac) x += step_deg;
                Sample s; s.t_us = i * 1000.0; s.x_deg = x + 0.005 * r.n(); s.y_deg = 0.005 * r.n();
                g.step(s, sac ? engine_says(gaze::State::Saccade, 0.95f) : kFix, 50.0, o[static_cast<size_t>(i)]);
            }
            return o;
        };
        auto ended = [](const std::vector<GuardOutput>& o, Reason* r, bool* acc) { for (auto& x : o) if (x.event_ended) { *r = x.event.reason; *acc = x.event.accepted; return true; } return false; };
        Reason r; bool acc;
        { SafetyGuard g; auto o = drive(g, 500, 30, 0.1, 1000); CHECK(ended(o, &r, &acc) && acc, "30 ms, 3 deg, 100 deg/s: emitted"); }
        { SafetyGuard g; auto o = drive(g, 500, 200, 0.03, 1000); CHECK(ended(o, &r, &acc) && !acc && r == Reason::EventTooLong, "200 ms -> dropped (EventTooLong)"); }
        { SafetyGuard g; auto o = drive(g, 500, 100, 0.5, 1000); CHECK(ended(o, &r, &acc) && !acc && r == Reason::EventTooLarge, "50 deg -> dropped (EventTooLarge)"); }
        { SafetyGuard g; auto o = drive(g, 500, 2, 0.05, 1000); CHECK(ended(o, &r, &acc) && !acc && r == Reason::EventTooShort, "2 ms -> dropped (EventTooShort)"); }
        { GuardConfig c; c.max_peak_deg_s = 300; SafetyGuard g(c); auto o = drive(g, 500, 30, 0.5, 1000); CHECK(ended(o, &r, &acc) && !acc && r == Reason::EventTooFast, "500 deg/s with limit 300 -> dropped (EventTooFast)"); }
        {   // two events 5 ms apart: the second violates the refractory time
            SafetyGuard g; std::vector<GuardOutput> o(1000); double x = 0; int ends = 0; Reason second = Reason::None;
            for (int i = 0; i < 1000; ++i) {
                const bool sac = (i >= 500 && i < 530) || (i >= 535 && i < 565);
                if (sac) x += 0.1;
                Sample s; s.t_us = i * 1000.0; s.x_deg = x; s.y_deg = 0.001 * (i % 7);
                g.step(s, sac ? engine_says(gaze::State::Saccade, 0.95f) : kFix, 50.0, o[static_cast<size_t>(i)]);
                if (o[static_cast<size_t>(i)].event_ended && ++ends == 2) second = o[static_cast<size_t>(i)].event.reason;
            }
            CHECK(second == Reason::EventRefractory, "second event 5 ms after the first -> dropped (EventRefractory)");
        }
    }

    section("[12] network stuck at 'saccade' (p = 1) on a fixation: no event is ever emitted, the guard goes DEGRADED, then labels fixation");
    {
        SafetyGuard g; std::vector<Sample> v(clean.begin(), clean.begin() + 250);   // a fixation (the first saccade starts at 300)
        v.resize(3000); Rng r(5); for (size_t i = 250; i < v.size(); ++i) { v[i].t_us = static_cast<double>(i) * 1000.0; v[i].x_deg = 0.01 * r.n(); v[i].y_deg = 0.01 * r.n(); }
        auto o = run(g, v, engine_says(gaze::State::Saccade, 1.0f));
        int emitted = 0, degraded_at = -1; for (size_t i = 0; i < o.size(); ++i) { emitted += o[i].event_ended && o[i].event.accepted; if (degraded_at < 0 && o[i].state == GuardState::Degraded) degraded_at = static_cast<int>(i); }
        CHECK(emitted == 0, "no event from a stuck network");
        CHECK(degraded_at > 0, "stuck network -> DEGRADED (NoMovement / RepeatedViolations)");
        int sacc_after = 0; for (size_t i = static_cast<size_t>(degraded_at) + 10; i < o.size(); ++i) sacc_after += o[i].label == Label::Saccade;
        CHECK(sacc_after == 0, "in DEGRADED the guard's own detector labels the fixation as fixation");
        std::printf("  DEGRADED from sample %d\n", degraded_at);
    }

    section("[13] SAFE ignores the network: frozen signal + network p = 1 -> all Invalid, no event");
    {
        SafetyGuard g; std::vector<Sample> v(clean.begin(), clean.begin() + 3000);
        for (size_t i = 1000; i < v.size(); ++i) { v[i].x_deg = 2.0; v[i].y_deg = 1.0; }
        auto o = run(g, v, engine_says(gaze::State::Saccade, 1.0f));
        int bad = 0; for (size_t i = 1100; i < o.size(); ++i) bad += o[i].label != Label::Invalid || (o[i].event_ended && o[i].event.accepted);
        CHECK(o[1100].state == GuardState::Safe, "frozen -> SAFE");
        CHECK(bad == 0, "in SAFE: only Invalid, no event, whatever the network says");
    }

    section("[14] determinism: the same input twice gives bit-identical output");
    {
        std::vector<Sample> v(clean.begin(), clean.begin() + 5000); v[1000].x_deg = std::nan(""); for (int i = 2000; i < 2100; ++i) v[static_cast<size_t>(i)].x_deg = 35;
        SafetyGuard a, b; auto oa = run(a, v, kFix), ob = run(b, v, kFix);
        bool same = true;
        for (size_t i = 0; i < v.size(); ++i)
            same &= oa[i].state == ob[i].state && oa[i].label == ob[i].label && oa[i].reasons == ob[i].reasons && oa[i].reason == ob[i].reason &&
                    oa[i].event_ended == ob[i].event_ended && std::memcmp(&oa[i].event.amplitude_deg, &ob[i].event.amplitude_deg, sizeof(float)) == 0;
        a.reset(); auto oc = run(a, v, kFix);
        for (size_t i = 0; i < v.size(); ++i) same &= oa[i].state == oc[i].state && oa[i].reasons == oc[i].reasons;
        CHECK(same, "identical outputs (also after reset())");
    }

    section("[15] behind the real engine (causal + 10 ms TCN + U'n'Eye window), clean synthetic gaze: stays NORMAL, events emitted, none dropped");
    {
        gaze::Models m;
        m.fast = make_causal_engine(root + "/models/causal.bin");
        m.refine = make_causal_engine(root + "/models/tcn_l10.bin");
        m.window = make_native_engine(root + "/models/combined.bin", 200);
        gaze::GazeEngine eng(gaze::Config{}, std::move(m));
        SafetyGuard g; gaze::Output eo; GuardOutput go;
        int not_normal = 0, emitted = 0, dropped = 0, violations = 0; std::array<int, static_cast<int>(Reason::Count)> drops{};
        for (const auto& s : clean) {
            eng.push(s, eo); g.step(s, eo, 50.0, go);
            not_normal += go.state != GuardState::Normal;
            if (go.event_ended) {
                if (go.event.accepted) ++emitted;
                else { ++dropped; ++drops[static_cast<size_t>(go.event.reason)]; violations += go.event.reason != Reason::EventTooShort && go.event.reason != Reason::EventAborted; }
            }
        }
        std::printf("  20 s: %d samples not NORMAL, %d events emitted, %d dropped", not_normal, emitted, dropped);
        for (int k = 0; k < static_cast<int>(Reason::Count); ++k) if (drops[static_cast<size_t>(k)]) std::printf(" [%s %d]", reason_name(static_cast<Reason>(k)), drops[static_cast<size_t>(k)]);
        std::printf("\n");
        CHECK(not_normal == 0, "clean input behind the real engine stays NORMAL");
        CHECK(emitted >= 30, "the ~36 saccades of the synthetic signal are emitted (one every 250-850 ms)");
        CHECK(violations <= emitted / 10, "few physiology drops on clean input (1-2 sample blips, EventTooShort, are filtered, not counted)");
    }

    std::printf(g_fail ? "FAILED: %d check(s)\n" : "all safety-guard checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}

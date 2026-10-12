// T04: batch parity with dsp:: on the golden traces, streaming against batch (golden + synthetic traces), publication delay, processing time, invalid samples / blinks, option validation.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include "eyelive/hmm_detector.hpp"
#include "eyelive/trace.hpp"
using namespace eyelive;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static VectorTrace make_trace(const std::vector<double>& x, const std::vector<double>& y, const std::vector<uint8_t>& valid, double fs) {
    SessionInfo s; s.rate_hz = fs; VectorTrace tr(s);
    for (size_t i = 0; i < x.size(); ++i) { Sample sm; sm.t = double(i) / fs; sm.x = float(x[i]); sm.y = float(y[i]); sm.flags = valid[i] ? kFlagValid : kFlagBlink; tr.push(sm); }
    return tr;
}
static dsp::Events read_events(std::ifstream& f) { int n; f >> n; dsp::Events e(n); for (auto& p : e) f >> p.first >> p.second; return e; }

struct Golden { std::string name; double fs; std::vector<double> x, y; std::vector<uint8_t> valid; dsp::Events hmm, first, last; };
static std::vector<Golden> load_golden(const std::string& root) {
    std::vector<Golden> g; std::ifstream f(root + "/app/dsp/tests/golden_hmm_accel.txt"); int n; f >> n;
    for (int c = 0; c < n; ++c) { Golden t; int T; f >> t.name >> t.fs >> T; t.x.resize(T); t.y.resize(T); t.valid.resize(T);
        for (int i = 0; i < T; ++i) { std::string a, b; f >> a >> b; t.x[i] = std::strtod(a.c_str(), nullptr); t.y[i] = std::strtod(b.c_str(), nullptr); t.valid[i] = std::isfinite(t.x[i]) && std::isfinite(t.y[i]); }
        t.hmm = read_events(f); t.first = read_events(f); t.last = read_events(f); g.push_back(std::move(t)); }
    return g;
}
static dsp::HmmParams params_for(double fs) { dsp::HmmParams p; if (std::fabs(fs - 1000) < 1) p = {fs, 6, 0.416573763, 1.32039225, 0.201105669, 0.766239583, 0.00554218027, 0.949469511}; else p = {fs, 3, 0.406975567, 1.3010565, 0.20460619, 0.573560297, 0.00969478964, 0.928820044}; return p; }

struct Syn { std::vector<double> x, y; std::vector<uint8_t> valid; };
static Syn synth(double seconds, double fs, uint64_t seed) {
    std::mt19937_64 rng(seed); std::normal_distribution<double> nd(0, 1); std::uniform_real_distribution<double> ud(0, 1);
    const size_t n = size_t(seconds * fs); Syn s; s.x.assign(n, 0); s.y.assign(n, 0); s.valid.assign(n, 1); double px = 0, py = 0, dx = 0, dy = 0; size_t i = 0; const double noise = 0.02 + 0.03 * ud(rng);
    while (i < n) {
        const size_t fix = size_t((0.3 + 1.2 * ud(rng)) * fs);
        for (size_t k = 0; k < fix && i < n; ++k, ++i) { dx = dx * 0.999 + 2e-5 * nd(rng); dy = dy * 0.999 + 2e-5 * nd(rng); px += dx; py += dy; s.x[i] = px + noise * nd(rng); s.y[i] = py + noise * nd(rng); }
        const double amp = 1.0 + 11.0 * ud(rng), th = 6.2831853 * ud(rng), dur = (21 + 2.2 * amp) * fs / 1000.0; const size_t len = size_t(dur);
        for (size_t k = 0; k < len && i < n; ++k, ++i) { const double u = double(k + 1) / len, sm = u * u * u * (10 - 15 * u + 6 * u * u); s.x[i] = px + amp * std::cos(th) * sm + noise * nd(rng); s.y[i] = py + amp * std::sin(th) * sm + noise * nd(rng); }
        px += amp * std::cos(th); py += amp * std::sin(th); if (std::fabs(px) > 12 || std::fabs(py) > 12) { px *= 0.3; py *= 0.3; }
    }
    return s;
}

struct StreamResult { std::vector<Event> ev; std::vector<int64_t> pub_at; double us_per_sample = 0; };
static StreamResult stream(HmmDetector& d, const VectorTrace& tr) {
    StreamResult r; d.reset(tr.info()); std::vector<Event> fin, prov; std::vector<Event> none; const auto t0 = std::chrono::steady_clock::now();
    for (int64_t n = 1; n <= tr.size(); ++n) { fin.clear(); d.advance(tr, n, fin, prov); for (auto& e : fin) { r.ev.push_back(e); r.pub_at.push_back(n); } }
    r.us_per_sample = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / double(tr.size()); return r;
}
// fraction of reference events (ending before `until`) found in the stream with onset / offset within 2 samples; duplicates = stream events overlapping the previous one
struct Cmp { int ref = 0, matched = 0, extra = 0, dup = 0; };
static Cmp compare(const std::vector<Event>& ref, const std::vector<Event>& got, int64_t until, int off_tol = 2) {
    Cmp c; for (const auto& r : ref) { if (r.offset >= until) continue; ++c.ref; for (const auto& g : got) if (std::abs(g.onset - r.onset) <= 2 && std::abs(g.offset - r.offset) <= off_tol) { ++c.matched; break; } }
    for (const auto& g : got) { if (g.offset >= until) continue; bool f = false; for (const auto& r : ref) if (std::abs(g.onset - r.onset) <= 2 && std::abs(g.offset - r.offset) <= off_tol) { f = true; break; } if (!f) ++c.extra; }
    for (size_t i = 1; i < got.size(); ++i) if (got[i].onset <= got[i - 1].offset) ++c.dup; return c;
}

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : "."; auto gold = load_golden(root);
    CHECK(gold.size() == 5, "golden file: %zu traces", gold.size());
    // 1. detect_all == dsp chain, exactly
    for (auto& g : gold) {
        auto tr = make_trace(g.x, g.y, g.valid, g.fs); dsp::Kin k; dsp::score_trace(g.x, g.y, g.valid, g.fs, k); const dsp::HmmParams hp = params_for(g.fs); const dsp::Events ev = dsp::runs_of(dsp::hmm_detect(k, hp));
        for (int rule = 0; rule < 3; ++rule) {
            HmmDetector d; d.reset(tr.info()); const char* names[3] = {"last", "first", "none"}; CHECK(d.set_option("offset_rule", names[rule]), "offset_rule %s", names[rule]);
            dsp::Refine rf; rf.offset_first = rule == 1; dsp::Events want = rule == 2 ? ev : dsp::runs_of(dsp::refine_events(ev, k.fx, k.fy, g.valid, g.fs, rf));
            std::vector<Event> got; d.detect_all(tr, got); bool same = got.size() == want.size(); for (size_t i = 0; same && i < want.size(); ++i) same = got[i].onset == want[i].first && got[i].offset == want[i].second;
            CHECK(same, "%s offset_rule=%s: detect_all gives %zu events, dsp chain %zu", g.name.c_str(), names[rule], got.size(), want.size());
        }
        // and against the Python reference (as the dsp parity test does: 'last' within 1 sample)
        HmmDetector d; d.reset(tr.info()); std::vector<Event> got; d.detect_all(tr, got); CHECK(got.size() == g.last.size(), "%s: %zu events vs Python %zu", g.name.c_str(), got.size(), g.last.size());
    }
    std::printf("batch: detect_all identical to dsp::hmm_detect + dsp::refine_events on %zu golden traces (3 offset rules each)\n", gold.size());
    // 2. streaming against batch on the golden traces and 20 synthetic traces (1 kHz) + 3 at 500 Hz. Two comparisons: the HMM events alone (offset_rule none: must agree to 2 samples) and the refined events (rule "last":
    //    the offset is the last sample above 3 x noise inside a 25 ms margin, a noise-tail criterion that moves by several samples with a 1 % change of the noise estimate, so its tolerance is 15 samples)
    std::vector<double> delays;
    for (int pass = 0; pass < 2; ++pass) {
        const char* rule = pass == 0 ? "none" : "last"; const int tol = pass == 0 ? 2 : 15; Cmp tot; double worst_us = 0;
        auto run_one = [&](const VectorTrace& tr, const char* name) {
            HmmDetector d; d.set_option("offset_rule", rule); std::vector<Event> ref; d.reset(tr.info()); d.detect_all(tr, ref); HmmDetector s; s.set_option("offset_rule", rule); auto r = stream(s, tr); const int64_t until = tr.size() - int64_t(0.3 * tr.info().rate_hz);
            Cmp c = compare(ref, r.ev, until, tol); tot.ref += c.ref; tot.matched += c.matched; tot.extra += c.extra; tot.dup += c.dup; worst_us = std::max(worst_us, r.us_per_sample);
            if (pass == 1) for (size_t i = 0; i < r.ev.size(); ++i) delays.push_back(double(r.pub_at[i] - 1 - r.ev[i].offset) * 1000.0 / tr.info().rate_hz);
            if (c.matched < c.ref || c.dup) std::printf("  [%s] %s: reference %d, matched %d, extra %d, duplicates %d\n", rule, name, c.ref, c.matched, c.extra, c.dup);
        };
        for (auto& g : gold) run_one(make_trace(g.x, g.y, g.valid, g.fs), g.name.c_str());
        for (int i = 0; i < 20; ++i) { auto s = synth(20.0, 1000.0, 100 + i); run_one(make_trace(s.x, s.y, s.valid, 1000.0), ("synthetic " + std::to_string(i)).c_str()); }
        for (int i = 0; i < 3; ++i) { auto s = synth(20.0, 500.0, 300 + i); run_one(make_trace(s.x, s.y, s.valid, 500.0), ("synthetic 500 Hz " + std::to_string(i)).c_str()); }
        const double frac = tot.ref ? double(tot.matched) / tot.ref : 1.0;
        std::printf("streaming vs batch, offset_rule=%s: %d of %d reference events matched (%.1f %%; onset within 2 samples, offset within %d), %d extra, %d duplicated; worst processing time %.1f us per sample\n", rule, tot.matched, tot.ref, 100 * frac, tol, tot.extra, tot.dup, worst_us);
        CHECK(frac >= 0.99, "[%s] matched %.3f < 0.99", rule, frac); CHECK(tot.dup == 0, "[%s] %d duplicated events", rule, tot.dup); CHECK(worst_us < 100.0, "processing time %.1f us per sample", worst_us);
        if (worst_us >= 20.0) std::printf("  note: above the 20 us target\n");
    }
    std::sort(delays.begin(), delays.end()); HmmDetector probe; SessionInfo si; si.rate_hz = 1000; probe.reset(si); const double declared = probe.info().decision_delay_ms;
    const double p50 = delays.empty() ? 0 : delays[delays.size() / 2], p95 = delays.empty() ? 0 : delays[size_t(0.95 * (delays.size() - 1))], mx = delays.empty() ? 0 : delays.back();
    std::printf("publication delay after the (refined) offset, events of all traces: median %.0f ms, p95 %.0f ms, max %.0f ms (declared decision_delay_ms %.0f)\n", p50, p95, mx, declared);
    CHECK(p50 <= declared + 5.0, "median delay %.0f ms above the declared %.0f ms", p50, declared); CHECK(mx <= 250.0, "worst delay %.0f ms", mx);
    // 3. invalid samples and a blink inside an event
    {
        auto s = synth(6.0, 1000.0, 7); std::vector<uint8_t> v = s.valid; for (int i = 2000; i < 2003; ++i) { v[i] = 0; s.x[i] = std::nan(""); s.y[i] = std::nan(""); }
        auto tr = make_trace(s.x, s.y, v, 1000.0); HmmDetector d; auto r = stream(d, tr); std::vector<Event> b; HmmDetector d2; d2.detect_all(tr, b);
        for (auto& e : r.ev) CHECK(!(e.onset >= 2000 && e.offset <= 2002), "an event was created inside the invalid samples"); std::printf("blink test: %zu streaming / %zu batch events, no crash\n", r.ev.size(), b.size());
        // a fixation with a 3-sample blink and nothing else: no event
        Syn q; q.x.assign(4000, 0.0); q.y.assign(4000, 0.0); q.valid.assign(4000, 1); std::mt19937_64 rng(1); std::normal_distribution<double> nd(0, 0.02); for (int i = 0; i < 4000; ++i) { q.x[i] = nd(rng); q.y[i] = nd(rng); } for (int i = 1500; i < 1503; ++i) { q.valid[i] = 0; q.x[i] = std::nan(""); q.y[i] = std::nan(""); }
        auto tq = make_trace(q.x, q.y, q.valid, 1000.0); HmmDetector d3; auto r3 = stream(d3, tq); std::vector<Event> b3; HmmDetector d4; d4.detect_all(tq, b3); CHECK(r3.ev.empty() && b3.empty(), "fixation + blink produced %zu streaming / %zu batch events", r3.ev.size(), b3.size());
    }
    // 4. options
    {
        HmmDetector d; CHECK(d.set_option("offset_rule", "first") && d.set_option("offset_rule", "last") && d.set_option("offset_rule", "none"), "valid offset_rule rejected");
        CHECK(!d.set_option("offset_rule", "middle"), "bad value accepted"); CHECK(!d.set_option("unknown_key", "1"), "unknown key accepted"); CHECK(d.set_option("refine_win_ms", "20") && !d.set_option("refine_win_ms", "abc") && !d.set_option("refine_win_ms", "-3"), "refine_win_ms validation");
        CHECK(d.set_option("margin_ms", "30") && !d.set_option("margin_ms", "1e9"), "margin_ms validation");
    }
    // 5. unsupported rate: a clear error, no events, no crash
    { SessionInfo s2; s2.rate_hz = 2000; HmmDetector d; d.reset(s2); CHECK(!d.error().empty(), "no error for 2000 Hz"); auto s = synth(2.0, 2000.0, 5); auto tr = make_trace(s.x, s.y, s.valid, 2000.0); auto r = stream(d, tr); CHECK(r.ev.empty(), "events at an unsupported rate"); std::printf("2000 Hz: '%s'\n", d.error().c_str()); }
    std::printf(fails ? "FAILED\n" : "all hmm_accel tests passed\n"); return fails ? 1 : 0;
}

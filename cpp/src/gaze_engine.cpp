// Deterministic streaming gaze-state engine. See include/uneye/gaze_engine.hpp and cpp/GAZE_ENGINE.md.
// Rules followed in everything that runs per sample: fixed-size arrays only (no heap), bounded loops, no clock / randomness / I/O, no recursion.
#include "uneye/gaze_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "uneye/gaze_tables.hpp"

namespace uneye {
namespace gaze {
namespace {

constexpr int kRing = 512, kMask = kRing - 1;          // bins kept (must be a power of two, > window + refine delay + noise window use)
constexpr int kNoiseWin = 200;                          // samples used for the robust noise estimate
constexpr int kStayWin = 150;                           // bins used for the fixation scatter (forecast uncertainty)
constexpr int kNoiseEvery = 8;                          // recompute the noise estimates every 8 samples
constexpr double kLn2 = 0.6931471805599453;
// Calibration of the forecast uncertainty (measured on 300 trials of set A so that the 80 % interval covers 80 %, see GAZE_ENGINE.md):
constexpr double kStayScaleFactor = 0.8;               // 'stay' intervals from the fixation scatter are slightly too wide
constexpr double kBallisticInflationEarly = 4.5;       // first 10 ms after the onset: the amplitude is hardly known yet
constexpr double kBallisticInflationLate = 1.8;
constexpr double kPi = 3.14159265358979323846;

struct Bin {
    int64_t index = -1;
    double x = 0, y = 0;                  // last valid position (held over invalid samples)
    float dx = 0, dy = 0;                 // per-sample displacement fed to the networks (deg/sample; 0 for invalid samples)
    float speed = 0;                      // deg/s, mean velocity of the last 3 samples
    float p = 0;                          // final saccade probability / confidence of the label
    uint32_t flags = 0;
    State state = State::Unknown;
    Source source = Source::None;
    uint8_t revisions = 0;
    uint8_t stage = 0;                    // 0 fast, 1 refined by the 10 ms network, 2 finalized by the window network
    bool valid = false, above = false, weak = false, strong = false, heur = false, final_ = false;
};

float median_of(float* a, int n) {                       // partially sorts the scratch array; n >= 1
    float* mid = a + n / 2;
    std::nth_element(a, mid, a + n);
    return *mid;
}

bool good_prob(const float* p) {                         // two-class softmax output
    return std::isfinite(p[0]) && std::isfinite(p[1]) && p[0] >= -1e-4f && p[0] <= 1.0001f && p[1] >= -1e-4f && p[1] <= 1.0001f &&
           std::fabs(p[0] + p[1] - 1.0f) < 0.02f;
}

}  // namespace

struct GazeEngine::Impl {
    Config cfg;
    Models models;
    int refine_delay;
    bool rate_ok;
    int wlen = 0, wage = 0, whop = 0;                       // window network: length, age at which bins are finalized, run every whop samples
    std::array<float, 2 * 256> win_in{};
    std::array<float, 5 * 256> win_out{};

    std::array<Bin, kRing> ring;
    int64_t n = 0;                                          // next bin index
    // input guard
    bool have_valid = false;
    double last_x = 0, last_y = 0;
    int invalid_run = 0;
    int64_t first_invalid = -1;
    int64_t blink_until = -1;                               // bins up to here are blink margin
    bool have_t = false;
    double last_t = 0;
    // noise model (robust, from the data themselves)
    std::array<float, kNoiseWin> nvx{}, nvy{};
    int ncount = 0, npos = 0, nsince = 0;
    double sigx = 20.0, sigy = 20.0;                        // deg/s, robust standard deviation of the 3-sample velocity
    // heuristic runs
    int run_above = 0;
    // networks
    float p_fast = 0, p_ref = 0;
    bool fast_ok = false, ref_ok = false;
    int nn_good_run = 1 << 20;                              // consecutive valid network outputs (starts "trusted")
    bool nn_trust = true, nn_stuck = false;
    float last_pf_bits = -1.0f; int same_run = 0;
    double veto_ema = 0, miss_ema = 0, flip_ema = 0;
    bool last_nn_label = false;
    int clear_len = 0; double clear_sum = 0; int trust_hold = 0;
    // forecast uncertainty of 'stay in place' (Laplace scale per horizon and axis)
    double stay_scale[2][2] = {{0.05, 0.05}, {0.05, 0.05}};
    // per push bookkeeping
    int64_t rev_first = -1, rev_last = -1;
    double sp_now = 0;
    Stats st;
    std::array<float, kNoiseWin> scratch{};

    Impl(const Config& c, Models m) : cfg(c), models(std::move(m)), refine_delay(models.refine_delay), rate_ok(std::fabs(c.fs_hz - 1000.0) <= 100.0) {
        if (refine_delay < 1) refine_delay = 1;
        if (refine_delay > 64) refine_delay = 64;
        if (models.window) {
            wlen = models.window->window();
            if (wlen < 50 || wlen > 256 || models.window->classes() != 2) throw std::invalid_argument("window network: need 50..256 samples and 2 classes");
            whop = std::min(std::max(models.window_hop, 1), 64);
            wage = std::min(std::max(models.window_age, refine_delay + 1), wlen - 1 - whop);
            if (wage < refine_delay + 1) throw std::invalid_argument("window network: window too short for the chosen age");
            win_in.fill(0.0f);
            models.window->infer(win_in.data(), win_out.data());   // warm-up: the network sizes its work buffers here, never again
        }
        reset_state();
    }

    void reset_state() noexcept {
        for (auto& b : ring) b = Bin{};
        n = 0; have_valid = false; invalid_run = 0; first_invalid = -1; blink_until = -1; have_t = false;
        ncount = npos = nsince = 0; sigx = sigy = 20.0; run_above = 0;
        p_fast = p_ref = 0; fast_ok = ref_ok = false; nn_stuck = false; last_pf_bits = -1.0f; same_run = 0; veto_ema = 0; miss_ema = 0; flip_ema = 0; last_nn_label = false; clear_len = 0; clear_sum = 0; trust_hold = 0;
        nn_good_run = 1 << 20; nn_trust = true;
        for (auto& r : stay_scale) { r[0] = r[1] = 0.05; }
        if (models.fast) models.fast->reset();
        if (models.refine) models.refine->reset();
    }

    Bin& bin(int64_t i) noexcept { return ring[static_cast<size_t>(i) & kMask]; }
    const Bin& bin(int64_t i) const noexcept { return ring[static_cast<size_t>(i) & kMask]; }
    bool in_ring(int64_t i) const noexcept { return i >= 0 && i < n && i > n - kRing + 8; }

    void mark_revised(int64_t i) noexcept {
        if (rev_first < 0 || i < rev_first) rev_first = i;
        if (i > rev_last) rev_last = i;
    }

    // ------------------------------------------------------------------ evidence in the data (physics layer)
    bool weak_evidence(int64_t i) const noexcept {          // any 'some movement' bin in [i - hw, i + hw]
        for (int64_t k = i - cfg.gate_half_window; k <= i + cfg.gate_half_window; ++k)
            if (in_ring(k) && bin(k).weak) return true;
        return false;
    }
    bool strong_evidence(int64_t i) const noexcept {        // at least min_run 'obviously moving' bins in the window
        int c = 0;
        for (int64_t k = i - cfg.gate_half_window; k <= i + cfg.gate_half_window; ++k)
            if (in_ring(k) && bin(k).strong) ++c;
        return c >= cfg.min_run;
    }

    // ------------------------------------------------------------------ label of one bin from the network output and the physics layer
    void fuse(Bin& b, float pnn, bool nn_ok, bool trusted) noexcept {
        b.flags &= ~(flag::NnVetoed | flag::HeuristicOverride | flag::NnInvalid | flag::NnDegraded | flag::NnStuck);
        if (!b.valid || b.state == State::Blink || b.state == State::Invalid) { b.p = 0; return; }   // guard decided
        const bool use_nn = models.fast && cfg.mode != Mode::HeuristicOnly && nn_ok && (trusted || cfg.mode == Mode::NetworkOnly);
        if (!use_nn) {
            if (cfg.mode == Mode::NetworkOnly) { b.p = 0; b.source = Source::Network; b.state = State::Fixation; return; }
            if (models.fast && cfg.mode == Mode::Fused) b.flags |= nn_ok ? (nn_stuck ? flag::NnStuck : flag::NnDegraded) : flag::NnInvalid;
            b.p = b.heur ? 1.0f : 0.0f;
            b.source = Source::Heuristic;
            b.state = b.heur ? State::Saccade : State::Fixation;
            return;
        }
        float p = pnn; Source src = Source::Network;
        if (cfg.mode == Mode::Fused) {
            if (!weak_evidence(b.index)) {                  // guard rail: no movement in the data -> the network cannot declare a saccade
                if (p >= cfg.p_threshold) { b.flags |= flag::NnVetoed; src = Source::Veto; ++st.vetoed; }
                p = 0;
            } else if (p < cfg.p_threshold && strong_evidence(b.index)) {   // the eye is obviously moving and the network misses it
                p = 0.9f; b.flags |= flag::HeuristicOverride; src = Source::Override; ++st.overridden;
            }
        }
        b.p = p; b.source = src;
        b.state = p >= cfg.p_threshold ? State::Saccade : State::Fixation;
    }

    // ------------------------------------------------------------------ one logical sample (a real one or a synthetic 'missing' one)
    void process(bool ok, double x, double y, uint32_t sflags) noexcept {
        const int64_t i = n;
        Bin& b = bin(i);
        b = Bin{};
        b.index = i; b.flags = sflags;
        bool valid = ok;
        if (valid && !rate_ok) { valid = false; b.flags |= flag::UnsupportedRate; }
        if (valid && have_valid) {
            const double jump = std::hypot(x - last_x, y - last_y) / static_cast<double>(invalid_run + 1);
            if (jump * cfg.fs_hz > cfg.max_speed_deg_s) { valid = false; b.flags |= flag::TooFast; }
        }
        double dx = 0, dy = 0;
        const int prev_invalid = invalid_run;
        if (valid) {
            if (have_valid && prev_invalid == 0) { dx = x - last_x; dy = y - last_y; }   // training convention: no velocity across a gap
            last_x = x; last_y = y; have_valid = true; invalid_run = 0;
            b.valid = true;
        } else {
            if ((b.flags & (flag::NonFinite | flag::OutOfRange | flag::TooFast | flag::TimeBackwards | flag::Missing | flag::UnsupportedRate)) == 0) b.flags |= flag::Missing;
            if (invalid_run == 0) first_invalid = i;
            ++invalid_run;
        }
        b.x = last_x; b.y = last_y;
        b.dx = static_cast<float>(dx); b.dy = static_cast<float>(dy);
        ++n;
        ++st.samples; if (!valid) ++st.invalid;

        const int margin = static_cast<int>(cfg.blink_margin_ms * cfg.fs_hz / 1000.0 + 0.5);
        // blink = an invalid run of at least blink_min_samples; the margin before it and after it is blink too (revision)
        if (!valid) {
            if (invalid_run >= cfg.blink_min_samples) blink_until = i + margin;      // short glitches get no blink margin
            if (invalid_run == cfg.blink_min_samples) {
                for (int64_t k = first_invalid - margin; k <= i; ++k)
                    if (in_ring(k) && k != i) { Bin& o = bin(k); if (o.state != State::Blink) { o.state = State::Blink; o.source = Source::Guard; o.p = 0; o.flags |= flag::BlinkMargin; ++o.revisions; mark_revised(k); } }
            }
            b.state = invalid_run >= cfg.blink_min_samples ? State::Blink : State::Invalid;
            b.source = Source::Guard;
        } else if (i <= blink_until) {
            b.state = State::Blink; b.source = Source::Guard; b.flags |= flag::BlinkMargin;
        }

        // ---- physics layer: 3-sample velocity, robust noise, velocity-threshold evidence
        {
            float sx = b.dx, sy = b.dy; int c = 1;
            for (int k = 1; k <= 2; ++k) if (in_ring(i - k)) { sx += bin(i - k).dx; sy += bin(i - k).dy; ++c; }
            const double vx = sx / c * cfg.fs_hz, vy = sy / c * cfg.fs_hz;
            const double sp = std::hypot(vx, vy);
            sp_now = sp;
            b.speed = static_cast<float>(sp);
            if (valid) {
                nvx[static_cast<size_t>(npos)] = static_cast<float>(vx); nvy[static_cast<size_t>(npos)] = static_cast<float>(vy);
                npos = (npos + 1) % kNoiseWin; if (ncount < kNoiseWin) ++ncount;
                if (++nsince >= kNoiseEvery && ncount >= 50) { nsince = 0; update_noise(); }
                auto ell = [&](double lam) { const double a = vx / (lam * sigx), c2 = vy / (lam * sigy); return a * a + c2 * c2; };
                b.above = ell(cfg.lambda) > 1.0 && sp > cfg.floor_deg_s;
                b.weak = ell(cfg.lambda_weak) > 1.0 && sp > cfg.floor_weak_deg_s;
                b.strong = ell(cfg.lambda_strong) > 1.0 && sp > cfg.floor_strong_deg_s;
            }
            run_above = b.above ? run_above + 1 : 0;
            if (run_above == cfg.min_run) {                   // the run is a detection: its first bins are marked retroactively
                for (int k = 0; k < run_above; ++k) if (in_ring(i - k)) bin(i - k).heur = true;
            } else if (run_above > cfg.min_run) b.heur = true;
        }

        // ---- networks (fed at every sample, zeros for invalid ones, so that their state stays aligned with time)
        float pf[2] = {0, 0}, pr[2] = {0, 0};
        fast_ok = ref_ok = false;
        if (models.fast) {
            models.fast->step(b.dx, b.dy, pf);
            fast_ok = good_prob(pf); p_fast = fast_ok ? pf[1] : 0.0f;
            if (valid) {                                          // stuck detector: bit-identical output while the data vary
                if (std::memcmp(&pf[1], &last_pf_bits, sizeof(float)) == 0) ++same_run; else same_run = 0;
                last_pf_bits = pf[1];
                if (same_run >= cfg.stuck_samples) nn_stuck = true;
                else if (same_run == 0 && nn_stuck) nn_stuck = false;
            }
        }
        if (models.refine) {
            models.refine->step(b.dx, b.dy, pr);
            ref_ok = good_prob(pr); p_ref = ref_ok ? pr[1] : 0.0f;
        }
        if (models.fast) {
            if (!fast_ok || (models.refine && !ref_ok)) { nn_good_run = 0; ++st.nn_invalid; }
            else if (nn_good_run < (1 << 20)) ++nn_good_run;
            if (valid && fast_ok) {                                // the watchdog judges the raw network output, used or not
                const bool veto_candidate = p_fast >= cfg.p_threshold && !weak_evidence(i);
                veto_ema += ((veto_candidate ? 1.0 : 0.0) - veto_ema) / 512.0;
                const bool lab = p_fast >= cfg.p_threshold;
                flip_ema += ((lab != last_nn_label ? 1.0 : 0.0) - flip_ema) / 256.0; last_nn_label = lab;
                if (b.above && sp_now > cfg.clear_speed_deg_s) { ++clear_len; clear_sum += p_fast; }
                else {
                    if (clear_len >= cfg.clear_min_samples) {      // a clear movement just ended: did the network see it?
                        const bool missed = clear_sum / clear_len < 0.3;
                        miss_ema = 0.8 * miss_ema + 0.2 * (missed ? 1.0 : 0.0);
                        ++st.clear_events; if (missed) ++st.clear_missed;
                    }
                    clear_len = 0; clear_sum = 0;
                }
            }
            const bool bad = nn_good_run < 200 || nn_stuck || veto_ema > cfg.veto_ema_degraded || miss_ema > cfg.miss_degraded || flip_ema > cfg.flip_ema_degraded;
            const bool good = nn_good_run >= 200 && !nn_stuck && veto_ema < cfg.veto_ema_recover && miss_ema < cfg.miss_recover && flip_ema < cfg.flip_ema_recover;
            if (nn_trust && bad) {
                nn_trust = false; trust_hold = cfg.trust_hold_samples;
                if (nn_good_run < 200) ++st.trust_drops_invalid; else if (nn_stuck) ++st.trust_drops_stuck; else if (flip_ema > cfg.flip_ema_degraded) ++st.trust_drops_flicker; else if (veto_ema > cfg.veto_ema_degraded) ++st.trust_drops_veto; else ++st.trust_drops_miss;
            }
            else if (!nn_trust) { if (trust_hold > 0) --trust_hold; else if (good) nn_trust = true; }
        }

        // ---- label of this bin (provisional) and revision of the bin refine_delay ago
        if (b.state != State::Blink && b.state != State::Invalid) fuse(b, p_fast, fast_ok, nn_trust);
        else b.p = 0;
        st.veto_ema = veto_ema;

        const int64_t r = i - refine_delay;
        if (in_ring(r)) {
            Bin& o = bin(r);
            const State old = o.state; const float oldp = o.p;
            if (models.refine && o.state != State::Blink && o.state != State::Invalid) {
                // the refiner's gate looks refine_delay samples further than the fast one: its own evidence window is complete now
                fuse(o, p_ref, ref_ok, nn_trust);
            } else if (o.state != State::Blink && o.state != State::Invalid && !models.fast) {
                fuse(o, 0, false, false);                         // heuristic-only engine: final label once the run is complete
            }
            o.stage = 1; o.final_ = !models.window;
            if (o.state != old || std::fabs(o.p - oldp) > 0.5f) { ++o.revisions; mark_revised(r); }
        }
        if (b.state == State::Unknown) { b.state = State::Fixation; b.source = Source::Guard; }
        if (models.window && whop > 0 && (n % whop) == 0) run_window();
    }

    // ------------------------------------------------------------------ third stage: the window network finalizes the bins that just reached `wage`
    void run_window() noexcept {
        const int64_t first_pos = n - wlen;                    // bin index of window position 0 (negative: before the recording: zeros)
        for (int j = 0; j < wlen; ++j) {
            const int64_t k = first_pos + j;
            const bool have = k >= 0 && in_ring(k);
            win_in[static_cast<size_t>(2 * j)] = have ? bin(k).dx : 0.0f;
            win_in[static_cast<size_t>(2 * j + 1)] = have ? bin(k).dy : 0.0f;
        }
        models.window->infer(win_in.data(), win_out.data());
        ++st.window_runs;
        bool ok = true;                                        // every output of the window must be a probability, or the whole run is discarded
        for (int j = 0; j < wlen && ok; ++j) {
            const float p0 = win_out[static_cast<size_t>(j)], p1 = win_out[static_cast<size_t>(wlen + j)];
            ok = std::isfinite(p0) && std::isfinite(p1) && p0 >= -1e-4f && p0 <= 1.0001f && p1 >= -1e-4f && p1 <= 1.0001f && std::fabs(p0 + p1 - 1.0f) < 0.02f;
        }
        if (!ok) { ++st.window_invalid; return; }               // keep the labels of the previous stage
        // conservative guard rail: while the watchdog distrusts the networks (health Degraded) a third network must not overrule the
        // heuristic fallback; the bins are only marked final
        const bool apply = nn_trust;
        for (int age = wage; age < wage + whop; ++age) {
            const int64_t k = n - 1 - age;
            if (!in_ring(k)) continue;
            Bin& o = bin(k);
            const int j = wlen - 1 - age;
            const State old = o.state; const float oldp = o.p;
            if (apply && o.state != State::Blink && o.state != State::Invalid && cfg.mode != Mode::HeuristicOnly && models.fast)
                fuse(o, models.window_blend * oldp + (1.0f - models.window_blend) * win_out[static_cast<size_t>(wlen + j)], true, true);
            if (apply) o.stage = 2;
            o.final_ = true;
            if (o.state != old || std::fabs(o.p - oldp) > 0.5f) { ++o.revisions; mark_revised(k); }
        }
    }

    void update_noise() noexcept {                             // Engbert & Kliegl: sd = sqrt(median(v^2) - median(v)^2), per axis
        for (int axis = 0; axis < 2; ++axis) {
            const std::array<float, kNoiseWin>& v = axis == 0 ? nvx : nvy;
            for (int k = 0; k < ncount; ++k) scratch[static_cast<size_t>(k)] = v[static_cast<size_t>(k)] * v[static_cast<size_t>(k)];
            const double m2 = median_of(scratch.data(), ncount);
            for (int k = 0; k < ncount; ++k) scratch[static_cast<size_t>(k)] = v[static_cast<size_t>(k)];
            const double m1 = median_of(scratch.data(), ncount);
            const double sd = std::sqrt(std::max(m2 - m1 * m1, 1.0));       // deg/s, floor 1 deg/s
            (axis == 0 ? sigx : sigy) = sd;
        }
        update_stay_scale();
    }

    void update_stay_scale() noexcept {                        // Laplace scale of |x(t) - x(t-h)| over recent fixation bins
        const int hs[2] = {10, 20};
        for (int hi = 0; hi < 2; ++hi) {
            const int h = static_cast<int>(hs[hi] * cfg.fs_hz / 1000.0 + 0.5);
            for (int axis = 0; axis < 2; ++axis) {
                int c = 0;
                for (int k = 0; k < kStayWin && c < kNoiseWin; ++k) {
                    const int64_t t = n - 1 - k;
                    if (!in_ring(t) || !in_ring(t - h)) break;
                    const Bin& a = bin(t); const Bin& o = bin(t - h);
                    if (!a.valid || !o.valid || a.state != State::Fixation || o.state != State::Fixation) continue;
                    scratch[static_cast<size_t>(c++)] = static_cast<float>(std::fabs((axis == 0 ? a.x - o.x : a.y - o.y)));
                }
                if (c >= 20) stay_scale[hi][axis] = std::max(median_of(scratch.data(), c) / kLn2, 0.003);
            }
        }
    }

    // ------------------------------------------------------------------ the saccade in progress and the forecast
    int64_t ongoing_onset() const noexcept {                    // first bin of the run of Saccade bins ending at the newest bin (-1: none)
        const int64_t last = n - 1;
        if (last < 0 || bin(last).state != State::Saccade) return -1;
        int64_t onset = last, k = last - 1;
        int gap = 0;
        while (in_ring(k)) {
            const Bin& p = bin(k);
            if (p.state == State::Saccade) { onset = k; gap = 0; }
            else if (p.valid && p.state == State::Fixation && gap < cfg.merge_gap_samples) ++gap;   // short dropout inside the run
            else break;
            --k;
        }
        return onset;
    }

    void forecast(Output& out) noexcept {
        const int64_t last = n - 1;
        const Bin& cur = bin(last);
        out.forecast = Forecast{};
        if (!cur.valid || cur.state == State::Blink || cur.state == State::Invalid) { out.flags |= flag::NoForecast; return; }
        // 'stay': the current position (a smoothed one lags behind smooth pursuit)
        const double mx = cur.x, my = cur.y;
        Interval* iv[2] = {&out.forecast.at10, &out.forecast.at20};
        for (int h = 0; h < 2; ++h) *iv[h] = Interval{mx, my, stay_scale[h][0] * kStayScaleFactor, stay_scale[h][1] * kStayScaleFactor, true};
        const int64_t on = out.event_onset;
        if (on < 0 || !in_ring(on)) return;
        // ballistic: fixed tables (fit offline on human-labeled saccades), features = distance moved d, current speed v, peak speed vmax
        const Bin& b0 = in_ring(on - 1) ? bin(on - 1) : bin(on);
        const double ddx = cur.x - b0.x, ddy = cur.y - b0.y, d = std::hypot(ddx, ddy);
        double ux, uy;
        if (d > 0.02) { ux = ddx / d; uy = ddy / d; }
        else {                                                  // direction from the current velocity
            double sx = 0, sy = 0; for (int k = 0; k < 3; ++k) if (in_ring(last - k)) { sx += bin(last - k).dx; sy += bin(last - k).dy; }
            const double sn = std::hypot(sx, sy);
            if (sn < 1e-9) return;
            ux = sx / sn; uy = sy / sn;
        }
        double vmax = 0;
        for (int64_t k = on; k <= last; ++k) if (in_ring(k)) vmax = std::max(vmax, static_cast<double>(bin(k).speed) / 1000.0);
        const double v = cur.speed / 1000.0;                    // deg/ms
        const double tau_ms = static_cast<double>(last - on) * 1000.0 / cfg.fs_hz;
        int bi = static_cast<int>(tau_ms / kLandBinMs); bi = std::min(std::max(bi, 0), kLandBins - 1);
        double r[kLandTargets];
        for (int t = 0; t < kLandTargets; ++t) r[t] = kLandCoef[t][bi][0] * d + kLandCoef[t][bi][1] * v + kLandCoef[t][bi][2] * vmax;
        r[0] = std::max(r[0], 0.0);
        r[1] = std::min(std::max(r[1], 0.0), r[0]);
        r[2] = std::min(std::max(r[2], r[1]), r[0]);
        const double infl = tau_ms < 10.0 ? kBallisticInflationEarly : kBallisticInflationLate;
        for (int h = 0; h < 2; ++h) *iv[h] = Interval{cur.x + r[h + 1] * ux, cur.y + r[h + 1] * uy, std::fabs(ux) * kLandScale[h + 1][bi] * infl + 0.003, std::fabs(uy) * kLandScale[h + 1][bi] * infl + 0.003, true};
        out.forecast.ballistic = true;
        predicted_amplitude = static_cast<float>(d + r[0]);
    }
    float predicted_amplitude = 0;

    // ------------------------------------------------------------------ one push
    void push(const Sample& s, Output& out) noexcept {
        rev_first = rev_last = -1;
        uint32_t sflags = 0;
        bool ok = std::isfinite(s.x_deg) && std::isfinite(s.y_deg);
        if (!ok) sflags |= flag::NonFinite;
        else if (std::fabs(s.x_deg) > cfg.max_abs_deg || std::fabs(s.y_deg) > cfg.max_abs_deg) { ok = false; sflags |= flag::OutOfRange; }
        // timestamps: gaps become invalid bins, a long gap resets everything, a backwards timestamp makes the sample invalid
        if (s.t_us >= 0.0 && std::isfinite(s.t_us)) {
            const double dt_us = 1e6 / cfg.fs_hz;
            if (have_t) {
                if (s.t_us <= last_t) { ok = false; sflags |= flag::TimeBackwards; }
                else {
                    const double missing_d = std::floor((s.t_us - last_t) / dt_us + 0.5) - 1.0;
                    if (missing_d >= cfg.gap_reset_samples) { reset_state(); ++st.resets; sflags |= flag::Reset | flag::Gap; }
                    else if (missing_d >= 1.0) {
                        const int missing = static_cast<int>(missing_d);
                        for (int k = 0; k < missing; ++k) process(false, 0, 0, flag::Missing | flag::Gap);
                        sflags |= flag::Gap;
                    }
                    last_t = s.t_us;
                }
            } else { last_t = s.t_us; have_t = true; }
        } else if (s.t_us >= 0.0 && !std::isfinite(s.t_us)) sflags |= flag::NonFinite;
        process(ok, ok ? s.x_deg : 0.0, ok ? s.y_deg : 0.0, sflags);

        const int64_t last = n - 1;
        const Bin& b = bin(last);
        out = Output{};
        out.index = last; out.state = b.state; out.p_saccade = b.p; out.source = b.source; out.flags = b.flags | sflags;
        out.revised_first = rev_first; out.revised_last = rev_last;
        out.event_onset = ongoing_onset();
        predicted_amplitude = 0;
        forecast(out);
        // health
        Health h = Health::Ok;
        if (models.fast && (!nn_trust)) { h = Health::Degraded; out.flags |= nn_stuck ? flag::NnStuck : flag::NnDegraded; }
        if (static_cast<double>(invalid_run) * 1000.0 / cfg.fs_hz > 500.0 || !rate_ok) h = Health::Failed;
        out.health = h;
        if (!rate_ok) out.flags |= flag::UnsupportedRate;
    }

    // ------------------------------------------------------------------ segmentation of the last window_bins bins
    void snapshot(Snapshot& o) const noexcept {
        o = Snapshot{};
        o.newest = n - 1;
        if (n == 0) return;
        const int w = std::min(cfg.window_bins, kMaxBins);
        const int64_t first = std::max<int64_t>(n - w, std::max<int64_t>(0, n - kRing + 8));
        for (int64_t k = first; k < n; ++k) {
            const Bin& b = bin(k);
            if (o.n_bins >= kMaxBins) break;
            o.bins[static_cast<size_t>(o.n_bins)] = BinView{b.index, b.state, b.p, b.source, b.flags, b.revisions, b.final_, b.stage};
            ++o.n_bins;
        }
        int64_t k = first;
        while (k < n && o.n_events < kMaxEvents) {
            if (bin(k).state != State::Saccade) { ++k; continue; }
            int64_t s0 = k, e = k; int64_t j = k + 1; int gap = 0;
            while (j < n) {                                             // extend, merging short gaps of valid fixation bins
                const Bin& b = bin(j);
                if (b.state == State::Saccade) { e = j; gap = 0; ++j; }
                else if (b.state == State::Fixation && b.valid && gap < cfg.merge_gap_samples) { ++gap; ++j; }
                else break;
            }
            k = e + 1;
            if (e - s0 + 1 < cfg.min_event_samples) continue;
            if (o.n_events >= kMaxEvents) break;               // explicit bound (the loop condition already guarantees it)
            Event& ev = o.events[static_cast<size_t>(o.n_events)];   // filled in place (o was value-initialised above)
            ev.onset = s0; ev.offset = e; ev.ongoing = (e == n - 1);
            ev.provisional = e > n - 1 - (models.window ? wage + whop : refine_delay);
            const Bin& b0 = (s0 - 1 >= first) ? bin(s0 - 1) : bin(s0);
            const Bin& b1 = bin(e);
            const double dx = b1.x - b0.x, dy = b1.y - b0.y;
            ev.amplitude_deg = static_cast<float>(std::hypot(dx, dy));
            ev.direction_deg = static_cast<float>(std::atan2(dy, dx) * 180.0 / kPi);
            float pk = 0; for (int64_t q = s0; q <= e; ++q) pk = std::max(pk, bin(q).speed);
            ev.peak_speed_deg_s = pk;
            ev.predicted_amplitude_deg = ev.amplitude_deg;
            if (ev.ongoing && last_prediction_valid && last_prediction_onset == s0) ev.predicted_amplitude_deg = last_prediction_amp;
            const double amp = ev.ongoing ? ev.predicted_amplitude_deg : ev.amplitude_deg;
            ev.cls = amp < cfg.micro_amplitude_deg ? EventClass::Microsaccade : EventClass::Saccade;
            ++o.n_events;
        }
    }
    // the amplitude predicted by the tables at the newest bin (stored by push) for the ongoing event
    bool last_prediction_valid = false; int64_t last_prediction_onset = -1; float last_prediction_amp = 0;
};

GazeEngine::GazeEngine(const Config& cfg, Models models) : p_(new Impl(cfg, std::move(models))) {}
GazeEngine::~GazeEngine() = default;

void GazeEngine::push(const Sample& s, Output& out) noexcept {
    p_->push(s, out);
    p_->last_prediction_valid = out.forecast.ballistic && out.event_onset >= 0;
    p_->last_prediction_onset = out.event_onset;
    p_->last_prediction_amp = p_->predicted_amplitude;
}
void GazeEngine::snapshot(Snapshot& out) const noexcept { p_->snapshot(out); }
void GazeEngine::reset() noexcept { p_->reset_state(); p_->st = Stats{}; }
int64_t GazeEngine::samples() const noexcept { return p_->n; }
GazeEngine::Stats GazeEngine::stats() const noexcept { return p_->st; }

}  // namespace gaze
}  // namespace uneye

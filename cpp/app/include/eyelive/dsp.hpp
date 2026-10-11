// EyeLive signal processing, header-only, same numerical conventions as the Python reference (free_saccade/detectors.py, foundation/universal_hmm.py, foundation/accel_boundaries.py):
//   fill (linear interpolation over invalid samples), 5-point velocity, detrended velocity (minus a running median of 100 ms), robust noise, score log10(1 + |v| / sigma),
//   minimum-duration HMM (fixation + a chain of d saccade states, Viterbi), invalid-zone suppression, and the acceleration-zero boundary refinement (Savitzky-Golay, order 3).
// All functions work on whole traces (batch); the streaming engine calls them on the recent past. Positions in degrees, time in seconds, rates in Hz.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace eyelive {
namespace dsp {

using Mask = std::vector<uint8_t>;
using Events = std::vector<std::pair<int, int>>;                    // inclusive [onset, offset] sample indices

inline double median_of(std::vector<double> v) {                    // numpy.median
    if (v.empty()) return 0.0;
    size_t n = v.size(), h = n / 2; std::nth_element(v.begin(), v.begin() + h, v.end()); double hi = v[h];
    if (n % 2) return hi;
    double lo = *std::max_element(v.begin(), v.begin() + h); return 0.5 * (lo + hi);
}

// linear interpolation over invalid samples, constant extrapolation at both ends, zeros if nothing is valid (detectors.fill)
inline void fill_positions(const std::vector<double>& x, const std::vector<double>& y, const Mask& valid, std::vector<double>& fx, std::vector<double>& fy) {
    size_t T = x.size(); fx = x; fy = y;
    for (int a = 0; a < 2; ++a) {
        std::vector<double>& f = a == 0 ? fx : fy; const std::vector<double>& src = a == 0 ? x : y;
        long first = -1, last = -1;
        for (size_t t = 0; t < T; ++t) if (valid[t]) { if (first < 0) first = (long)t; last = (long)t; }
        if (first < 0) { std::fill(f.begin(), f.end(), 0.0); continue; }
        long prev = -1;
        for (size_t t = 0; t < T; ++t) {
            if (valid[t]) { f[t] = src[t]; prev = (long)t; continue; }
            if ((long)t < first) { f[t] = src[first]; continue; }
            if ((long)t > last) { f[t] = src[last]; continue; }
            size_t nx = t; while (!valid[nx]) ++nx;
            double w = double(t - prev) / double(nx - prev); f[t] = src[prev] * (1 - w) + src[nx] * w;
        }
    }
}

// 5-point moving-difference velocity in deg/s (Engbert & Kliegl); first / last 2 samples are 0 (detectors.velocity)
inline void velocity5(const std::vector<double>& p, double fs, std::vector<double>& v) {
    size_t T = p.size(); v.assign(T, 0.0);
    for (size_t t = 2; t + 2 < T; ++t) v[t] = (p[t + 2] + p[t + 1] - p[t - 1] - p[t - 2]) * fs / 6.0;
}

// scipy.ndimage.median_filter(v, size=w, mode="nearest") on one axis
inline void running_median(const std::vector<double>& v, int w, std::vector<double>& out) {
    long T = (long)v.size(), h = w / 2; out.assign(T, 0.0); std::vector<double> win(w);
    for (long t = 0; t < T; ++t) {
        for (long k = -h; k <= h; ++k) win[k + h] = v[std::min(std::max(t + k, 0L), T - 1)];
        std::nth_element(win.begin(), win.begin() + h, win.end()); out[t] = win[h];
    }
}

// sqrt(max(median(v^2) - median(v)^2, 1e-12)) of one axis (detectors.robust_sigma)
inline double robust_sigma_axis(const std::vector<double>& v) {
    std::vector<double> sq(v.size()); for (size_t i = 0; i < v.size(); ++i) sq[i] = v[i] * v[i];
    double m2 = median_of(sq), m1 = median_of(v); return std::sqrt(std::max(m2 - m1 * m1, 1e-12));
}

struct Kin {                                                       // everything derived from one trace
    std::vector<double> fx, fy, vx, vy, dvx, dvy; std::vector<float> score; Mask valid; double noise = 0;
};

// the score used by the universal HMM: log10(1 + |detrended velocity| / noise), noise = mean over the two axes of the robust sigma of the detrended velocity (universal_hmm.unit_free_score)
inline void score_trace(const std::vector<double>& x, const std::vector<double>& y, const Mask& valid, double fs, Kin& k) {
    size_t T = x.size(); k.valid = valid; fill_positions(x, y, valid, k.fx, k.fy); velocity5(k.fx, fs, k.vx); velocity5(k.fy, fs, k.vy);
    int w = int(0.1 * fs) | 1; std::vector<double> mx, my; running_median(k.vx, w, mx); running_median(k.vy, w, my); k.dvx.resize(T); k.dvy.resize(T);
    for (size_t t = 0; t < T; ++t) { k.dvx[t] = k.vx[t] - mx[t]; k.dvy[t] = k.vy[t] - my[t]; }
    k.noise = std::max(0.5 * (robust_sigma_axis(k.dvx) + robust_sigma_axis(k.dvy)), 1e-9); k.score.resize(T);
    for (size_t t = 0; t < T; ++t) k.score[t] = (float)std::log10(1.0 + std::hypot(k.dvx[t], k.dvy[t]) / k.noise);
}

struct HmmParams { double fs = 1000; int d = 6; double mu0 = 0.4166, mu1 = 1.3204, sd0 = 0.2011, sd1 = 0.7662, p_on = 0.00554, p_stay = 0.9495; };

// Viterbi on the score; state 0 = fixation, 1..d = the saccade chain (every saccade lasts at least d samples); invalid samples are uninformative (log-likelihood 0)
inline Mask viterbi(const std::vector<float>& f, const Mask& valid, const HmmParams& h) {
    size_t T = f.size(); int K = h.d + 1; Mask lab(T, 0); if (T == 0) return lab;
    std::vector<double> A(K * K, -1e9);
    A[0] = std::log(1 - h.p_on); A[1] = std::log(h.p_on);
    for (int i = 1; i < h.d; ++i) A[i * K + i + 1] = 0.0;
    A[h.d * K + h.d] = std::log(h.p_stay); A[h.d * K + 0] = std::log(1 - h.p_stay);
    std::vector<int8_t> back(T * K, 0); std::vector<double> delta(K, -1e9), nd(K), ll(K);
    auto emit = [&](size_t t) {
        if (!valid[t]) { std::fill(ll.begin(), ll.end(), 0.0); return; }
        double z0 = (f[t] - h.mu0) / h.sd0, z1 = (f[t] - h.mu1) / h.sd1; ll[0] = -0.5 * z0 * z0 - std::log(h.sd0);
        double s = -0.5 * z1 * z1 - std::log(h.sd1); for (int k = 1; k < K; ++k) ll[k] = s;
    };
    emit(0); delta[0] = ll[0];
    for (size_t t = 1; t < T; ++t) {
        emit(t);
        for (int j = 0; j < K; ++j) {
            double best = -1e300; int arg = 0;
            for (int i = 0; i < K; ++i) { double s = delta[i] + A[i * K + j]; if (s > best) { best = s; arg = i; } }
            nd[j] = best + ll[j]; back[t * K + j] = (int8_t)arg;
        }
        delta = nd;
    }
    int st = (int)(std::max_element(delta.begin(), delta.end()) - delta.begin());
    for (size_t t = T - 1;; --t) { lab[t] = st > 0; if (t == 0) break; st = back[t * K + st]; }
    return lab;
}

inline Events runs_of(const Mask& m) {                              // detectors.runs_1d, inclusive ends
    Events ev; long T = (long)m.size();
    for (long t = 0; t < T; ++t) if (m[t] && (t == 0 || !m[t - 1])) { long e = t; while (e + 1 < T && m[e + 1]) ++e; ev.emplace_back((int)t, (int)e); t = e; }
    return ev;
}

inline void drop_invalid_zone_and_short(Mask& lab, const Mask& valid, double fs, int min_samples, double margin_ms = 20.0) {
    long T = (long)lab.size(), m = (long)std::nearbyint(margin_ms * fs / 1000.0); std::vector<uint8_t> bad(T, 0);
    for (long t = 0; t < T; ++t) if (!valid[t]) for (long k = std::max(0L, t - m); k <= std::min(T - 1, t + m); ++k) bad[k] = 1;
    for (long t = 0; t < T; ++t) if (bad[t]) lab[t] = 0;
    for (auto& r : runs_of(lab)) if (r.second - r.first + 1 < min_samples) for (int t = r.first; t <= r.second; ++t) lab[t] = 0;       // cleanup(min_ms = d samples, merge 0)
}

// HMM events of a whole trace (the label-free detector of foundation/universal_hmm.py)
inline Mask hmm_detect(const Kin& k, const HmmParams& h) {
    Mask lab = viterbi(k.score, k.valid, h); drop_invalid_zone_and_short(lab, k.valid, h.fs, h.d); return lab;
}

// ---------------------------------------------------------------- Savitzky-Golay derivatives (order 3, scipy mode="interp") and the acceleration-zero boundaries
struct SG { int w = 0; std::vector<double> P; };                     // P: 4 x w pseudo-inverse of the Vandermonde matrix over x = -h..h

inline SG make_sg(int w) {
    SG s; s.w = w; int h = (w - 1) / 2; const int n = 4; double VtV[4][4] = {}, V[64][4];
    for (int i = 0; i < w; ++i) { double x = i - h; V[i][0] = 1; V[i][1] = x; V[i][2] = x * x; V[i][3] = x * x * x; }
    for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b) for (int i = 0; i < w; ++i) VtV[a][b] += V[i][a] * V[i][b];
    double M[4][8]; for (int a = 0; a < n; ++a) { for (int b = 0; b < n; ++b) { M[a][b] = VtV[a][b]; M[a][n + b] = a == b; } }
    for (int c = 0; c < n; ++c) { int p = c; for (int r = c + 1; r < n; ++r) if (std::fabs(M[r][c]) > std::fabs(M[p][c])) p = r; for (int k = 0; k < 2 * n; ++k) std::swap(M[c][k], M[p][k]);
        double d = M[c][c]; for (int k = 0; k < 2 * n; ++k) M[c][k] /= d; for (int r = 0; r < n; ++r) if (r != c) { double f = M[r][c]; for (int k = 0; k < 2 * n; ++k) M[r][k] -= f * M[c][k]; } }
    s.P.assign(n * w, 0.0); for (int a = 0; a < n; ++a) for (int i = 0; i < w; ++i) { double acc = 0; for (int b = 0; b < n; ++b) acc += M[a][n + b] * V[i][b]; s.P[a * w + i] = acc; }
    return s;
}

// first and second derivative (per second, per second^2) of a smoothed trace: a cubic is fitted on the window [i - h, i + h] (at the edges on the first / last window, "interp") and differentiated at i
inline void sg_deriv(const std::vector<double>& y, const SG& sg, double fs, std::vector<double>& d1, std::vector<double>& d2) {
    long T = (long)y.size(), w = sg.w, h = (w - 1) / 2; d1.assign(T, 0.0); d2.assign(T, 0.0);
    for (long i = 0; i < T; ++i) {
        long s0 = std::min(std::max(i - h, 0L), T - w); double c[4] = {0, 0, 0, 0};
        for (int a = 0; a < 4; ++a) { double acc = 0; for (long j = 0; j < w; ++j) acc += sg.P[a * w + j] * y[s0 + j]; c[a] = acc; }
        double u = double(i - (s0 + h)); d1[i] = (c[1] + 2 * c[2] * u + 3 * c[3] * u * u) * fs; d2[i] = (2 * c[2] + 6 * c[3] * u) * fs * fs;
    }
}

struct Refine { double win_ms = 15.0, margin_ms = 25.0; bool offset_first = false, offset_too = true; };   // offset_first: PSO not included (first inflection); offset_too = false: onset only

inline Mask refine_events(const Events& ev, const std::vector<double>& fx, const std::vector<double>& fy, const Mask& valid, double fs, const Refine& r) {
    long T = (long)fx.size(); Mask out(T, 0); if (ev.empty()) return out;
    int w = (int(std::nearbyint(r.win_ms * fs / 1000.0)) | 1); w = std::max(w, 5); w = std::max(std::min<long>(w, (T / 2) * 2 - 1), 5L); SG sg = make_sg(w);
    std::vector<double> vx, ax, vy, ay; sg_deriv(fx, sg, fs, vx, ax); sg_deriv(fy, sg, fs, vy, ay);
    std::vector<double> sp(T), at(T); for (long t = 0; t < T; ++t) { sp[t] = std::hypot(vx[t], vy[t]); at[t] = (vx[t] * ax[t] + vy[t] * ay[t]) / std::max(sp[t], 1e-9); }
    long nv = 0; for (long t = 0; t < T; ++t) nv += valid[t]; std::vector<double> sx, sy; for (long t = 0; t < T; ++t) if (nv < 20 || valid[t]) { sx.push_back(vx[t]); sy.push_back(vy[t]); }
    double sgm = 0.5 * (robust_sigma_axis(sx) + robust_sigma_axis(sy)), floor_ = 3.0 * sgm; long m = std::max((long)std::nearbyint(r.margin_ms * fs / 1000.0), 2L);
    for (size_t q = 0; q < ev.size(); ++q) {
        long s = ev[q].first, e = ev[q].second, lo = std::max(s - m, q > 0 ? (long)ev[q - 1].second + 1 : 0L), hi = std::min(e + m, q + 1 < ev.size() ? (long)ev[q + 1].first - 1 : T - 1);
        long pk = s; for (long t = s; t <= e; ++t) if (sp[t] > sp[pk]) pk = t;
        long j = pk - 1; while (j >= lo && at[j] > 0) --j; long on = std::min(j + 1, pk), off = e;
        if (r.offset_too) {
            if (r.offset_first) { j = pk + 1; while (j <= hi && at[j] < 0) ++j; off = j > pk + 1 ? j - 1 : e; }
            else {
                long last = -1; for (long t = pk; t <= hi; ++t) if (sp[t] >= floor_) last = t;
                if (last < 0) off = e; else { j = last; while (j <= hi && at[j] < 0) ++j; off = std::min(j, hi); }
            }
        }
        if (off >= on) for (long t = on; t <= off; ++t) out[t] = 1;
    }
    return out;
}

}  // namespace dsp
}  // namespace eyelive

#include "eyelive/train/dataset.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace eyelive { namespace train {

int window_for_rate(double rate_hz) { return std::max(25, int(std::lround(rate_hz / 25.0)) * 25); }

double noise_std(const std::vector<float>& v) {
    std::vector<double> d;
    for (size_t i = 1; i < v.size(); ++i)
        if (std::isfinite(v[i]) && std::isfinite(v[i - 1])) d.push_back(std::fabs(double(v[i]) - double(v[i - 1])));
    if (d.size() < 8) return 0.0;
    std::sort(d.begin(), d.end());
    return d[d.size() / 2] / 0.6745;
}

void fill_invalid(std::vector<float>& v, double std_, std::mt19937_64& rng) {
    std::normal_distribution<double> nd(0.0, std_);
    const size_t n = v.size();
    size_t i = 0;
    while (i < n) {
        if (std::isfinite(v[i])) { ++i; continue; }
        size_t j = i;
        while (j < n && !std::isfinite(v[j])) ++j;
        const double c = i > 0 ? double(v[i - 1]) : (j < n ? double(v[j]) : 0.0);
        for (size_t k = i; k < j; ++k) v[k] = float(c + nd(rng));
        i = j;
    }
}

static void append_window(Windows& w, const std::vector<float>& x, const std::vector<float>& y, const std::vector<uint8_t>& l, size_t s, int orient, double lim) {   // lim: speed limit in position units per sample (0 = off)
    const int W = w.W; const size_t base = w.vel.size(); w.vel.resize(base + 2 * size_t(W));
    for (int t = 0; t < W; ++t) {
        float dx = 0, dy = 0;
        if (t > 0) { dx = x[s + t] - x[s + t - 1]; dy = y[s + t] - y[s + t - 1]; }
        if (!std::isfinite(dx) || !std::isfinite(dy)) dx = dy = 0;                 // blinks / lost samples (the original sets NaN to 0)
        if (lim > 0) { const double n = std::hypot(double(dx), double(dy)); if (n > lim) { dx = float(dx * lim / n); dy = float(dy * lim / n); } }   // speed limit, direction kept
        w.vel[base + t] = dx; w.vel[base + W + t] = dy;
    }
    w.lab.insert(w.lab.end(), l.begin() + s, l.begin() + s + W); w.orient.push_back(orient); ++w.n;
}


// labelled extent [a, b] of a trace (-1 if nothing is labelled)
static void labelled_extent(const std::vector<uint8_t>& l, long& a, long& b) {
    a = b = -1;
    for (size_t i = 0; i < l.size(); ++i) if (l[i] != kIgnoreLabel) { if (a < 0) a = long(i); b = long(i); }
}

// a copy of [s, s + W) of x / y / l, padded (last value, ignored label) when the trace is shorter than W
static void slice(const std::vector<float>& x, const std::vector<float>& y, const std::vector<uint8_t>& l, long s, int W,
                  std::vector<float>& xs, std::vector<float>& ys, std::vector<uint8_t>& ls) {
    xs.assign(size_t(W), 0.f); ys.assign(size_t(W), 0.f); ls.assign(size_t(W), kIgnoreLabel);
    const long T = long(x.size());
    for (int t = 0; t < W; ++t) {
        const long i = s + t;
        if (i < T) { xs[size_t(t)] = x[size_t(i)]; ys[size_t(t)] = y[size_t(i)]; ls[size_t(t)] = l[size_t(i)]; }
        else if (T > 0) { xs[size_t(t)] = x[size_t(T - 1)]; ys[size_t(t)] = y[size_t(T - 1)]; }
    }
}

void resample_train(Dataset& d, const TrainConfig& cfg, std::mt19937_64& rng) {
    const int W = d.train.W;
    d.train = Windows(); d.train.W = W;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const long step = std::max(1L, long(W) - long(W) / 5);           // 0.8 W, windows overlap by 20 %
    const long jitter = std::max(1L, long(0.3 * W));                 // up to 0.3 W before the labelled start
    std::vector<float> xs, ys, xr, yr; std::vector<uint8_t> ls;
    for (size_t k = 0; k < d.tx.size(); ++k) {
        long a, b; labelled_extent(d.tl[k], a, b);
        if (a < 0) continue;
        const long T = long(d.tx[k].size());
        for (long s0 = a; s0 <= b; s0 += step) {
            long s = s0 - long(U(rng) * double(jitter));
            s = std::max(0L, std::min(s, std::max(0L, T - W)));
            slice(d.tx[k], d.ty[k], d.tl[k], s, W, xs, ys, ls);
            append_window(d.train, xs, ys, ls, 0, 0, d.lim);
            const int nrot = cfg.rotation_aug ? std::max(0, cfg.rotations) : 0;
            for (int r = 1; r <= nrot; ++r) {
                const double th = 2.0 * 3.14159265358979323846 * U(rng), c = std::cos(th), sn = std::sin(th), m = U(rng) < 0.5 ? -1.0 : 1.0;
                xr.resize(size_t(W)); yr.resize(size_t(W));
                for (int t = 0; t < W; ++t) { xr[size_t(t)] = float((xs[size_t(t)] * c + ys[size_t(t)] * sn) * m); yr[size_t(t)] = float(-xs[size_t(t)] * sn + ys[size_t(t)] * c); }
                append_window(d.train, xr, yr, ls, 0, r, d.lim);
            }
        }
    }
}

// cfg.resample_each_epoch: whole traces held out for validation, training traces kept for resample_train
static Dataset make_dataset_resampled(const std::vector<TrainTrace>& traces, const TrainConfig& cfg, Dataset d, int W, double lim) {
    const int K = cfg.n_classes;
    d.lim = lim;
    std::vector<size_t> use;
    for (size_t i = 0; i < traces.size(); ++i) {
        const auto& tr = traces[i];
        if (tr.x.size() != tr.y.size() || tr.x.size() != tr.labels.size()) { d.error = "trace arrays differ in length"; return d; }
        bool any = false; for (auto v : tr.labels) if (v < K) { any = true; break; }
        if (any) use.push_back(i);
    }
    if (use.size() < 2) { d.error = "not enough reviewed data: need at least two traces with labels (whole traces are held out for validation)"; return d; }
    std::mt19937_64 rng(cfg.seed); std::shuffle(use.begin(), use.end(), rng);
    size_t nval = std::max<size_t>(1, size_t(std::lround(cfg.val_fraction * double(use.size())))); nval = std::min(nval, use.size() - 1);
    std::mt19937_64 frng(cfg.seed + 31);
    std::vector<float> xs, ys; std::vector<uint8_t> ls;
    for (size_t j = 0; j < use.size(); ++j) {
        const auto& tr = traces[use[j]];
        std::vector<float> fx = tr.x, fy = tr.y; std::vector<uint8_t> l = tr.labels;
        fill_invalid(fx, std::max(3.0 * noise_std(fx), 1e-3), frng);
        fill_invalid(fy, std::max(3.0 * noise_std(fy), 1e-3), frng);
        for (auto& v : l) if (v >= K) v = kIgnoreLabel;
        if (j < nval) {                                               // validation trace: fixed windows over its labelled extent, no augmentation
            long a, b; labelled_extent(l, a, b);
            const long T = long(fx.size());
            for (long s = a; s <= b; s += W) {
                const long s1 = std::max(0L, std::min(s, std::max(0L, T - W)));
                slice(fx, fy, l, s1, W, xs, ys, ls);
                append_window(d.val, xs, ys, ls, 0, 0, lim);
            }
        } else { d.tx.push_back(std::move(fx)); d.ty.push_back(std::move(fy)); d.tl.push_back(std::move(l)); }
    }
    std::mt19937_64 r(cfg.seed + 7);
    resample_train(d, cfg, r);
    if (d.train.n == 0 || d.val.n == 0) d.error = "not enough reviewed data for training and validation windows";
    return d;
}

Dataset make_dataset(const std::vector<TrainTrace>& traces, const TrainConfig& cfg) {
    Dataset d; d.n_classes = cfg.n_classes; const int K = cfg.n_classes;
    if (traces.empty()) { d.error = "no training trace"; return d; }
    d.rate_hz = traces[0].rate_hz; const int W = window_for_rate(d.rate_hz); d.train.W = d.val.W = W;
    const double lim = cfg.max_speed_deg_s > 0 ? cfg.max_speed_deg_s / d.rate_hz : 0.0;     // deg per sample (the trace positions are in degrees at this point)
    if (cfg.resample_each_epoch) return make_dataset_resampled(traces, cfg, d, W, lim);
    struct Win { const TrainTrace* tr; size_t s; };
    std::vector<Win> wins;
    size_t used = 0;                                                                    // samples taken so far, over ALL traces (the budget is a total)
    for (auto it = traces.rbegin(); it != traces.rend(); ++it) {                       // most recent first, at most max_seconds_of_data
        const auto& tr = *it; if (tr.x.size() != tr.y.size() || tr.x.size() != tr.labels.size()) { d.error = "trace arrays differ in length"; return d; }
        const size_t budget = size_t(std::max(0.0, cfg.max_seconds_of_data) * tr.rate_hz);
        size_t end = tr.x.size();
        while (end >= size_t(W) && used + W <= budget + W / 2) { wins.push_back({&tr, end - W}); end -= W; used += W; }
        if (used >= budget) break;
    }
    if (wins.size() < 2) { d.error = "not enough reviewed data: need at least two windows of about 1 s"; return d; }
    std::mt19937_64 rng(cfg.seed); std::shuffle(wins.begin(), wins.end(), rng);
    size_t nval = std::max<size_t>(1, size_t(std::lround(cfg.val_fraction * double(wins.size())))); nval = std::min(nval, wins.size() - 1);
    // labels: classes >= K and non-finite positions are ignored. Invalid
    // stretches are first filled with noise (see fill_invalid): without this,
    // a blink labelled 2 on a NaN stretch would lose its label exactly where
    // the blink is and 3-class training collapses (the flat segment reads as
    // fixation).
    std::mt19937_64 frng(cfg.seed + 31);
    std::vector<std::vector<float>> fx(traces.size()), fy(traces.size());
    for (size_t i = 0; i < traces.size(); ++i) {
        fx[i] = traces[i].x; fy[i] = traces[i].y;
        const double sx = std::max(3.0 * noise_std(fx[i]), 1e-3);
        const double sy = std::max(3.0 * noise_std(fy[i]), 1e-3);
        fill_invalid(fx[i], sx, frng);
        fill_invalid(fy[i], sy, frng);
    }
    auto prep = [&](const Win& w, std::vector<float>& x, std::vector<float>& y, std::vector<uint8_t>& l) {
        const size_t ti = size_t(w.tr - traces.data());
        x.assign(fx[ti].begin() + w.s, fx[ti].begin() + w.s + W); y.assign(fy[ti].begin() + w.s, fy[ti].begin() + w.s + W); l.assign(w.tr->labels.begin() + w.s, w.tr->labels.begin() + w.s + W);
        for (int t = 0; t < W; ++t) if (l[t] >= K) l[t] = kIgnoreLabel;
    };
    std::vector<float> x, y; std::vector<uint8_t> l; std::normal_distribution<float> nd(0.f, 1.f); std::uniform_real_distribution<float> ud(0.f, 1.f);
    for (size_t i = 0; i < wins.size(); ++i) {
        prep(wins[i], x, y, l);
        if (i < nval) { append_window(d.val, x, y, l, 0, 0, lim); continue; }
        const int nrot = cfg.rotation_aug ? std::max(0, cfg.rotations) : 0;
        for (int r = 0; r <= nrot; ++r) {
            std::vector<float> xr = x, yr = y;
            if (r > 0) { const double th = 3.14159265358979323846 * (0.25 + 2.0 * (r - 1) / nrot), c = std::cos(th), s = std::sin(th); for (int t = 0; t < W; ++t) { xr[t] = float(x[t] * c + y[t] * s); yr[t] = float(-x[t] * s + y[t] * c); } }
            if (cfg.gain_aug) { const float g = 0.7f + 0.7f * ud(rng); for (int t = 0; t < W; ++t) { xr[t] *= g; yr[t] *= g; } }
            if (cfg.noise_aug) { const float sd = 0.005f * ud(rng); for (int t = 0; t < W; ++t) { xr[t] += sd * nd(rng); yr[t] += sd * nd(rng); } }
            append_window(d.train, xr, yr, l, 0, r, lim);
        }
    }
    return d;
}

Scores score_predictions(const std::vector<uint8_t>& pred, const std::vector<uint8_t>& truth, int W, int K) {
    Scores sc; const size_t n = truth.size(); std::vector<double> conf(size_t(K) * K, 0.0); double tot = 0;
    for (size_t i = 0; i < n; ++i) if (truth[i] < K) { conf[size_t(truth[i]) * K + pred[i]] += 1; tot += 1; }
    if (tot > 0) { double po = 0, pe = 0; for (int a = 0; a < K; ++a) { po += conf[size_t(a) * K + a]; double r = 0, c = 0; for (int b = 0; b < K; ++b) { r += conf[size_t(a) * K + b]; c += conf[size_t(b) * K + a]; } pe += r * c; } po /= tot; pe /= tot * tot; sc.kappa = pe < 1 ? (po - pe) / (1 - pe) : 1.0; }
    int tp_p = 0, np = 0, tp_t = 0, nt = 0; const size_t nw = n / size_t(W);
    for (size_t w = 0; w < nw; ++w) {
        const uint8_t* p = &pred[w * W]; const uint8_t* t = &truth[w * W];
        auto runs = [&](const uint8_t* a, int minlen) { std::vector<std::pair<int, int>> r; for (int i = 0; i < W;) { if (a[i] == 1) { int j = i; while (j < W && a[j] == 1) ++j; if (j - i >= minlen) r.push_back({i, j - 1}); i = j; } else ++i; } return r; };
        auto pr = runs(p, 3), tr = runs(t, 1);
        for (auto& a : pr) { ++np; for (auto& b : tr) if (a.first <= b.second && b.first <= a.second) { ++tp_p; break; } }
        for (auto& b : tr) { ++nt; for (auto& a : pr) if (a.first <= b.second && b.first <= a.second) { ++tp_t; break; } }
    }
    const double prec = np ? double(tp_p) / np : 0, rec = nt ? double(tp_t) / nt : 0; sc.event_f1 = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0; return sc;
}

}}  // namespace eyelive::train

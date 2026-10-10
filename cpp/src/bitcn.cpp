#include "uneye/bitcn.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>

namespace uneye {

static double median(std::vector<double>& v) {
    size_t n = v.size(), h = n / 2;
    if (n == 0) return 0.0;
    std::nth_element(v.begin(), v.begin() + h, v.end());
    double hi = v[h];
    if (n % 2) return hi;
    double lo = *std::max_element(v.begin(), v.begin() + h);
    return 0.5 * (lo + hi);
}

void features2(const float* pos, int T, double fs, float* out) {
    if (T <= 0) return;
    std::vector<double> p(2 * (size_t)T);
    std::vector<unsigned char> valid(T);
    for (int t = 0; t < T; ++t) valid[t] = std::isfinite(pos[2 * t]) && std::isfinite(pos[2 * t + 1]);
    for (int a = 0; a < 2; ++a) {                              // linear interpolation over invalid samples, constant extrapolation at the ends
        int first = -1, last = -1;
        for (int t = 0; t < T; ++t) if (valid[t]) { if (first < 0) first = t; last = t; }
        if (first < 0) { for (int t = 0; t < T; ++t) p[2 * (size_t)t + a] = 0.0; continue; }
        int prev = -1;
        for (int t = 0; t < T; ++t) {
            if (valid[t]) { p[2 * (size_t)t + a] = pos[2 * (size_t)t + a]; prev = t; continue; }
            if (t < first) { p[2 * (size_t)t + a] = pos[2 * (size_t)first + a]; continue; }
            if (t > last) { p[2 * (size_t)t + a] = pos[2 * (size_t)last + a]; continue; }
            int nxt = t; while (!valid[nxt]) ++nxt;
            double w = double(t - prev) / double(nxt - prev);
            p[2 * (size_t)t + a] = pos[2 * (size_t)prev + a] * (1 - w) + pos[2 * (size_t)nxt + a] * w;
        }
    }
    std::vector<double> v(2 * (size_t)T, 0.0);
    for (int t = 2; t < T - 2; ++t)
        for (int a = 0; a < 2; ++a)
            v[2 * (size_t)t + a] = (p[2 * (size_t)(t + 2) + a] + p[2 * (size_t)(t + 1) + a] - p[2 * (size_t)(t - 1) + a] - p[2 * (size_t)(t - 2) + a]) * fs / 6.0;
    double sg = 0;
    for (int a = 0; a < 2; ++a) {
        std::vector<double> c(T), c2(T);
        for (int t = 0; t < T; ++t) { c[t] = v[2 * (size_t)t + a]; c2[t] = c[t] * c[t]; }
        double m2 = median(c2), m1 = median(c);
        sg += std::sqrt(std::max(m2 - m1 * m1, 1e-12)) / 2.0;
    }
    sg = std::max(sg, 1e-9);
    for (int t = 0; t < T; ++t)
        for (int a = 0; a < 2; ++a)
            out[2 * (size_t)t + a] = valid[t] ? (float)std::asinh(v[2 * (size_t)t + a] / sg) : 0.0f;
}

BiTCN::BiTCN(const std::string& path) {
    std::unique_ptr<FILE, int (*)(FILE*)> f(std::fopen(path.c_str(), "rb"), &std::fclose);      // closed on every exit, including exceptions
    if (!f) throw std::runtime_error("cannot open " + path);
    auto rd = [&](void* p, size_t n) { if (std::fread(p, 1, n, f.get()) != n) throw std::runtime_error("truncated " + path); };
    int32_t h[5]; rd(h, sizeof h);
    if (h[0] != 0x42544e32) throw std::runtime_error("bad magic in " + path);
    if (h[1] < 1 || h[1] > 16 || h[2] < 1 || h[2] > 512 || h[3] < 0 || h[3] > 64 || h[4] < 1 || h[4] > 31 || h[4] % 2 == 0) throw std::runtime_error("implausible header in " + path);
    nin_ = h[1]; ch_ = h[2]; int nb = h[3]; stem_.k = h[4]; stem_.dil = 1;
    auto vec = [&](std::vector<float>& v, size_t n) { v.resize(n); rd(v.data(), n * 4); };
    vec(stem_.w, (size_t)ch_ * nin_ * stem_.k); vec(stem_.b, ch_); vec(stem_.scale, ch_); vec(stem_.shift, ch_);
    for (int i = 0; i < nb; ++i) {
        Layer L; int32_t d; rd(&d, 4);
        if (d < 1 || d > 4096) throw std::runtime_error("implausible dilation in " + path);
        L.dil = d; L.k = 3;
        vec(L.w, (size_t)ch_ * ch_ * 3); vec(L.b, ch_); vec(L.scale, ch_); vec(L.shift, ch_); blocks_.push_back(std::move(L));
    }
    vec(head_w_, ch_); rd(&head_b_, 4);
}

int BiTCN::receptive_field() const {
    int r = stem_.k;
    for (auto& b : blocks_) r += 2 * b.dil;
    return r;
}

// y[c][t] = act(sum_{i,j} w[c][i][j] x[i][t + (j - k/2) * dil] + b[c]) with zero padding; layout channel-major (ch x T)
static void conv(const BiTCN*, const std::vector<float>& w, const std::vector<float>& b, int cin, int cout, int k, int dil, const float* x, int T, float* y) {
    int half = k / 2;
    for (int c = 0; c < cout; ++c) {
        float* yc = y + (size_t)c * T;
        for (int t = 0; t < T; ++t) yc[t] = b[c];
        for (int i = 0; i < cin; ++i) {
            const float* xi = x + (size_t)i * T;
            for (int j = 0; j < k; ++j) {
                float wv = w[((size_t)c * cin + i) * k + j]; int off = (j - half) * dil;
                int t0 = std::max(0, -off), t1 = std::min(T, T - off);
                for (int t = t0; t < t1; ++t) yc[t] += wv * xi[t + off];
            }
        }
    }
}

void BiTCN::forward(const float* feats, int T, float* logit) const {
    std::vector<float> in((size_t)nin_ * T), h((size_t)ch_ * T), tmp((size_t)ch_ * T);
    for (int t = 0; t < T; ++t) for (int i = 0; i < nin_; ++i) in[(size_t)i * T + t] = feats[(size_t)t * nin_ + i];
    conv(this, stem_.w, stem_.b, nin_, ch_, stem_.k, 1, in.data(), T, h.data());
    for (int c = 0; c < ch_; ++c) for (int t = 0; t < T; ++t) { float& v = h[(size_t)c * T + t]; v = stem_.scale[c] * std::max(v, 0.0f) + stem_.shift[c]; }
    for (auto& L : blocks_) {
        conv(this, L.w, L.b, ch_, ch_, 3, L.dil, h.data(), T, tmp.data());
        for (int c = 0; c < ch_; ++c) for (int t = 0; t < T; ++t) { size_t i = (size_t)c * T + t; h[i] += L.scale[c] * std::max(tmp[i], 0.0f) + L.shift[c]; }
    }
    for (int t = 0; t < T; ++t) {
        float s = head_b_;
        for (int c = 0; c < ch_; ++c) s += head_w_[c] * h[(size_t)c * T + t];
        logit[t] = s;
    }
}

std::vector<unsigned char> detect(const BiTCN& net, const float* pos, int T, double fs) {
    std::vector<float> f((size_t)T * net.nin()), lg(T);
    features2(pos, T, fs, f.data()); net.forward(f.data(), T, lg.data());
    std::vector<unsigned char> m(T);
    for (int t = 0; t < T; ++t) m[t] = lg[t] > 0 && std::isfinite(pos[2 * (size_t)t]) && std::isfinite(pos[2 * (size_t)t + 1]);
    return m;
}

}  // namespace uneye

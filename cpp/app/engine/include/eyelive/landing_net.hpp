// Landing-point network: from the last L samples of an ongoing saccade (positions in degrees, 1 kHz) to the displacement still to come, no rotation augmentation in the training (absolute position
// and direction are inputs: the dynamics depend on the starting point), see foundation/landing_train.py (variant 'ctx') and foundation/landing_gazebase.py. Header-only, no allocation per call.
// File (little endian): magic "EYLNET01", u32 version = 1, u32 L, u32 nin = 15, u32 h1, u32 h2, u32 nout = 3, then float32 W1 (h1 x L*nin, row-major), b1, W2 (h2 x h1), b2, W3 (nout x h2), b3.
// Network: flatten the (L, 15) window (time-major) -> Linear -> GELU -> Linear -> GELU -> Linear -> (dx, dy, remaining duration / 50 ms).
// Channels of every sample of the window (idx = t - L + 1 ... t; p0 = position before the onset; cur = position at t; theta = angle of cur - p0):
//   vx, vy (deg / sample: P[idx] - P[idx-1], 0 if one of them is missing), valid (both present), (P[idx] - p0) / 10 (2, 0 before the onset), in-saccade (idx >= onset), (idx - onset) / 50 (0 before the onset),
//   P[idx] / 10 (2, 0 if missing), p0 / 10 (2), cos theta, sin theta, cos 2 theta, sin 2 theta.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace eyelive {

class LandingNet {
 public:
    bool load(const std::string& path, std::string* err) {
        std::ifstream f(path, std::ios::binary); if (!f) { if (err) *err = "cannot open " + path; return false; }
        char magic[8]; uint32_t hdr[6]; f.read(magic, 8); f.read(reinterpret_cast<char*>(hdr), sizeof hdr);
        if (!f || std::memcmp(magic, "EYLNET01", 8) != 0) { if (err) *err = "not a landing network file"; return false; }
        if (hdr[0] != 1 || hdr[2] != 15 || hdr[5] != 3 || hdr[1] < 8 || hdr[1] > 512 || hdr[3] == 0 || hdr[3] > 4096 || hdr[4] == 0 || hdr[4] > 4096) { if (err) *err = "unsupported landing network"; return false; }
        L_ = int(hdr[1]); h1_ = int(hdr[3]); h2_ = int(hdr[4]);
        auto rd = [&](std::vector<float>& v, size_t n) { v.resize(n); f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(float))); };
        rd(w1_, size_t(h1_) * size_t(L_) * 15); rd(b1_, size_t(h1_)); rd(w2_, size_t(h2_) * size_t(h1_)); rd(b2_, size_t(h2_)); rd(w3_, size_t(3) * size_t(h2_)); rd(b3_, 3);
        if (!f) { if (err) *err = "truncated landing network file"; w1_.clear(); return false; }
        return true;
    }
    bool loaded() const { return !w1_.empty(); }
    int window() const { return L_; }
    // pos(idx, &x, &y) -> bool: the position (degrees) of sample idx, false when missing. t = newest sample, onset = first sample of the saccade (detector). Returns false if there is no direction yet.
    template <class PosFn> bool predict(PosFn pos, int64_t t, int64_t onset, double* dx, double* dy, double* remaining_ms = nullptr) const {
        double p0x = 0, p0y = 0, cx = 0, cy = 0; if (!loaded() || !pos(onset - 1, &p0x, &p0y) || !pos(t, &cx, &cy)) return false;
        const double ddx = cx - p0x, ddy = cy - p0y, d = std::hypot(ddx, ddy); if (d < 0.02) return false;
        const double th = std::atan2(ddy, ddx), c1 = std::cos(th), s1 = std::sin(th), c2 = std::cos(2 * th), s2 = std::sin(2 * th);
        in_.assign(size_t(L_) * 15, 0.f); float* x = in_.data();
        for (int k = 0; k < L_; ++k, x += 15) {
            const int64_t idx = t - L_ + 1 + k; double px = 0, py = 0, qx = 0, qy = 0; const bool ok = pos(idx, &px, &py), okm = pos(idx - 1, &qx, &qy); const bool valid = ok && okm; const bool ins = idx >= onset;
            x[0] = valid ? float(px - qx) : 0.f; x[1] = valid ? float(py - qy) : 0.f; x[2] = valid ? 1.f : 0.f;
            x[3] = (ins && ok) ? float((px - p0x) / 10.0) : 0.f; x[4] = (ins && ok) ? float((py - p0y) / 10.0) : 0.f; x[5] = ins ? 1.f : 0.f; x[6] = ins ? float(double(idx - onset) / 50.0) : 0.f;
            x[7] = ok ? float(px / 10.0) : 0.f; x[8] = ok ? float(py / 10.0) : 0.f; x[9] = float(p0x / 10.0); x[10] = float(p0y / 10.0); x[11] = float(c1); x[12] = float(s1); x[13] = float(c2); x[14] = float(s2);
        }
        h1v_.resize(size_t(h1_)); h2v_.resize(size_t(h2_)); const size_t n_in = size_t(L_) * 15;
        for (int j = 0; j < h1_; ++j) { const float* w = &w1_[size_t(j) * n_in]; float s = b1_[size_t(j)]; for (size_t i = 0; i < n_in; ++i) s += w[i] * in_[i]; h1v_[size_t(j)] = gelu(s); }
        for (int j = 0; j < h2_; ++j) { const float* w = &w2_[size_t(j) * size_t(h1_)]; float s = b2_[size_t(j)]; for (int i = 0; i < h1_; ++i) s += w[i] * h1v_[size_t(i)]; h2v_[size_t(j)] = gelu(s); }
        float o[3]; for (int j = 0; j < 3; ++j) { const float* w = &w3_[size_t(j) * size_t(h2_)]; float s = b3_[size_t(j)]; for (int i = 0; i < h2_; ++i) s += w[i] * h2v_[size_t(i)]; o[j] = s; }
        *dx = o[0]; *dy = o[1]; if (remaining_ms) *remaining_ms = double(o[2]) * 50.0; return true;
    }
 private:
    static float gelu(float v) { return 0.5f * v * (1.f + std::erf(v * 0.70710678f)); }
    int L_ = 0, h1_ = 0, h2_ = 0; std::vector<float> w1_, b1_, w2_, b2_, w3_, b3_; mutable std::vector<float> in_, h1v_, h2v_;     // scratch: one LandingNet per thread
};

}  // namespace eyelive

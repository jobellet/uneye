// Dynamic saliency from a stack of timestamped frames: PQFT (phase spectrum of the quaternion Fourier transform, Guo, Ma & Zhang, CVPR 2008).
// Quaternion image per frame: motion M = |I_t - I_{t-1}| (scaled so that 1 / 30 s between frames is the reference), red-green RG, blue-yellow BY, intensity I, on a size x size grid
// (default 64 x 64, a power of two). The quaternion is split into two complex images, f1 = M + i RG and f2 = BY + i I; their 2-D FFTs share a joint modulus, the amplitude is set to
// one (phase only), the inverse FFTs give the map |f1'|^2 + |f2'|^2, smoothed by a Gaussian. No training, no dependency, no allocation after construction; one frame costs four 64 x 64
// FFTs. The map is normalised to a maximum of 1 and can be turned into weighted candidates with eyelive::peaks_from_map (candidates.hpp).
#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

namespace eyelive {

class SaliencyPQFT {
 public:
    explicit SaliencyPQFT(int size = 64, double blur_sigma_px = 3.0, double center_sigma = 0.0)      // center_sigma > 0: Gaussian centre prior, sigma as a fraction of the map size (0.3 is typical), 0 = none
        : n_(size), sigma_(blur_sigma_px), csig_(center_sigma) {
        int p = 1; while (p < n_) p <<= 1; n_ = p;
        tw_.resize(size_t(n_ / 2)); for (int k = 0; k < n_ / 2; ++k) tw_[size_t(k)] = std::polar(1.0f, float(-2.0 * M_PI * k / n_));
        rev_.resize(size_t(n_)); int lg = 0; while ((1 << lg) < n_) ++lg; for (int i = 0; i < n_; ++i) { int r = 0; for (int b = 0; b < lg; ++b) if (i & (1 << b)) r |= 1 << (lg - 1 - b); rev_[size_t(i)] = r; }
        const size_t N = size_t(n_) * size_t(n_);
        f1_.assign(N, {}); f2_.assign(N, {}); prev_.assign(N, 0.f); cur_.assign(N, 0.f); rg_.assign(N, 0.f); by_.assign(N, 0.f); map_.assign(N, 0.f); tmp_.assign(N, 0.f); col_.assign(size_t(n_), {});
        const int r = std::max(1, int(std::ceil(3 * sigma_))); kern_.assign(size_t(2 * r + 1), 0.f); float s = 0; for (int i = -r; i <= r; ++i) { kern_[size_t(i + r)] = float(std::exp(-0.5 * i * i / (sigma_ * sigma_))); s += kern_[size_t(i + r)]; } for (auto& v : kern_) v /= s;
    }
    int size() const { return n_; }
    const std::vector<float>& map() const { return map_; }                    // size x size, row-major, row 0 = top of the frame, maximum 1 (0 before the second frame... see valid())
    bool valid() const { return frames_ >= 1; }
    // gray (channels = 1) or RGB8 (channels = 3, interleaved) frame of w x h pixels taken at t_s seconds. The first frame (or one more than 0.5 s after the previous) has no motion channel.
    void push(const uint8_t* px, int w, int h, int channels, double t_s) {
        if (!px || w < 1 || h < 1 || (channels != 1 && channels != 3)) return;
        const size_t N = size_t(n_) * size_t(n_);
        // area-average resampling to n x n
        for (int y = 0; y < n_; ++y) {
            const int y0 = int(int64_t(y) * h / n_), y1 = std::max(y0 + 1, int(int64_t(y + 1) * h / n_));
            for (int x = 0; x < n_; ++x) {
                const int x0 = int(int64_t(x) * w / n_), x1 = std::max(x0 + 1, int(int64_t(x + 1) * w / n_));
                double r = 0, g = 0, b = 0; int c = 0;
                for (int yy = y0; yy < y1; ++yy) for (int xx = x0; xx < x1; ++xx) { const uint8_t* p = px + (size_t(yy) * size_t(w) + size_t(xx)) * size_t(channels); if (channels == 1) { r += p[0]; g += p[0]; b += p[0]; } else { r += p[0]; g += p[1]; b += p[2]; } ++c; }
                const double inv = 1.0 / (255.0 * std::max(c, 1)); r *= inv; g *= inv; b *= inv; const size_t k = size_t(y) * size_t(n_) + size_t(x);
                cur_[k] = float((r + g + b) / 3.0); rg_[k] = float(r - g); by_[k] = float(b - 0.5 * (r + g));
            }
        }
        const double dt = t_s - t_prev_; const bool have_prev = frames_ > 0 && dt > 1e-6 && dt < 0.5; const float gain = have_prev ? float(std::min(1.0 / 30.0 / dt, 8.0)) : 0.f;   // rate normalisation
        for (size_t k = 0; k < N; ++k) {
            const float m = have_prev ? std::fabs(cur_[k] - prev_[k]) * gain : 0.f;
            f1_[k] = {m, rg_[k]}; f2_[k] = {by_[k], cur_[k]};
        }
        fft2(f1_, false); fft2(f2_, false);
        for (size_t k = 0; k < N; ++k) { const float a = std::abs(f1_[k]), b = std::abs(f2_[k]); const float m = std::sqrt(a * a + b * b) + 1e-9f; f1_[k] /= m; f2_[k] /= m; }
        fft2(f1_, true); fft2(f2_, true);
        for (size_t k = 0; k < N; ++k) tmp_[k] = std::norm(f1_[k]) + std::norm(f2_[k]);
        blur(tmp_, map_);
        if (csig_ > 0) { const double s = csig_ * n_; for (int y = 0; y < n_; ++y) for (int x = 0; x < n_; ++x) { const double dx = x + 0.5 - n_ / 2.0, dy = y + 0.5 - n_ / 2.0; map_[size_t(y) * size_t(n_) + size_t(x)] *= float(std::exp(-(dx * dx + dy * dy) / (2 * s * s))); } }
        float mx = 0; for (float v : map_) mx = std::max(mx, v); if (mx > 0) for (auto& v : map_) v /= mx;
        prev_ = cur_; t_prev_ = t_s; ++frames_;
    }

 private:
    using C = std::complex<float>;
    void fft1(C* a, bool inverse) {
        for (int i = 0; i < n_; ++i) { const int j = rev_[size_t(i)]; if (i < j) std::swap(a[i], a[j]); }
        for (int len = 2; len <= n_; len <<= 1) { const int step = n_ / len;
            for (int i = 0; i < n_; i += len) for (int k = 0; k < len / 2; ++k) { C w = tw_[size_t(k * step)]; if (inverse) w = std::conj(w); const C u = a[i + k], v = a[i + k + len / 2] * w; a[i + k] = u + v; a[i + k + len / 2] = u - v; } }
        if (inverse) for (int i = 0; i < n_; ++i) a[i] /= float(n_);
    }
    void fft2(std::vector<C>& a, bool inverse) {
        for (int y = 0; y < n_; ++y) fft1(&a[size_t(y) * size_t(n_)], inverse);
        for (int x = 0; x < n_; ++x) { for (int y = 0; y < n_; ++y) col_[size_t(y)] = a[size_t(y) * size_t(n_) + size_t(x)]; fft1(col_.data(), inverse); for (int y = 0; y < n_; ++y) a[size_t(y) * size_t(n_) + size_t(x)] = col_[size_t(y)]; }
    }
    void blur(const std::vector<float>& in, std::vector<float>& out) {            // separable, edges replicated; uses out as the second buffer through rg_ (scratch)
        const int r = int(kern_.size() / 2);
        for (int y = 0; y < n_; ++y) for (int x = 0; x < n_; ++x) { float s = 0; for (int k = -r; k <= r; ++k) s += kern_[size_t(k + r)] * in[size_t(y) * size_t(n_) + size_t(std::min(std::max(x + k, 0), n_ - 1))]; out[size_t(y) * size_t(n_) + size_t(x)] = s; }
        for (int y = 0; y < n_; ++y) for (int x = 0; x < n_; ++x) { float s = 0; for (int k = -r; k <= r; ++k) s += kern_[size_t(k + r)] * out[size_t(std::min(std::max(y + k, 0), n_ - 1)) * size_t(n_) + size_t(x)]; rg_[size_t(y) * size_t(n_) + size_t(x)] = s; }   // rg_ is free after the frame was read
        out = rg_;
    }
    int n_; double sigma_, csig_; std::vector<C> tw_; std::vector<int> rev_; std::vector<C> f1_, f2_, col_; std::vector<float> prev_, cur_, rg_, by_, map_, tmp_, kern_; double t_prev_ = -1; int64_t frames_ = 0;
};

}  // namespace eyelive

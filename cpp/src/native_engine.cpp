// Pure C++ U-Net forward pass (eval mode). Mirrors uneye/functions.py::UNet.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "uneye/engine.hpp"

namespace uneye {
namespace {

struct Conv { int cin, cout, k; std::vector<float> w, b; };      // w[o][i][k]
struct BN { int c; std::vector<float> g, b, m, v; };

std::vector<float> read_f(std::ifstream& f, size_t n) {
    std::vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    if (!f) throw std::runtime_error("native engine: weight file too short");
    return v;
}
Conv read_conv(std::ifstream& f, int cout, int cin, int k) {
    Conv c{cin, cout, k, {}, {}};
    c.w = read_f(f, size_t(cout) * cin * k);
    c.b = read_f(f, cout);
    return c;
}
BN read_bn(std::ifstream& f, int c) {
    BN n{c, {}, {}, {}, {}};
    n.g = read_f(f, c); n.b = read_f(f, c); n.m = read_f(f, c); n.v = read_f(f, c);
    return n;
}

// x[C][T] -> in place ReLU then BatchNorm (inference)
void relu_bn(std::vector<float>& x, int C, int T, const BN& bn) {
    for (int c = 0; c < C; ++c) {
        const float s = bn.g[c] / std::sqrt(bn.v[c] + 1e-5f);
        const float o = bn.b[c] - bn.m[c] * s;
        float* p = &x[size_t(c) * T];
        for (int t = 0; t < T; ++t) p[t] = std::max(p[t], 0.f) * s + o;
    }
}

// All work buffers are sized ONCE in the NativeEngine constructor. The helpers below only read and write inside them
// (no resize / assign / push_back), so infer() never touches the heap.

// 'same' conv1d, zero padded. in[Cin][T] -> out[Cout][T]; every output value is first set to the bias, so no zero-fill is needed
void conv1d(const std::vector<float>& in, std::vector<float>& out, int T, const Conv& c) {
    const int pd = (c.k - 1) / 2;
    for (int o = 0; o < c.cout; ++o) {
        float* po = &out[size_t(o) * T];
        std::fill(po, po + T, c.b[o]);
        for (int i = 0; i < c.cin; ++i) {
            const float* pi = &in[size_t(i) * T];
            const float* w = &c.w[(size_t(o) * c.cin + i) * c.k];
            for (int k = 0; k < c.k; ++k) {
                const int sh = k - pd;
                const int t0 = std::max(0, -sh), t1 = std::min(T, T - sh);
                const float wk = w[k];
                for (int t = t0; t < t1; ++t) po[t] += wk * pi[t + sh];
            }
        }
    }
}

void maxpool(const std::vector<float>& in, std::vector<float>& out, int C, int T, int mp) {
    const int To = T / mp;
    for (int c = 0; c < C; ++c)
        for (int t = 0; t < To; ++t) {
            float m = in[size_t(c) * T + t * mp];
            for (int j = 1; j < mp; ++j) m = std::max(m, in[size_t(c) * T + t * mp + j]);
            out[size_t(c) * To + t] = m;
        }
}

// ConvTranspose1d(kernel=stride=mp). w[i][o][j]. in[Cin][T] -> out[Cout][T*mp]
void upconv(const std::vector<float>& in, std::vector<float>& out, int T, const Conv& c) {
    const int mp = c.k, To = T * mp;
    for (int o = 0; o < c.cout; ++o) {
        float* po = &out[size_t(o) * To];
        std::fill(po, po + To, c.b[o]);
        for (int i = 0; i < c.cin; ++i) {
            const float* pi = &in[size_t(i) * T];
            const float* w = &c.w[(size_t(i) * c.cout + o) * mp];
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < mp; ++j) po[t * mp + j] += pi[t] * w[j];
        }
    }
}

// out must already have the size a.size() + b.size()
void concat(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) {
    std::copy(a.begin(), a.end(), out.begin());
    std::copy(b.begin(), b.end(), out.begin() + a.size());
}

class NativeEngine : public Engine {
public:
    NativeEngine(const std::string& path, int window) : T_(window) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + path);
        char magic[4];
        int32_t hdr[4];
        f.read(magic, 4);
        f.read(reinterpret_cast<char*>(hdr), sizeof hdr);
        if (!f || std::memcmp(magic, "UNEY", 4) != 0) throw std::runtime_error("bad model file " + path);
        classes_ = hdr[1]; ks_ = hdr[2]; mp_ = hdr[3];
        if (T_ % (mp_ * mp_) != 0) throw std::runtime_error("window must be a multiple of mp^2");
        c0_ = read_conv(f, 10, 2, ks_);  // file layout [10][1][ks][2] -> reorder to [10][2][ks]
        {
            std::vector<float> w(c0_.w.size());
            for (int o = 0; o < 10; ++o)
                for (int k = 0; k < ks_; ++k)
                    for (int c = 0; c < 2; ++c) w[(size_t(o) * 2 + c) * ks_ + k] = c0_.w[(size_t(o) * ks_ + k) * 2 + c];
            c0_.w = std::move(w);
        }
        // careful: read_conv read bias after weights, sizes identical (10*2*ks)
        b0_ = read_bn(f, 10);
        c1_ = read_conv(f, 20, 10, ks_); b1_ = read_bn(f, 20);
        c2_ = read_conv(f, 20, 20, ks_); b2_ = read_bn(f, 20);
        c3_ = read_conv(f, 20, 20, ks_); b3_ = read_bn(f, 20);
        u1_ = read_conv(f, 20, 20, mp_); bu1_ = read_bn(f, 20);
        c4_ = read_conv(f, 20, 40, ks_); b4_ = read_bn(f, 20);
        u2_ = read_conv(f, 20, 20, mp_); bu2_ = read_bn(f, 20);
        c5_ = read_conv(f, 20, 40, ks_); b5_ = read_bn(f, 20);
        c6_ = read_conv(f, 10, 20, ks_); b6_ = read_bn(f, 10);
        c7_ = read_conv(f, classes_, 10, 1);
        // Work buffers: sized here, once. T = window, T1 = T / mp, T2 = T1 / mp (T is a multiple of mp^2, checked above).
        const size_t T = size_t(T_), T1 = T / mp_, T2 = T1 / mp_;
        in_.assign(2 * T, 0.f);
        c0o_.assign(10 * T, 0.f); c1o_.assign(20 * T, 0.f);
        p1_.assign(20 * T1, 0.f); c2o_.assign(20 * T1, 0.f);
        p2_.assign(20 * T2, 0.f); c3o_.assign(20 * T2, 0.f);
        up1_.assign(20 * T1, 0.f); cat1_.assign(40 * T1, 0.f); c4o_.assign(20 * T1, 0.f);
        up2_.assign(20 * T, 0.f); cat2_.assign(40 * T, 0.f); c5o_.assign(20 * T, 0.f);
        c6o_.assign(10 * T, 0.f); out_.assign(size_t(classes_) * T, 0.f);
    }
    int window() const override { return T_; }
    int classes() const override { return classes_; }

    void infer(const float* dxy, float* prob) override {
        const int T = T_, T1 = T / mp_, T2 = T1 / mp_;
        // c0 : (ks x 2) conv on interleaved input, zero padded in time
        for (int t = 0; t < T; ++t) { in_[t] = dxy[2 * t]; in_[T + t] = dxy[2 * t + 1]; }
        conv1d(in_, c0o_, T, c0_); relu_bn(c0o_, 10, T, b0_);
        conv1d(c0o_, c1o_, T, c1_); relu_bn(c1o_, 20, T, b1_);
        maxpool(c1o_, p1_, 20, T, mp_);
        conv1d(p1_, c2o_, T1, c2_); relu_bn(c2o_, 20, T1, b2_);
        maxpool(c2o_, p2_, 20, T1, mp_);
        conv1d(p2_, c3o_, T2, c3_); relu_bn(c3o_, 20, T2, b3_);
        upconv(c3o_, up1_, T2, u1_); relu_bn(up1_, 20, T1, bu1_);
        concat(p1_, up1_, cat1_);
        conv1d(cat1_, c4o_, T1, c4_); relu_bn(c4o_, 20, T1, b4_);
        upconv(c4o_, up2_, T1, u2_); relu_bn(up2_, 20, T, bu2_);
        concat(c1o_, up2_, cat2_);
        conv1d(cat2_, c5o_, T, c5_); relu_bn(c5o_, 20, T, b5_);
        conv1d(c5o_, c6o_, T, c6_); relu_bn(c6o_, 10, T, b6_);
        conv1d(c6o_, out_, T, c7_);
        for (int t = 0; t < T; ++t) {  // softmax over classes
            float m = -1e30f;
            for (int c = 0; c < classes_; ++c) m = std::max(m, out_[size_t(c) * T + t]);
            float s = 0;
            for (int c = 0; c < classes_; ++c) s += (prob[size_t(c) * T + t] = std::exp(out_[size_t(c) * T + t] - m));
            for (int c = 0; c < classes_; ++c) prob[size_t(c) * T + t] /= s;
        }
    }

private:
    int T_, classes_ = 2, ks_ = 5, mp_ = 5;
    Conv c0_, c1_, c2_, c3_, u1_, c4_, u2_, c5_, c6_, c7_;
    BN b0_, b1_, b2_, b3_, bu1_, b4_, bu2_, b5_, b6_;
    std::vector<float> in_, c0o_, c1o_, p1_, c2o_, p2_, c3o_, up1_, cat1_, c4o_, up2_, cat2_, c5o_, c6o_, out_;
};

}  // namespace

std::unique_ptr<Engine> make_native_engine(const std::string& p, int w) {
    return std::make_unique<NativeEngine>(p, w);
}

}  // namespace uneye

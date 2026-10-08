#include "uneye/causal_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace uneye {
namespace {

std::vector<float> rd(std::ifstream& f, size_t n) {
    std::vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    if (!f) throw std::runtime_error("causal engine: weight file too short");
    return v;
}

struct BN { std::vector<float> scale, shift; };  // inference BatchNorm folded to y = x*scale + shift
BN rd_bn(std::ifstream& f, int c) {
    auto g = rd(f, c), b = rd(f, c), m = rd(f, c), v = rd(f, c);
    BN n; n.scale.resize(c); n.shift.resize(c);
    for (int i = 0; i < c; ++i) { n.scale[i] = g[i] / std::sqrt(v[i] + 1e-5f); n.shift[i] = b[i] - m[i] * n.scale[i]; }
    return n;
}

// Causal dilated conv with its own input history.
struct Layer {
    int cin, cout, k, d, len;      // len = (k-1)*d + 1 samples of history
    std::vector<float> w, b;       // w[o][i][k]
    BN bn;
    bool residual = false;
    std::vector<float> hist;       // [cin][len] circular
    int pos = 0;                   // index of the newest sample in the circular buffer

    void reset() { std::fill(hist.begin(), hist.end(), 0.f); pos = 0; }
    // in: cin values at time t -> out: cout values (ReLU, BN, optional residual)
    void run(const float* in, float* out) {
        pos = (pos + 1) % len;
        for (int i = 0; i < cin; ++i) hist[size_t(i) * len + pos] = in[i];
        for (int o = 0; o < cout; ++o) {
            float acc = b[o];
            for (int i = 0; i < cin; ++i) {
                const float* h = &hist[size_t(i) * len];
                const float* wk = &w[(size_t(o) * cin + i) * k];
                for (int j = 0; j < k; ++j) {           // kernel tap j looks (k-1-j)*d samples back
                    int idx = pos - (k - 1 - j) * d;
                    if (idx < 0) idx += len;
                    acc += wk[j] * h[idx];
                }
            }
            acc = std::max(acc, 0.f) * bn.scale[o] + bn.shift[o];
            out[o] = residual ? acc + in[o] : acc;
        }
    }
};

class Causal : public StepEngine {
public:
    explicit Causal(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + path);
        char magic[4]; int32_t h[8];
        f.read(magic, 4); f.read(reinterpret_cast<char*>(h), sizeof h);
        if (!f || std::memcmp(magic, "UNCZ", 4) != 0) throw std::runtime_error("not a causal model: " + path);
        classes_ = h[1]; const int C = h[2], ks = h[3], stem_ks = h[4], nd = h[5]; in_ch_ = h[6];
        std::vector<int32_t> dil(nd);
        f.read(reinterpret_cast<char*>(dil.data()), nd * sizeof(int32_t));
        layers_.resize(1 + nd);
        auto setup = [&](Layer& L, int cin, int cout, int k, int d, bool res) {
            L.cin = cin; L.cout = cout; L.k = k; L.d = d; L.len = (k - 1) * d + 1; L.residual = res;
            L.w = rd(f, size_t(cout) * cin * k); L.b = rd(f, cout); L.bn = rd_bn(f, cout);
            L.hist.assign(size_t(cin) * L.len, 0.f);
        };
        setup(layers_[0], in_ch_, C, stem_ks, 1, false);
        for (int i = 0; i < nd; ++i) setup(layers_[1 + i], C, C, ks, dil[i], true);
        hw_ = rd(f, size_t(classes_) * C); hb_ = rd(f, classes_);
        C_ = C; rf_ = stem_ks;
        for (int d : dil) rf_ += (ks - 1) * d;
        a_.assign(C, 0.f); b_.assign(C, 0.f);
    }
    int classes() const override { return classes_; }
    int receptive_field() const override { return rf_; }
    void reset() override { for (auto& l : layers_) l.reset(); }
    void step(float dx, float dy, float* prob) override {
        float in[3] = {dx, dy, std::sqrt(dx * dx + dy * dy + 1e-12f)};
        layers_[0].run(in, a_.data());
        for (size_t l = 1; l < layers_.size(); ++l) { layers_[l].run(a_.data(), b_.data()); std::swap(a_, b_); }
        float m = -1e30f;
        for (int c = 0; c < classes_; ++c) {
            float s = hb_[c];
            for (int i = 0; i < C_; ++i) s += hw_[size_t(c) * C_ + i] * a_[i];
            prob[c] = s; m = std::max(m, s);
        }
        float z = 0;
        for (int c = 0; c < classes_; ++c) z += (prob[c] = std::exp(prob[c] - m));
        for (int c = 0; c < classes_; ++c) prob[c] /= z;
    }
private:
    int classes_, in_ch_, C_, rf_;
    std::vector<Layer> layers_;
    std::vector<float> hw_, hb_, a_, b_;
};

}  // namespace

std::unique_ptr<StepEngine> make_causal_engine(const std::string& p) { return std::make_unique<Causal>(p); }

bool is_causal_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    char m[4] = {0, 0, 0, 0};
    f.read(m, 4);
    return f && std::memcmp(m, "UNCZ", 4) == 0;
}

}  // namespace uneye

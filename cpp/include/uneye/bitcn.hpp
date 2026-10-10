// Bidirectional TCN saccade detector with the two-channel input (vx, vy); native C++ (no ONNX Runtime). Offline / short-delay use: the network sees past AND future.
// features2 reproduces foundation/dino1d.py::feats2; BiTCN::forward reproduces foundation/bitcn.py::BiTCN (weights from foundation/export_bitcn.py).
#pragma once
#include <string>
#include <vector>

namespace uneye {

// pos: T x 2 interleaved (x, y), NaN = invalid. out: T x 2 interleaved arcsinh(v / sigma), 0 at invalid samples.
// v = 5-point moving difference (Engbert & Kliegl) * fs / 6 on the linearly interpolated trace; sigma = mean over x, y of the median-based std of the whole trace.
void features2(const float* pos, int T, double fs, float* out);

class BiTCN {
 public:
    explicit BiTCN(const std::string& bin_path);            // throws std::runtime_error
    int nin() const { return nin_; }
    int receptive_field() const;                              // in samples (one side is half of it)
    void forward(const float* feats, int T, float* logit) const;   // feats T x nin interleaved -> T logits (> 0 = saccade)
 private:
    struct Layer { int dil = 1, k = 3; std::vector<float> w, b, scale, shift; };
    int nin_ = 0, ch_ = 0;
    Layer stem_; std::vector<Layer> blocks_; std::vector<float> head_w_; float head_b_ = 0;
};

// raw positions (T x 2, NaN allowed) -> saccade mask (logit > 0)
std::vector<unsigned char> detect(const BiTCN& net, const float* pos, int T, double fs);

}  // namespace uneye

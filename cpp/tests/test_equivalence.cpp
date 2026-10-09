// Python (PyTorch float64) vs C++ (float32, streaming) equivalence of the causal TCN at the output, on 10 real sequences of set B and
// 5 stress sequences (zeros, large, tiny, noise, loud noise), 1000 samples each. Golden file: cpp/scripts/equivalence.py.
//
// Tolerance on |p_C++ - p_Py64|: 5e-5. Derivation (MacBook Air M1, clang 22.1.8, 205 sequences, cpp/scripts/equivalence.py):
//   measured max |C++ -O3 - Py64| = 6.0e-6 (Py32 vs Py64: 3.5e-6, i.e. float32 itself costs that much);
//   other builds of the same code (-O0, -ffp-contract=off) moved outputs by up to 3.3e-6.
//   5e-5 = about 8 x the measured maximum: tight enough to catch a real bug (a wrong weight or layer moves outputs by 1e-2 or more),
//   loose enough for another compiler / CPU. It is a measured-plus-margin bound, NOT a proof of bit-exactness.
// Labels at 0.5 must be identical except where the reference itself is within the tolerance of 0.5.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "uneye/causal_engine.hpp"

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : ".";
    constexpr double kTol = 5e-5;
    auto net = uneye::make_causal_engine(root + "/models/causal.bin");
    std::FILE* f = std::fopen((root + "/tests/golden_equivalence_causal.bin").c_str(), "rb");
    if (!f) { std::printf("FAIL: golden file missing\n"); return 1; }
    int32_t hdr[2];
    if (std::fread(hdr, sizeof(int32_t), 2, f) != 2) return 1;
    const size_t n = static_cast<size_t>(hdr[0]), T = static_cast<size_t>(hdr[1]);
    std::vector<float> in(n * T * 2); std::vector<double> ref(n * T);
    const bool ok_read = std::fread(in.data(), sizeof(float), in.size(), f) == in.size() && std::fread(ref.data(), sizeof(double), ref.size(), f) == ref.size();
    std::fclose(f);
    if (!ok_read) { std::printf("FAIL: short golden file\n"); return 1; }
    double max_err = 0; long label_diff = 0; float p[2];
    for (size_t s = 0; s < n; ++s) {
        net->reset();
        for (size_t t = 0; t < T; ++t) {
            net->step(in[(s * T + t) * 2], in[(s * T + t) * 2 + 1], p);
            const double r = ref[s * T + t], e = std::fabs(static_cast<double>(p[1]) - r);
            if (!(e <= max_err)) max_err = std::isnan(e) ? INFINITY : e;
            if ((p[1] >= 0.5f) != (r >= 0.5) && std::fabs(r - 0.5) > kTol) ++label_diff;
        }
    }
    std::printf("%zu sequences x %zu samples: max |C++ - PyTorch float64| = %.3g (tolerance %.0e), label differences: %ld\n", n, T, max_err, kTol, label_diff);
    const bool pass = max_err <= kTol && label_diff == 0;
    std::printf(pass ? "PASSED\n" : "FAILED\n");
    return pass ? 0 : 1;
}

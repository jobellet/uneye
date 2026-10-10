// C++ BiTCN (2 channels vx, vy) against PyTorch: features and logits of a trace with a blink and a dropout (golden_bitcn_2ch.txt, from foundation/export_bitcn.py).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "uneye/bitcn.hpp"

int main(int argc, char** argv) {
    std::string root = argc > 1 ? argv[1] : ".";
    std::ifstream f(root + "/tests/golden_bitcn_2ch.txt");
    if (!f) { std::printf("missing golden file\n"); return 1; }
    int T, C; f >> T >> C;
    auto rd = [&](size_t n) { std::vector<float> v(n); std::string s; for (auto& x : v) { f >> s; x = (float)std::strtod(s.c_str(), nullptr); } return v; };
    auto pos = rd(2 * (size_t)T); auto ref_f = rd((size_t)C * T); auto ref_l = rd(T);
    uneye::BiTCN net(root + "/models/bitcn_sup_bitcn_v2.bin");
    std::vector<float> feats((size_t)C * T), lg(T);
    uneye::features2(pos.data(), T, 1000.0, feats.data()); net.forward(feats.data(), T, lg.data());
    double mf = 0, ml = 0; int flips = 0;
    for (size_t i = 0; i < feats.size(); ++i) mf = std::fmax(mf, std::fabs(feats[i] - ref_f[i]));
    for (int t = 0; t < T; ++t) { ml = std::fmax(ml, std::fabs(lg[t] - ref_l[t])); flips += (lg[t] > 0) != (ref_l[t] > 0); }
    std::printf("receptive field %d samples; max|feature diff| = %.2e, max|logit diff| = %.2e, decision flips = %d\n", net.receptive_field(), mf, ml, flips);
    bool ok = mf < 1e-3 && ml < 1e-3 && flips == 0;
    std::printf("bitcn parity %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

// T09a: the C++ U-Net (double) against the PyTorch golden (cpp/scripts/export_unet_golden.py): output, loss, all parameter gradients and the batch-norm running statistics after one training-mode forward pass, within 1e-4.
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <algorithm>
#include <cmath>
#include "eyelive/train/unet.hpp"
using namespace eyelive::train;
int main(int, char** argv) {
    std::ifstream f(std::string(argv[1]) + "/app/train/tests/golden_unet_grads.txt"); if (!f) { std::printf("cannot open the golden file\n"); return 1; }
    std::map<std::string, std::vector<double>> g; int B = 0, T = 0, K = 0, ks = 0, mp = 0; std::string line;
    while (std::getline(f, line)) { std::istringstream is(line); std::string name; is >> name; if (name == "dims") { is >> B >> T >> K >> ks >> mp; continue; } size_t n; is >> n; auto& v = g[name]; v.resize(n); for (auto& x : v) is >> x; }
    UNet<double> net(K, ks, mp, 2);
    std::vector<std::vector<double>> st; for (int i = 0; i < 19; ++i) { for (int j = 0; j < 4; ++j) { auto it = g.find("state_" + std::to_string(i) + "_" + std::to_string(j)); if (it == g.end()) break; st.push_back(it->second); } }
    if (!net.load_state(st)) { std::printf("state layout mismatch\n"); return 1; }
    net.zero_grad(); auto& z = net.forward(g["x"].data(), B, T, true);
    std::vector<double> gz(z.size()); const double E = softmax_mcloss(z.data(), g["target"].data(), gz.data(), B, K, T) + 0.001 * net.l2();
    net.backward(gz.data()); net.add_l2_grad(0.001);
    int fails = 0; auto cmp = [&](const char* what, double a, double b) { const double e = std::fabs(a - b) / std::max(1.0, std::fabs(b)); if (e > 1e-4) { ++fails; std::printf("MISMATCH %s: %.8g vs golden %.8g\n", what, a, b); } };
    cmp("loss", E, g["loss"][0]);
    // output: softmax of the logits
    { double worst = 0; for (int b = 0; b < B; ++b) for (int t = 0; t < T; ++t) { double m = -1e300, s = 0; for (int k = 0; k < K; ++k) m = std::max(m, z[(size_t(b) * K + k) * T + t]);
        for (int k = 0; k < K; ++k) s += std::exp(z[(size_t(b) * K + k) * T + t] - m);
        for (int k = 0; k < K; ++k) worst = std::max(worst, std::fabs(std::exp(z[(size_t(b) * K + k) * T + t] - m) / s - g["out"][(size_t(b) * K + k) * T + t]));
      }
      cmp("max |softmax - golden out|", worst + 1.0, 1.0); }
    // parameter gradients: net.grads() lists 38 tensors (conv w, b, batch-norm gamma, beta, ... , c7 w, b), the golden has 2 per .bin group (19 groups) in the same order
    auto gr = net.grads(); size_t idx = 0; double worst = 0; int count = 0;
    for (int i = 0; i < 19; ++i) for (int j = 0; j < 2; ++j) {
        const auto& want = g["grad_" + std::to_string(i) + "_" + std::to_string(j)]; const auto& got = gr[idx++];
        if (want.size() != got.size()) { std::printf("gradient size mismatch at group %d tensor %d\n", i, j); return 1; }
        for (size_t q = 0; q < want.size(); ++q) { worst = std::max(worst, std::fabs(want[q] - got[q]) / std::max(1e-2, std::fabs(want[q]))); ++count; }
    }
    std::printf("loss %.8f (golden %.8f); %d gradient entries, worst relative difference %.2e\n", E, g["loss"][0], count, worst);
    if (worst > 1e-4) { std::printf("MISMATCH in gradients\n"); ++fails; }
    // batch-norm running statistics after the forward pass: state() has 6 tensors per block (conv w, b, gamma, beta, running mean, running var); golden group 2b+1 is the batch norm of block b
    auto stt = net.state(); double wst = 0;
    for (int b = 0; b < 9; ++b) for (int j = 0; j < 2; ++j) { const auto& want = g["after_" + std::to_string(2 * b + 1) + "_" + std::to_string(j)]; const auto& got = stt[6 * b + 4 + j];
        for (size_t q = 0; q < want.size(); ++q) wst = std::max(wst, std::fabs(want[q] - got[q]) / std::max(1e-2, std::fabs(want[q]))); }
    std::printf("batch-norm running statistics: worst relative difference %.2e\n", wst);
    if (wst > 1e-4) { std::printf("MISMATCH in running statistics\n"); ++fails; }
    std::printf(fails ? "FAILED\n" : "golden comparison passed\n"); return fails ? 1 : 0;
}

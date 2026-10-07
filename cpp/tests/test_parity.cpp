// Checks the C++ engines against outputs of the original PyTorch network.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

#include "uneye/engine.hpp"

static int check(const std::string& root, const std::string& name) {
    std::ifstream f(root + "/tests/golden_" + name + ".txt");
    if (!f) { std::printf("missing golden %s\n", name.c_str()); return 1; }
    int W, C; f >> W >> C;
    std::vector<float> in(2 * W), ref(C * W), out(C * W);
    for (auto& v : in) f >> v;
    for (auto& v : ref) f >> v;
    int bad = 0;
    auto cmp = [&](const char* tag, uneye::Engine& e) {
        e.infer(in.data(), out.data());
        double m = 0;
        for (size_t i = 0; i < out.size(); ++i) m = std::fmax(m, std::fabs(out[i] - ref[i]));
        std::printf("%-10s %-7s max|diff vs PyTorch| = %.2e %s\n", name.c_str(), tag, m, m < 1e-4 ? "OK" : "FAIL");
        if (m >= 1e-4) bad = 1;
    };
    auto nat = uneye::make_native_engine(root + "/models/" + name + ".bin", W);
    cmp("native", *nat);
    if (uneye::onnx_available()) {
        auto ox = uneye::make_onnx_engine(root + "/models/" + name + ".onnx", W, C);
        cmp("onnx", *ox);
    }
    return bad;
}

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : ".";
    int bad = 0;
    for (const char* n : {"combined", "dataset1", "andersson"}) bad |= check(root, n);
    return bad;
}

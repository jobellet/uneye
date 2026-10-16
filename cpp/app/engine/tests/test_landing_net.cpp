// LandingNet: parity with the PyTorch network (golden windows of GazeBase RAN exported by cpp/scripts/export_landing_mlp.py), cost per call, bad files.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "eyelive/landing_net.hpp"
using namespace eyelive;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : "."; std::string err; LandingNet net;
    for (const char* name : {"gazebase", "drone"}) {
    CHECK(net.load(root + "/models/landing_mlp_" + name + ".bin", &err), "load %s: %s", name, err.c_str()); CHECK(net.loaded() && net.window() == 64, "window %d", net.window());
    std::ifstream g(root + "/app/engine/tests/data/landing_mlp_" + name + "_golden.bin", std::ios::binary); CHECK(bool(g), "golden file missing"); uint32_t hdr[2] = {0, 0}; g.read(reinterpret_cast<char*>(hdr), sizeof hdr);
    const int n = int(hdr[0]), L = int(hdr[1]); CHECK(n > 0 && L == net.window(), "golden header %d %d", n, L); double worst = 0, scale = 0;
    for (int c = 0; c < n && g; ++c) {
        int32_t onset_rel = 0; std::vector<float> P(size_t(L + 1) * 2); float expect[3]; g.read(reinterpret_cast<char*>(&onset_rel), 4); g.read(reinterpret_cast<char*>(P.data()), std::streamsize(P.size() * 4)); g.read(reinterpret_cast<char*>(expect), 12);
        auto pos = [&](int64_t i, double* x, double* y) { if (i < 0 || i > L) return false; const float a = P[size_t(i) * 2], b = P[size_t(i) * 2 + 1]; if (std::isnan(a) || std::isnan(b)) return false; *x = a; *y = b; return true; };
        double dx = 0, dy = 0, rem = 0; CHECK(net.predict(pos, L, onset_rel, &dx, &dy, &rem), "case %d: no prediction", c);
        worst = std::max({worst, std::fabs(dx - expect[0]), std::fabs(dy - expect[1]), std::fabs(rem / 50.0 - expect[2])}); scale = std::max({scale, std::fabs(double(expect[0])), std::fabs(double(expect[1]))});
    }
    std::printf("landing net '%s': %d golden windows, largest difference with PyTorch %.2e deg (largest output %.2f deg)\n", name, n, worst, scale); CHECK(worst < 2e-3, "parity %.3g", worst); }
    // cost
    { const int L = net.window(); std::vector<float> P(size_t(L + 1) * 2); for (int i = 0; i <= L; ++i) { P[size_t(i) * 2] = 0.05f * float(i); P[size_t(i) * 2 + 1] = 0.02f * float(i); } auto pos = [&](int64_t i, double* x, double* y) { if (i < 0 || i > L) return false; *x = P[size_t(i) * 2]; *y = P[size_t(i) * 2 + 1]; return true; };
      const int R = 2000; double dx, dy; const auto t0 = std::chrono::steady_clock::now(); for (int r = 0; r < R; ++r) net.predict(pos, L, 40, &dx, &dy); const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / R;
      std::printf("landing net: %.0f us per prediction\n", us); CHECK(us < (std::getenv("CI") ? 20000 : 3000), "too slow: %.0f us", us); }
    // bad files
    { LandingNet bad; CHECK(!bad.load(root + "/models/does_not_exist.bin", &err), "missing file accepted"); CHECK(!bad.load(root + "/app/engine/tests/data/landing_mlp_drone_golden.bin", &err) && err.find("not a landing") != std::string::npos, "wrong file accepted (%s)", err.c_str()); }
    if (fails == 0) std::printf("landing net: OK\n"); return fails ? 1 : 0;
}

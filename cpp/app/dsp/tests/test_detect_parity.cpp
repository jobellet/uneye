// The C++ detection chain against the Python reference on real traces (golden_hmm_accel.txt from cpp/scripts/export_hmm.py): HMM events, then events refined by the acceleration-zero rule.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include "eyelive/dsp.hpp"
using namespace eyelive::dsp;

static bool load_hmm(const std::string& path, double fs, HmmParams& h) {
    std::ifstream f(path); std::string line; while (std::getline(f, line)) { if (line.empty() || line[0] == '#') continue; std::istringstream s(line); double a; HmmParams p;
        if (s >> a >> p.d >> p.mu0 >> p.mu1 >> p.sd0 >> p.sd1 >> p.p_on >> p.p_stay && std::fabs(a - fs) < 1e-6) { p.fs = fs; h = p; return true; } }
    return false;
}
static Events read_events(std::ifstream& f) { int n; f >> n; Events e(n); for (auto& p : e) f >> p.first >> p.second; return e; }
static bool same(const Events& a, const Events& b, int tol, int* worst) { if (a.size() != b.size()) return false; *worst = 0; for (size_t i = 0; i < a.size(); ++i) { *worst = std::max({*worst, std::abs(a[i].first - b[i].first), std::abs(a[i].second - b[i].second)}); } return *worst <= tol; }

int main(int argc, char** argv) {
    std::string root = argc > 1 ? argv[1] : "."; std::ifstream f(root + "/app/dsp/tests/golden_hmm_accel.txt"); if (!f) { std::printf("missing golden file\n"); return 1; }
    int n; f >> n; int bad = 0;
    for (int c = 0; c < n; ++c) {
        std::string name; double fs; int T; f >> name >> fs >> T; std::vector<double> x(T), y(T); Mask valid(T);
        for (int t = 0; t < T; ++t) { std::string a, b; f >> a >> b; x[t] = std::strtod(a.c_str(), nullptr); y[t] = std::strtod(b.c_str(), nullptr); valid[t] = std::isfinite(x[t]) && std::isfinite(y[t]); }
        Events ref_hmm = read_events(f), ref_first = read_events(f), ref_last = read_events(f);
        HmmParams h; if (!load_hmm(root + "/models/universal_hmm.txt", fs, h)) { std::printf("no HMM parameters for %g Hz\n", fs); return 1; }
        Kin k; score_trace(x, y, valid, fs, k); Mask lab = hmm_detect(k, h); Events ev = runs_of(lab);
        Refine rf; rf.offset_first = true; Events first = runs_of(refine_events(ev, k.fx, k.fy, valid, fs, rf)); rf.offset_first = false; Events last = runs_of(refine_events(ev, k.fx, k.fy, valid, fs, rf));
        int w1, w2, w3; bool ok1 = same(ev, ref_hmm, 0, &w1), ok2 = same(first, ref_first, 1, &w2), ok3 = same(last, ref_last, 1, &w3);
        std::printf("%-22s HMM events %zu/%zu %s (max shift %d) | refined 'first' %zu/%zu %s (%d) | refined 'last' %zu/%zu %s (%d)\n", name.c_str(), ev.size(), ref_hmm.size(), ok1 ? "OK" : "FAIL", w1, first.size(), ref_first.size(), ok2 ? "OK" : "FAIL", w2, last.size(), ref_last.size(), ok3 ? "OK" : "FAIL", w3);
        bad += !(ok1 && ok2 && ok3);
    }
    std::printf("detect parity %s\n", bad ? "FAIL" : "OK"); return bad ? 1 : 0;
}

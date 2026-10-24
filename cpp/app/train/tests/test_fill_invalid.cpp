// Invalid (NaN / tracker-loss) stretches of a training trace must be filled
// with noise BEFORE windowing: a blink labelled 2 on a NaN stretch would
// otherwise lose its label in make_dataset (prep() ignores non-finite
// positions), the network would never see a positive class-2 example, and
// 3-class training collapses (saccades come back as class 2).
#include <cmath>
#include <cstdio>
#include <random>
#include "eyelive/train/dataset.hpp"
using namespace eyelive;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// 6 s at 1 kHz: fixation noise, 3 saccades (50-sample ramps), one blink
// (NaN stretch, label 2) — the app's case: the user labelled a blink where
// the tracker was lost.
static TrainTrace app_like_trace() {
    const int N = 6000;
    TrainTrace tr; tr.rate_hz = 1000.0;
    tr.x.assign(N, 0.f); tr.y.assign(N, 0.f); tr.labels.assign(N, 0);
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 0.0015f);
    float cx = 0.f, cy = 0.f;
    for (int t = 0; t < N; ++t) {
        if (t >= 1000 && t < 1300) { tr.labels[t] = 2; tr.x[t] = NAN; tr.y[t] = NAN; continue; }
        if ((t >= 500 && t < 550) || (t >= 2500 && t < 2550) || (t >= 4500 && t < 4550)) {
            const float ph = float(t % 1000 - 500) / 50.f;
            const float env = std::sin(3.14159265f * ph);
            cx += 0.03f * env; cy += 0.012f * env;
            tr.labels[t] = 1;
        } else { cx += nd(rng); cy += nd(rng); }
        tr.x[t] = cx; tr.y[t] = cy;
    }
    return tr;
}

int main() {
    // 1. fill_invalid: only the non-finite samples change, labels stay possible
    {
        std::vector<float> v = {1.f, 2.f, NAN, NAN, NAN, 3.f, INFINITY, 4.f};
        std::vector<float> orig = v;
        std::mt19937_64 rng(7);
        train::fill_invalid(v, 0.01, rng);
        CHECK(std::isfinite(v[2]) && std::isfinite(v[3]) && std::isfinite(v[4]) && std::isfinite(v[6]), "non-finite samples not all filled");
        CHECK(v[0] == orig[0] && v[1] == orig[1] && v[5] == orig[5] && v[7] == orig[7], "valid samples were modified");
        CHECK(std::fabs(v[2] - 2.f) < 0.2 && std::fabs(v[6] - 3.f) < 0.2, "filled samples not centred on the neighbour value");
    }

    // 2. noise_std: robust to saccade outliers (MAD of the first differences)
    {
        std::mt19937 rng(3); std::normal_distribution<float> nd(0.f, 0.002f);
        std::vector<float> v(2000); float p = 0;
        for (size_t i = 0; i < v.size(); ++i) { p += nd(rng); v[i] = p; }
        for (size_t i = 0; i < 20; ++i) v[100 * i] += 0.5f;                     // 20 huge saccade steps
        const double s = train::noise_std(v);
        std::printf("noise_std on fixation noise 0.002 + 20 saccade steps: %.5f\n", s);
        CHECK(s > 0.0005 && s < 0.01, "noise_std %.5f outside the fixation noise range", s);
    }

    // 3. make_dataset keeps the blink labels of a NaN blink (the web-app bug)
    {
        TrainTrace tr = app_like_trace();
        TrainConfig cfg; cfg.n_classes = 3; cfg.val_fraction = 0.25; cfg.rotation_aug = false;
        cfg.noise_aug = false; cfg.gain_aug = false; cfg.seed = 1;
        train::Dataset ds = train::make_dataset({tr}, cfg);
        CHECK(ds.error.empty(), "make_dataset: %s", ds.error.c_str());
        size_t blink = 0, sacc = 0, fix = 0, ign = 0;
        for (uint8_t l : ds.train.lab) { if (l == 2) ++blink; else if (l == 1) ++sacc; else if (l == 0) ++fix; else ++ign; }
        for (uint8_t l : ds.val.lab) if (l == 2) ++blink;
        std::printf("make_dataset on a trace with a NaN blink: train+val labels fix=%zu sacc=%zu blink=%zu ignore=%zu\n", fix, sacc, blink, ign);
        CHECK(blink > 0, "blink labels lost: no class-2 sample survived make_dataset (the 3-class training collapse)");
        // the velocities of the filled blink windows must not be a flat zero block
        bool noisy = false;
        for (int w = 0; w < ds.train.n && !noisy; ++w)
            for (int t = 0; t < ds.train.W; ++t)
                if (ds.train.lab[size_t(w) * ds.train.W + t] == 2) {
                    const float dx = ds.train.vel[size_t(w) * 2 * ds.train.W + t];
                    if (std::fabs(dx) > 1e-4) { noisy = true; break; }
                }
        CHECK(noisy, "the blink windows read as a zero-velocity block (still indistinguishable from fixation)");
    }

    std::printf(fails ? "FAILED\n" : "all fill_invalid tests passed\n"); return fails ? 1 : 0;
}

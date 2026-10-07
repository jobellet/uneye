// Synthetic gaze generator with ground-truth saccade labels.
// Stand-in for a live eye tracker: yields one (x,y) sample at a time.
#pragma once
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace uneye {

struct SimConfig {
    double fs = 1000;
    double duration_s = 60;
    double rate_hz = 1.5;        // mean saccade rate (Poisson)
    double micro_frac = 0.6;     // fraction of events that are microsaccades (<1 deg)
    double noise_sd = 0.01;      // deg, measurement noise
    double drift_sd = 0.004;     // deg/sqrt(sample) random walk (fixational drift)
    double min_isi_ms = 80;
    uint32_t seed = 1;
};

struct SimTrace {
    std::vector<double> x, y;
    std::vector<int> label;       // 1 inside a saccade
    struct Ev { int64_t on, off; double amp; };
    std::vector<Ev> events;
};

// Main-sequence-like saccades: duration = 2.2*A + 21 ms, smooth (cosine) position profile.
inline SimTrace simulate(const SimConfig& c) {
    std::mt19937 g(c.seed);
    std::normal_distribution<double> N(0, 1);
    std::uniform_real_distribution<double> U(0, 1);
    const int64_t n = int64_t(c.duration_s * c.fs);
    SimTrace s;
    s.x.assign(n, 0); s.y.assign(n, 0); s.label.assign(n, 0);
    // place events
    int64_t t = int64_t(0.3 * c.fs);
    const double ms = c.fs / 1000.0;
    std::vector<double> step_x(n, 0), step_y(n, 0);
    while (true) {
        t += int64_t((-std::log(1 - U(g)) / c.rate_hz) * c.fs) + int64_t(c.min_isi_ms * ms);
        const bool micro = U(g) < c.micro_frac;
        const double amp = micro ? 0.1 + 0.9 * U(g) : 1.0 + 12.0 * std::pow(U(g), 2);
        const double dur_ms = 2.2 * amp + 21.0;
        const int64_t dur = std::max<int64_t>(3, int64_t(dur_ms * ms));
        if (t + dur + 50 >= n) break;
        const double th = 2 * M_PI * U(g);
        const double ax = amp * std::cos(th), ay = amp * std::sin(th);
        for (int64_t i = 0; i < dur; ++i) {  // cosine ease: derivative = velocity profile
            const double a = 0.5 * (1 - std::cos(M_PI * (i + 1) / dur)) - 0.5 * (1 - std::cos(M_PI * i / dur));
            step_x[t + i] += ax * a; step_y[t + i] += ay * a;
            s.label[t + i] = 1;
        }
        s.events.push_back({t, t + dur - 1, amp});
        t += dur;
    }
    double px = 0, py = 0, dx = 0, dy = 0;
    for (int64_t i = 0; i < n; ++i) {
        dx += c.drift_sd * N(g); dy += c.drift_sd * N(g);
        dx *= 0.98; dy *= 0.98;  // bounded drift
        px += step_x[i] + dx * 0.1; py += step_y[i] + dy * 0.1;
        s.x[i] = px + c.noise_sd * N(g);
        s.y[i] = py + c.noise_sd * N(g);
    }
    return s;
}

}  // namespace uneye

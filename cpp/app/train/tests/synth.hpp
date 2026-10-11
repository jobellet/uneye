// Synthetic eye traces for the trainer tests: fixation noise + slow drift, saccades with a minimum-jerk profile (duration 21 + 2.2 ms/deg), labels 1 during saccades (2 during blinks if requested)
#pragma once
#include <cmath>
#include <random>
#include <vector>
#include "eyelive/trainer.hpp"
inline eyelive::TrainTrace synth_trace(double seconds, double rate, uint64_t seed, bool blinks = false, double rot = 0.0) {
    std::mt19937_64 rng(seed); std::normal_distribution<double> nd(0, 1); std::uniform_real_distribution<double> ud(0, 1);
    const size_t n = size_t(seconds * rate); eyelive::TrainTrace tr; tr.rate_hz = rate; tr.x.assign(n, 0.f); tr.y.assign(n, 0.f); tr.labels.assign(n, 0);
    double px = 0, py = 0, dx = 0, dy = 0; size_t i = 0; std::vector<double> X(n), Y(n);
    while (i < n) {
        const size_t fix = size_t((0.25 + 0.9 * ud(rng)) * rate);
        for (size_t k = 0; k < fix && i < n; ++k, ++i) { dx += 2e-5 * nd(rng); dy += 2e-5 * nd(rng); dx *= 0.999; dy *= 0.999; px += dx; py += dy; X[i] = px + 0.01 * nd(rng); Y[i] = py + 0.01 * nd(rng); }
        if (i >= n) break;
        if (blinks && ud(rng) < 0.15) { const size_t bl = size_t(0.15 * rate); for (size_t k = 0; k < bl && i < n; ++k, ++i) { X[i] = px; Y[i] = py; tr.labels[i] = 2; } continue; }
        const double amp = 2 + 8 * ud(rng), th = 6.2831853 * ud(rng), dur = (21 + 2.2 * amp) * rate / 1000.0; const size_t len = size_t(dur);
        for (size_t k = 0; k < len && i < n; ++k, ++i) { const double u = double(k + 1) / len, s = u * u * u * (10 - 15 * u + 6 * u * u); X[i] = px + amp * std::cos(th) * s + 0.01 * nd(rng); Y[i] = py + amp * std::sin(th) * s + 0.01 * nd(rng); tr.labels[i] = 1; }
        px += amp * std::cos(th); py += amp * std::sin(th); if (std::fabs(px) > 12 || std::fabs(py) > 12) { px *= 0.3; py *= 0.3; }
    }
    for (size_t k = 0; k < n; ++k) { tr.x[k] = float(X[k] * std::cos(rot) + Y[k] * std::sin(rot)); tr.y[k] = float(-X[k] * std::sin(rot) + Y[k] * std::cos(rot)); }
    return tr;
}

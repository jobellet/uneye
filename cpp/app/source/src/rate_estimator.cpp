#include "eyelive/source.hpp"
#include <algorithm>
#include <cmath>

namespace eyelive {

static double get_median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    if (v.size() % 2 == 0) {
        return (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2.0;
    } else {
        return v[v.size() / 2];
    }
}

void RateEstimator::add(double t_seconds) {
    seen_++;
    if (!has_last_) {
        last_ = t_seconds;
        has_last_ = true;
        return;
    }
    double dt = t_seconds - last_;
    last_ = t_seconds;
    if (dt < 0) {
        gaps_++;
    } else {
        dts_.push_back(dt);
    }
}

bool RateEstimator::ready() const {
    if (seen_ >= 200) return true;
    double sum = 0;
    for (double dt : dts_) sum += dt;
    return sum >= 0.5;
}

double RateEstimator::rate_hz() const {
    double med = get_median(dts_);
    if (med <= 0) return 1000.0;
    return 1.0 / med;
}

double RateEstimator::standard_rate_hz() const {
    double rhz = rate_hz();
    double std_rates[] = {60.0, 120.0, 250.0, 300.0, 500.0, 1000.0, 1200.0, 2000.0};
    double best_rate = rhz;
    double min_diff = 1e9;
    for (double r : std_rates) {
        double diff = std::abs(rhz - r);
        if (diff < min_diff && diff <= 0.02 * r) {
            min_diff = diff;
            best_rate = r;
        }
    }
    return best_rate;
}

double RateEstimator::jitter() const {
    double med = get_median(dts_);
    if (med <= 0) return 0;
    std::vector<double> devs;
    devs.reserve(dts_.size());
    for (double dt : dts_) {
        devs.push_back(std::abs(dt - med));
    }
    double mad = get_median(devs);
    return mad / med;
}

uint64_t RateEstimator::gaps() const {
    double med = get_median(dts_);
    uint64_t count = gaps_;
    if (med > 0) {
        for (double dt : dts_) {
            if (dt > 3.0 * med) {
                count++;
            }
        }
    }
    return count;
}

}  // namespace eyelive

#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "eyelive/detector.hpp"

namespace eyelive {

class FastDetector : public IDetector {
 public:
    FastDetector();
    ~FastDetector() override = default;

    DetectorInfo info() const override;
    void reset(const SessionInfo& session) override;
    void set_scale(double deg_per_unit) override;
    bool set_option(const std::string& key, const std::string& value) override;

    void advance(const ITraceView& trace, int64_t to, std::vector<Event>& final_events, std::vector<Event>& provisional_events) override;
    void detect_all(const ITraceView& trace, std::vector<Event>& events) override;

 private:
    struct State {
        int64_t head = 0;
        int state = 0; // 0 = wait onset, 1 = wait final (over min_samples)
        int64_t onset_idx = -1;
        int on_count = 0;
        int off_count = 0;
        double peak_speed = 0.0;
        float start_x = 0;
        float start_y = 0;
        bool provisional_sent = false;
        double current_threshold = 0.0;
        int freeze_samples = 0; // post-saccadic refractory period where we don't update noise
    };

    double fs_ = 1000.0;
    double deg_per_unit_ = 1.0;
    bool needs_degrees_ = true;

    double lambda_ = 6.0;
    int min_samples_ = 3;
    double noise_window_s_ = 0.1;
    int noise_win_samples_ = 100;

    State st_;

    struct CircBuf {
        std::vector<double> data;
        int head = 0;
        int count = 0;
        int capacity = 0;
        void init(int cap) {
            capacity = cap;
            data.assign(cap, 0.0);
            head = 0;
            count = 0;
        }
        void push(double v) {
            data[head] = v;
            head = (head + 1) % capacity;
            if (count < capacity) count++;
        }
        double get(int i) const {
            return data[(head - count + i + capacity) % capacity];
        }
        double get_rev(int i) const {
            return data[(head - 1 - i + capacity) % capacity];
        }
        double get_oldest() const {
            return data[(head - count + capacity) % capacity];
        }
    };

    struct RollingMedian {
        std::vector<double> sorted;
        void init(int cap) {
            sorted.reserve(cap);
            sorted.clear();
        }
        void add(double v) {
            auto it = std::lower_bound(sorted.begin(), sorted.end(), v);
            sorted.insert(it, v);
        }
        void remove(double v) {
            auto it = std::lower_bound(sorted.begin(), sorted.end(), v);
            if (it != sorted.end() && *it == v) {
                sorted.erase(it);
            }
        }
        double median() const {
            if (sorted.empty()) return 0.0;
            size_t n = sorted.size();
            size_t h = n / 2;
            if (n % 2) return sorted[h];
            return 0.5 * (sorted[h - 1] + sorted[h]);
        }
    };

    CircBuf px_, py_;
    CircBuf vx_, vy_;
    CircBuf dvx_, dvy_;

    RollingMedian rm_vx_, rm_vy_;
    RollingMedian rm_dvx_, rm_dvx2_;
    RollingMedian rm_dvy_, rm_dvy2_;

    std::vector<int64_t> t_idx_;
    int t_head_ = 0;
    int t_count_ = 0;

    // Gating delay line (optimized to Circular Buffer to satisfy reviewer nitpick)
    struct VelSample {
        double vx = 0;
        double vy = 0;
    };
    std::vector<VelSample> delay_line_data_;
    int delay_line_head_ = 0;
    int delay_line_count_ = 0;
    int delay_capacity_ = 5;

    void process_sample(int64_t i, const ITraceView& trace, std::vector<Event>& final_events, std::vector<Event>& provisional_events);

    // helper to recompute noise
    void update_noise_window(double v_x, double v_y);
    double current_noise_ = 1e-5;
    double current_mx_ = 0;
    double current_my_ = 0;
};

}  // namespace eyelive

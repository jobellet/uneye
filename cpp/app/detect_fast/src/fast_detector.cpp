#include "eyelive/fast_detector.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace eyelive {

static DetectorRegistrar reg("fast_threshold", [](std::string*) { return std::make_unique<FastDetector>(); });

FastDetector::FastDetector() {
}

DetectorInfo FastDetector::info() const {
    DetectorInfo i;
    i.id = "fast_threshold";
    i.name = "Fast Threshold";
    i.n_classes = 2; // 0 = background, 1 = saccade
    i.causal = true;
    i.decision_delay_ms = 3.0;
    i.provisional_latency_ms = 3.0; // configurable based on lookahead
    i.needs_degrees = true;
    return i;
}

void FastDetector::reset(const SessionInfo& session) {
    fs_ = session.rate_hz;
    noise_win_samples_ = std::max(1, (int)std::nearbyint(noise_window_s_ * fs_));
    st_ = State();

    px_.init(5);
    py_.init(5);
    t_idx_.assign(5, 0);
    t_head_ = 0;
    t_count_ = 0;

    vx_.init(noise_win_samples_);
    vy_.init(noise_win_samples_);
    dvx_.init(noise_win_samples_);
    dvy_.init(noise_win_samples_);

    rm_vx_.init(noise_win_samples_);
    rm_vy_.init(noise_win_samples_);
    rm_dvx_.init(noise_win_samples_);
    rm_dvx2_.init(noise_win_samples_);
    rm_dvy_.init(noise_win_samples_);
    rm_dvy2_.init(noise_win_samples_);

    delay_capacity_ = 5;
    delay_line_data_.assign(delay_capacity_ + 1, VelSample());
    delay_line_head_ = 0;
    delay_line_count_ = 0;

    current_noise_ = 1e-5;
    current_mx_ = 0;
    current_my_ = 0;
}

void FastDetector::set_scale(double deg_per_unit) {
    deg_per_unit_ = deg_per_unit;
}

bool FastDetector::set_option(const std::string& key, const std::string& value) {
    try {
        if (key == "lambda") { lambda_ = std::stod(value); return true; }
        if (key == "min_samples") { min_samples_ = std::stoi(value); return true; }
        if (key == "noise_window_s") {
            noise_window_s_ = std::stod(value);
            noise_win_samples_ = std::max(1, (int)std::nearbyint(noise_window_s_ * fs_));
            return true;
        }
    } catch (...) {}
    return false;
}

void FastDetector::update_noise_window(double v_x, double v_y) {
    if (vx_.count == noise_win_samples_) {
        rm_vx_.remove(vx_.get_oldest());
        rm_vy_.remove(vy_.get_oldest());
    }
    vx_.push(v_x);
    vy_.push(v_y);
    rm_vx_.add(v_x);
    rm_vy_.add(v_y);

    current_mx_ = rm_vx_.median();
    current_my_ = rm_vy_.median();

    double dv_x = v_x - current_mx_;
    double dv_y = v_y - current_my_;

    if (dvx_.count == noise_win_samples_) {
        double old_dvx = dvx_.get_oldest();
        rm_dvx_.remove(old_dvx);
        rm_dvx2_.remove(old_dvx * old_dvx);

        double old_dvy = dvy_.get_oldest();
        rm_dvy_.remove(old_dvy);
        rm_dvy2_.remove(old_dvy * old_dvy);
    }
    dvx_.push(dv_x);
    dvy_.push(dv_y);
    rm_dvx_.add(dv_x);
    rm_dvx2_.add(dv_x * dv_x);
    rm_dvy_.add(dv_y);
    rm_dvy2_.add(dv_y * dv_y);

    double m1_x = rm_dvx_.median();
    double m2_x = rm_dvx2_.median();
    double noise_x = std::sqrt(std::max(m2_x - m1_x * m1_x, 1e-12));

    double m1_y = rm_dvy_.median();
    double m2_y = rm_dvy2_.median();
    double noise_y = std::sqrt(std::max(m2_y - m1_y * m1_y, 1e-12));

    current_noise_ = std::max(0.5 * (noise_x + noise_y), 1e-9);
    // Floor on sigma: 0.003 deg/ms = 3.0 deg/s equivalent
    current_noise_ = std::max(current_noise_, 3.0 / lambda_);
}

void FastDetector::process_sample(int64_t i, const ITraceView& trace, std::vector<Event>& final_events, std::vector<Event>& provisional_events) {
    double x = trace.x(i) * deg_per_unit_;
    double y = trace.y(i) * deg_per_unit_;
    bool v = trace.valid(i);

    if (!v && px_.count > 0) {
        x = px_.get_rev(0);
        y = py_.get_rev(0);
    }

    px_.push(x);
    py_.push(y);
    t_idx_[t_head_] = i;
    t_head_ = (t_head_ + 1) % 5;
    if (t_count_ < 5) t_count_++;

    if (t_count_ >= 5) {
        double px_m2 = px_.get(0);
        double px_m1 = px_.get(1);
        double px_p1 = px_.get(3);
        double px_p2 = px_.get(4);

        double py_m2 = py_.get(0);
        double py_m1 = py_.get(1);
        double py_p1 = py_.get(3);
        double py_p2 = py_.get(4);

        double v_x = (px_p2 + px_p1 - px_m1 - px_m2) * fs_ / 6.0;
        double v_y = (py_p2 + py_p1 - py_m1 - py_m2) * fs_ / 6.0;

        VelSample vs;
        vs.vx = v_x;
        vs.vy = v_y;
        delay_line_data_[delay_line_head_] = vs;
        delay_line_head_ = (delay_line_head_ + 1) % (delay_capacity_ + 1);
        if (delay_line_count_ <= delay_capacity_) delay_line_count_++;

        double dv_x = v_x - current_mx_;
        double dv_y = v_y - current_my_;
        double speed = std::hypot(dv_x, dv_y);

        double threshold = lambda_ * current_noise_;
        bool over_threshold = speed > threshold;

        int warmup_samples = (int)(0.05 * fs_);
        bool is_warm = (vx_.count >= warmup_samples);

        int idx_pos = (t_head_ - 3 + 5) % 5;
        int64_t current_idx = t_idx_[idx_pos];

        if (st_.freeze_samples > 0) {
            st_.freeze_samples--;
        }

        if (st_.state == 0) {
            if (over_threshold && is_warm) {
                st_.on_count++;
                if (st_.on_count == 1) {
                    st_.onset_idx = current_idx;
                    st_.peak_speed = speed;
                    st_.start_x = px_.get(2);
                    st_.start_y = py_.get(2);
                    st_.current_threshold = threshold;
                }
                st_.peak_speed = std::max(st_.peak_speed, speed);

                if (st_.on_count == min_samples_) {
                    st_.state = 1;
                    st_.provisional_sent = true;

                    Event ev;
                    ev.onset = st_.onset_idx;
                    ev.offset = current_idx;
                    ev.t_onset = trace.t(st_.onset_idx);
                    ev.t_offset = trace.t(current_idx);
                    ev.cls = 1;
                    ev.provisional = true;
                    ev.origin = EventOrigin::Detector;
                    ev.confidence = 1.0;
                    provisional_events.push_back(ev);
                }
            } else {
                st_.on_count = 0;
            }
        } else if (st_.state == 1) {
            st_.peak_speed = std::max(st_.peak_speed, speed);
            if (speed < 0.5 * st_.current_threshold) {
                st_.off_count++;
                if (st_.off_count >= 3) {
                    int64_t offset_idx = current_idx - 3;
                    if (offset_idx < st_.onset_idx) offset_idx = st_.onset_idx;

                    Event ev;
                    ev.onset = st_.onset_idx;
                    ev.offset = offset_idx;
                    ev.t_onset = trace.t(st_.onset_idx);
                    ev.t_offset = trace.t(offset_idx);
                    ev.cls = 1;
                    ev.provisional = false;
                    ev.origin = EventOrigin::Detector;
                    ev.confidence = 1.0;

                    double end_x = trace.x(offset_idx) * deg_per_unit_;
                    double end_y = trace.y(offset_idx) * deg_per_unit_;

                    ev.amplitude_deg = std::hypot(end_x - st_.start_x, end_y - st_.start_y);
                    ev.peak_speed_deg_s = st_.peak_speed;
                    ev.direction_rad = std::atan2(end_y - st_.start_y, end_x - st_.start_x);

                    final_events.push_back(ev);

                    st_.state = 0;
                    st_.on_count = 0;
                    st_.off_count = 0;
                    st_.provisional_sent = false;
                    st_.freeze_samples = (int)(0.02 * fs_);
                }
            } else {
                st_.off_count = 0;
            }
        }

        if (delay_line_count_ > delay_capacity_) {
            int old_idx = (delay_line_head_ - delay_line_count_ + (delay_capacity_ + 1)) % (delay_capacity_ + 1);
            VelSample old = delay_line_data_[old_idx];
            delay_line_count_--;

            // Allow warmup to ignore threshold check
            if (!is_warm || (st_.state == 0 && !over_threshold && st_.freeze_samples == 0)) {
                update_noise_window(old.vx, old.vy);
            }
        }
    }
}

void FastDetector::advance(const ITraceView& trace, int64_t to, std::vector<Event>& final_events, std::vector<Event>& provisional_events) {
    for (int64_t i = st_.head; i < to; ++i) {
        process_sample(i, trace, final_events, provisional_events);
    }
    st_.head = to;
}

void FastDetector::detect_all(const ITraceView& trace, std::vector<Event>& events) {
    reset(trace.info());
    std::vector<Event> prov;
    advance(trace, trace.size(), events, prov);

    if (st_.state == 1 && st_.provisional_sent) {
        Event ev;
        ev.onset = st_.onset_idx;
        ev.offset = trace.size() - 1;
        ev.t_onset = trace.t(st_.onset_idx);
        ev.t_offset = trace.t(ev.offset);
        ev.cls = 1;
        ev.provisional = false;
        ev.origin = EventOrigin::Detector;
        ev.confidence = 0.0;

        double end_x = trace.x(ev.offset) * deg_per_unit_;
        double end_y = trace.y(ev.offset) * deg_per_unit_;

        ev.amplitude_deg = std::hypot(end_x - st_.start_x, end_y - st_.start_y);
        ev.peak_speed_deg_s = st_.peak_speed;
        ev.direction_rad = std::atan2(end_y - st_.start_y, end_x - st_.start_x);

        events.push_back(ev);
    }
}

}  // namespace eyelive

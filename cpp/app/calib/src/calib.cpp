#include "eyelive/calib.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include "agreement.hpp"

namespace eyelive {

std::vector<Event> apply_calibration(const ITraceView& trace, const std::vector<Event>& detected, const CalibParams& p) {
    std::vector<Event> out;
    if (detected.empty()) return out;

    std::vector<Event> evs = detected;
    std::sort(evs.begin(), evs.end(), [](const Event& a, const Event& b) {
        return a.onset < b.onset;
    });

    std::vector<Event> merged;
    for (const auto& ev : evs) {
        if (!merged.empty() && ev.onset - merged.back().offset - 1 <= p.merge_gap) {
            merged.back().offset = std::max(merged.back().offset, ev.offset);
        } else {
            merged.push_back(ev);
        }
    }

    int64_t T = trace.size();

    for (const auto& m : merged) {
        int64_t s = std::max<int64_t>(0, m.onset);
        int64_t e = std::min<int64_t>(T - 1, m.offset);

        float amp = 0.0f;
        if (s <= e && trace.valid(s)) {
            float ox = trace.x(s);
            float oy = trace.y(s);
            for (int64_t i = s; i <= e; ++i) {
                if (trace.valid(i)) {
                    float dx = trace.x(i) - ox;
                    float dy = trace.y(i) - oy;
                    float dist = std::sqrt(dx * dx + dy * dy);
                    if (dist > amp) amp = dist;
                }
            }
        }

        if (amp < p.amp_min_deg) continue;

        int64_t s2 = std::max<int64_t>(m.onset - p.onset_shift, 0);
        int64_t e2 = std::min<int64_t>(m.offset + p.offset_shift, T - 1);

        if (e2 >= s2) {
            Event calib_ev = m;
            calib_ev.onset = s2;
            calib_ev.offset = e2;
            if (s2 < T) calib_ev.t_onset = trace.t(s2);
            if (e2 < T) calib_ev.t_offset = trace.t(e2);
            out.push_back(calib_ev);
        }
    }

    return out;
}

CalibParams fit_calibration(const ITraceView& trace, const std::vector<Event>& detected, const std::vector<Event>& expert, uint8_t cls, double* score_out) {
    std::vector<float> amps = {0.0f, 0.2f, 0.4f, 0.6f, 1.0f};
    std::vector<int> pres = {-8, -4, 0, 4, 8};
    std::vector<int> posts = {-24, -16, -8, -4, 0, 4, 8};
    std::vector<int> gaps = {0, 8, 20};

    CalibParams best_p;
    double best_obj = -9.0;

    for (float a : amps) {
        for (int pre : pres) {
            for (int post : posts) {
                for (int g : gaps) {
                    CalibParams p;
                    p.amp_min_deg = a;
                    p.onset_shift = pre;
                    p.offset_shift = post;
                    p.merge_gap = g;

                    std::vector<Event> calib_evs = apply_calibration(trace, detected, p);

                    AgreementStats s = agreement(calib_evs, expert, trace.size(), cls);
                    double obj = 0.5 * (s.event_f1 + (std::isnan(s.kappa) ? 0.0 : s.kappa));

                    if (obj > best_obj) {
                        best_obj = obj;
                        best_p = p;
                    }
                }
            }
        }
    }

    if (score_out) *score_out = best_obj;
    return best_p;
}

class ScaledTrace : public ITraceView {
 public:
    ScaledTrace(const ITraceView& base, double scale) : base_(base), scale_(scale) {
        info_ = base_.info();
        info_.deg_per_unit = base_.info().deg_per_unit * scale;
    }
    int64_t size() const override { return base_.size(); }
    double t(int64_t i) const override { return base_.t(i); }
    float x(int64_t i) const override { return base_.x(i) * scale_; }
    float y(int64_t i) const override { return base_.y(i) * scale_; }
    uint32_t flags(int64_t i) const override { return base_.flags(i); }
    uint8_t label(int64_t i) const override { return base_.label(i); }
    const SessionInfo& info() const override { return info_; }
 private:
    const ITraceView& base_;
    double scale_;
    SessionInfo info_;
};

ScaleChoice select_scale(const ITraceView& trace, IDetector& network, const std::vector<Event>& reference, const std::vector<double>& grid) {
    ScaleChoice choice;
    std::vector<double> cand = grid;
    if (cand.empty()) {
        double val = 0.005;
        while (val <= 4.0001) {
            cand.push_back(val);
            val *= 1.4;
        }
    }

    double best_obj = -1.0;
    double second_best_obj = -1.0;
    double best_c = 1.0;
    double second_best_c = 1.0;

    for (double c : cand) {
        ScaledTrace st(trace, c);

        network.reset(st.info());
        if (network.info().needs_degrees) {
            network.set_scale(c);
        }

        std::vector<Event> final_evs, prov_evs;
        network.detect_all(st, final_evs);

        AgreementStats s = agreement(final_evs, reference, trace.size(), 1);
        double obj = 0.5 * (s.event_f1 + (std::isnan(s.kappa) ? 0.0 : s.kappa));

        choice.curve.push_back({c, obj});

        if (obj > best_obj) {
            second_best_obj = best_obj;
            second_best_c = best_c;
            best_obj = obj;
            best_c = c;
        } else if (obj > second_best_obj) {
            second_best_obj = obj;
            second_best_c = c;
        }
    }

    choice.deg_per_unit = best_c;
    choice.agreement = best_obj;

    bool very_different = false;
    if (second_best_obj >= 0) {
        double ratio = best_c / second_best_c;
        if (ratio > 1.5 || ratio < 1.0/1.5) {
            if (best_obj - second_best_obj <= 0.02) {
                very_different = true;
            }
        }
    }

    if (best_obj < 0.4 || very_different) {
        choice.low_confidence = true;
    } else {
        choice.low_confidence = false;
    }

    return choice;
}

} // namespace eyelive

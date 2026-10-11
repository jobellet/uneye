#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#ifdef NDEBUG
#undef NDEBUG      // the test relies on assert(): keep it active in Release
#endif
#include <cassert>
#include <cmath>
#include <chrono>

#include "eyelive/calib.hpp"
#include "eyelive/trace.hpp"
#include "eyelive/types.hpp"
#include "../src/agreement.hpp"

using namespace eyelive;

// Helper to check within tolerance
bool near(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) <= tol;
}

// Simple test double detector for select_scale test
class TestDetector : public IDetector {
public:
    DetectorInfo info() const override {
        DetectorInfo di;
        di.id = "test_det";
        di.needs_degrees = true; // so it uses the scale
        return di;
    }

    void reset(const SessionInfo& session) override {}

    void set_scale(double deg_per_unit) override {
        scale_ = deg_per_unit;
    }

    void advance(const ITraceView& trace, int64_t to, std::vector<Event>& final_events, std::vector<Event>& provisional_events) override {}

    void detect_all(const ITraceView& trace, std::vector<Event>& events) override {
        events.clear();
        // create a perfect match only if scale is near 0.04
        if (near(scale_, 0.04, 0.01)) { // python grid has 0.0384
            // mock event: size 5 so it overlaps and is >=3
            Event ev;
            ev.onset = 0;
            ev.offset = 10;
            ev.cls = 1; // saccade
            events.push_back(ev);
        } else {
            // no events
        }
    }
private:
    double scale_ = 1.0;
};

void test_golden(const std::string& cpp_dir) {
    std::string path = cpp_dir + "/app/calib/tests/golden_calib.txt";
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Cannot open " << path << std::endl;
        exit(1);
    }

    std::string line;
    int test_idx = 0;
    while (std::getline(in, line)) {
        if (line.rfind("TEST", 0) == 0) {
            VectorTrace trace;
            std::vector<Event> det_evs;
            std::vector<Event> exp_evs;
            std::vector<Event> calib_evs;

            float apply_in_a = 0, apply_in_pre = 0, apply_in_post = 0, apply_in_gap = 0;
            double score_f1 = 0, score_kappa = 0;
            float fit_a = 0, fit_pre = 0, fit_post = 0, fit_gap = 0;
            double fit_score = 0;

            while (std::getline(in, line)) {
                if (line == "POS") {
                    for (int i = 0; i < 1000; ++i) {
                        float x, y;
                        in >> x >> y;
                        Sample s;
                        s.t = i / 1000.0;
                        s.x = x;
                        s.y = y;
                        trace.push(s);
                    }
                    std::getline(in, line); // consume newline
                } else if (line.rfind("DET", 0) == 0) {
                    int n;
                    std::stringstream ss(line);
                    std::string word;
                    ss >> word >> n;
                    for (int i = 0; i < n; ++i) {
                        int onset, offset;
                        in >> onset >> offset;
                        Event ev;
                        ev.onset = onset; ev.offset = offset; ev.cls = 1;
                        det_evs.push_back(ev);
                    }
                    std::getline(in, line); // consume newline
                } else if (line.rfind("EXP", 0) == 0) {
                    int n;
                    std::stringstream ss(line);
                    std::string word;
                    ss >> word >> n;
                    for (int i = 0; i < n; ++i) {
                        int onset, offset;
                        in >> onset >> offset;
                        Event ev;
                        ev.onset = onset; ev.offset = offset; ev.cls = 1;
                        exp_evs.push_back(ev);
                    }
                    std::getline(in, line); // consume newline
                } else if (line.rfind("APPLY_IN", 0) == 0) {
                    std::stringstream ss(line);
                    std::string word;
                    ss >> word >> apply_in_a >> apply_in_pre >> apply_in_post >> apply_in_gap;
                } else if (line.rfind("APPLY_OUT", 0) == 0) {
                    int n;
                    std::stringstream ss(line);
                    std::string word;
                    ss >> word >> n;
                    for (int i = 0; i < n; ++i) {
                        int onset, offset;
                        in >> onset >> offset;
                        Event ev;
                        ev.onset = onset; ev.offset = offset; ev.cls = 1;
                        calib_evs.push_back(ev);
                    }
                    std::getline(in, line); // consume newline
                } else if (line.rfind("SCORE", 0) == 0) {
                    std::stringstream ss(line);
                    std::string word;
                    ss >> word >> score_f1 >> score_kappa;
                } else if (line.rfind("FIT", 0) == 0) {
                    std::stringstream ss(line);
                    std::string word;
                    ss >> word >> fit_a >> fit_pre >> fit_post >> fit_gap >> fit_score;
                    break;
                }
            }

            // 1. apply_calibration
            CalibParams p;
            p.amp_min_deg = apply_in_a;
            p.onset_shift = apply_in_pre;
            p.offset_shift = apply_in_post;
            p.merge_gap = apply_in_gap;

            std::vector<Event> out_evs = apply_calibration(trace, det_evs, p);
            assert(out_evs.size() == calib_evs.size());
            for (size_t i = 0; i < out_evs.size(); ++i) {
                assert(out_evs[i].onset == calib_evs[i].onset);
                assert(out_evs[i].offset == calib_evs[i].offset);
            }

            // 2. test scoring
            AgreementStats stats = agreement(out_evs, exp_evs, trace.size(), 1);
            if (!near(stats.event_f1, score_f1)) {
                std::cerr << "F1 mismatch: expected " << score_f1 << ", got " << stats.event_f1 << std::endl;
                assert(false);
            }
            // Kappa can be nan in Python, which becomes 0.0 in Python script export
            double exp_k = score_kappa;
            double got_k = std::isnan(stats.kappa) ? 0.0 : stats.kappa;
            if (!near(got_k, exp_k)) {
                std::cerr << "Kappa mismatch: expected " << exp_k << ", got " << got_k << std::endl;
                assert(false);
            }

            // 3. test fit_calibration
            double out_score;
            CalibParams best_p = fit_calibration(trace, det_evs, exp_evs, 1, &out_score);
            assert(near(best_p.amp_min_deg, fit_a));
            assert(best_p.onset_shift == fit_pre);
            assert(best_p.offset_shift == fit_post);
            assert(best_p.merge_gap == fit_gap);
            assert(near(out_score, fit_score));

            test_idx++;
        }
    }
    assert(test_idx == 10);
    std::cout << "Golden test passed." << std::endl;
}

void test_select_scale() {
    VectorTrace trace;
    // mock trace of 100 samples
    for (int i = 0; i < 100; ++i) {
        Sample s;
        s.t = i; s.x = 0; s.y = 0;
        trace.push(s);
    }

    // reference event
    std::vector<Event> ref;
    Event ev;
    ev.onset = 0;
    ev.offset = 10;
    ev.cls = 1;
    ref.push_back(ev);

    TestDetector det;
    ScaleChoice sc = select_scale(trace, det, ref);

    // grid should have chosen a scale near 0.04
    // 0.005 * 1.4^k = 0.005, 0.007, 0.0098, 0.01372, 0.0192, 0.0268, 0.0376, 0.0527
    std::cout << "Chosen scale: " << sc.deg_per_unit << std::endl;
    assert(near(sc.deg_per_unit, 0.0376, 0.005));
    assert(!sc.low_confidence);

    std::cout << "Select scale test passed." << std::endl;
}

void test_fit_performance() {
    VectorTrace trace;
    for (int i = 0; i < 50000; ++i) { // 50 s at 1 kHz
        Sample s;
        s.t = i / 1000.0; s.x = i*0.001; s.y = i*0.001;
        trace.push(s);
    }

    std::vector<Event> det;
    std::vector<Event> exp;
    for (int i = 0; i < 30; ++i) {
        Event ev1, ev2;
        ev1.onset = i * 1000 + 100;
        ev1.offset = i * 1000 + 120;
        ev1.cls = 1;
        det.push_back(ev1);

        ev2.onset = i * 1000 + 105;
        ev2.offset = i * 1000 + 115;
        ev2.cls = 1;
        exp.push_back(ev2);
    }

    auto t0 = std::chrono::steady_clock::now();
    double sc = 0;
    fit_calibration(trace, det, exp, 1, &sc);
    auto t1 = std::chrono::steady_clock::now();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Fit calibration took " << sec << " seconds." << std::endl;
    assert(sec < 1.0);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: test_calib <cpp_dir>" << std::endl;
        return 1;
    }
    std::string cpp_dir = argv[1];
    test_golden(cpp_dir);
    test_select_scale();
    test_fit_performance();
    return 0;
}

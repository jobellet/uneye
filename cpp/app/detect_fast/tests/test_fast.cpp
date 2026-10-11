#include "eyelive/fast_detector.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <chrono>
#include <algorithm>

using namespace eyelive;

void generate_synthetic(double fs, double noise_rms, double amp, VectorTrace& trace, int& expected_onset, int& expected_offset) {
    SessionInfo info;
    info.rate_hz = fs;
    info.deg_per_unit = 1.0;
    trace = VectorTrace(info);

    int N = (int)(2.0 * fs);
    std::vector<double> x(N, 0.0), y(N, 0.0);

    for (int i = 0; i < N; ++i) {
        x[i] = noise_rms * ((double)rand() / RAND_MAX * 2 - 1) * sqrt(3.0);
        y[i] = noise_rms * ((double)rand() / RAND_MAX * 2 - 1) * sqrt(3.0);
    }

    // Smooth the noise
    for (int i = 1; i < N - 1; ++i) {
        x[i] = 0.25 * x[i-1] + 0.5 * x[i] + 0.25 * x[i+1];
        y[i] = 0.25 * y[i-1] + 0.5 * y[i] + 0.25 * y[i+1];
    }

    for (int i = 0; i < N; ++i) {
        double drift = 0.5 * noise_rms * std::sin(2 * M_PI * i / fs * 1.0);
        x[i] += drift;
        y[i] += drift;
    }

    int onset = (int)(1.0 * fs);
    double dur = 0.025 * std::sqrt(amp);
    int dur_samples = (int)(dur * fs);
    if (dur_samples < 5) dur_samples = 5;
    int offset = onset + dur_samples;

    for (int i = onset; i <= offset; ++i) {
        double t = (double)(i - onset) / dur_samples;
        double val = 0.5 - 0.5 * std::cos(M_PI * t);
        x[i] += amp * val;
    }
    for (int i = offset + 1; i < N; ++i) {
        x[i] += amp;
    }

    expected_onset = onset;
    expected_offset = offset;

    for (int i = 0; i < N; ++i) {
        Sample s;
        s.t = (double)i / fs;
        s.x = x[i];
        s.y = y[i];
        s.flags = kFlagValid;
        trace.push(s);
    }
}

int main(int argc, char** argv) {
    FastDetector det;
    SessionInfo info;
    info.rate_hz = 1000;
    VectorTrace trace(info);
    for (int i = 0; i < 10000; ++i) {
        Sample s;
        s.t = i / 1000.0;
        s.x = ((double)rand() / RAND_MAX) * 0.01;
        s.y = ((double)rand() / RAND_MAX) * 0.01;
        s.flags = kFlagValid;
        trace.push(s);
    }

    det.reset(info);
    std::vector<Event> fin, prov;
    auto t0 = std::chrono::high_resolution_clock::now();
    det.advance(trace, trace.size(), fin, prov);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / trace.size();
    std::cout << "Processing time per sample: " << us << " us" << std::endl;

    double fs_list[] = {500, 1000, 2000};

    std::cout << "| fs | noise | amp | recall | med latency | p95 latency |" << std::endl;
    std::cout << "|---|---|---|---|---|---|" << std::endl;

    for (double fs : fs_list) {
        for (double noise : {0.005, 0.01, 0.02, 0.05}) {
            for (double amp : {0.2, 0.3, 0.5, 1.0, 5.0, 15.0}) {
                std::vector<double> latencies;
                int misses = 0;
                int iters = 100;

                for (int iter = 0; iter < iters; ++iter) {
                    int expected_onset, expected_offset;
                    VectorTrace tr;
                    generate_synthetic(fs, noise, amp, tr, expected_onset, expected_offset);

                    det.reset(tr.info());
                    std::vector<Event> f_ev, p_ev;
                    det.advance(tr, tr.size(), f_ev, p_ev);

                    bool found = false;
                    for (auto& ev : f_ev) {
                        double err = (ev.onset - expected_onset) * 1000.0 / fs;
                        if (std::abs(err) < 100.0) { // within 100ms
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        misses++;
                    }

                    for (auto& ev : p_ev) {
                        double err = (ev.onset - expected_onset) * 1000.0 / fs;
                        if (err > -20 && err < 50) {
                            latencies.push_back(err);
                            break; // first one
                        }
                    }
                }

                double recall = 1.0 - (double)misses / iters;
                double med = -1, p95 = -1;
                if (!latencies.empty()) {
                    std::sort(latencies.begin(), latencies.end());
                    med = latencies[latencies.size() / 2];
                    p95 = latencies[(int)(latencies.size() * 0.95)];
                }

                if (fs == 1000 && noise <= 0.02 && amp >= 0.3) {
                    std::cout << "| " << fs << " | " << noise << " | " << amp << " | " << recall << " | " << med << " | " << p95 << " |" << std::endl;
                    if (recall < 0.95) {
                        std::cerr << "Recall failed! recall=" << recall << " < 0.95" << std::endl;
                        return 1;
                    }
                    if (med > 5.0) {
                        std::cerr << "Median latency failed! med=" << med << " > 5 ms" << std::endl;
                        return 1;
                    }
                    if (p95 > 8.0) {
                        std::cerr << "P95 latency failed! p95=" << p95 << " > 8 ms" << std::endl;
                        return 1;
                    }
                } else if (fs == 1000) {
                    std::cout << "| " << fs << " | " << noise << " | " << amp << " | " << recall << " | " << med << " | " << p95 << " |" << std::endl;
                }
            }
        }
    }

    // False alarms
    for (double noise : {0.005, 0.01, 0.02, 0.05}) {
        SessionInfo info;
        info.rate_hz = 1000;
        info.deg_per_unit = 1.0;
        VectorTrace tr(info);
        int N = 60000; // 60s
        std::vector<double> x(N), y(N);
        for (int i = 0; i < N; ++i) {
            x[i] = noise * ((double)rand() / RAND_MAX * 2 - 1) * sqrt(3.0);
            y[i] = noise * ((double)rand() / RAND_MAX * 2 - 1) * sqrt(3.0);
        }
        for (int i = 1; i < N - 1; ++i) {
            x[i] = 0.25 * x[i-1] + 0.5 * x[i] + 0.25 * x[i+1];
            y[i] = 0.25 * y[i-1] + 0.5 * y[i] + 0.25 * y[i+1];
        }
        for (int i = 0; i < N; ++i) {
            Sample s;
            s.t = (double)i / 1000.0;
            s.x = x[i];
            s.y = y[i];
            s.flags = kFlagValid;
            tr.push(s);
        }
        det.reset(tr.info());
        std::vector<Event> f_ev, p_ev;
        det.advance(tr, tr.size(), f_ev, p_ev);
        std::cout << "False alarms at noise " << noise << " : " << f_ev.size() << " per 60s" << std::endl;

        // Assert for acceptance tests
        if (noise <= 0.02 && f_ev.size() > 1) {
            std::cerr << "Too many false alarms! " << f_ev.size() << std::endl;
            return 1;
        }
    }

    // Retraction rule
    {
        SessionInfo info;
        info.rate_hz = 1000;
        info.deg_per_unit = 1.0;
        VectorTrace tr(info);
        for (int i = 0; i < 200; ++i) {
            Sample s;
            s.t = (double)i / 1000.0;
            s.x = 0; s.y = 0;
            s.flags = kFlagValid;
            tr.push(s);
        }
        for (int i = 200; i < 210; ++i) {
            Sample s;
            s.t = (double)i / 1000.0;
            s.x = (i - 200) * 0.1; s.y = 0;
            s.flags = kFlagValid;
            tr.push(s);
        }
        det.reset(tr.info());
        std::vector<Event> f_ev;
        det.detect_all(tr, f_ev);

        bool retracted = false;
        for (const auto& ev : f_ev) {
            if (!ev.provisional && ev.confidence == 0.0) retracted = true;
        }
        std::cout << "Retraction rule triggered: " << (retracted ? "yes" : "no") << std::endl;
        if (!retracted) return 1;
    }

    std::cout << "All tests passed." << std::endl;
    return 0;
}

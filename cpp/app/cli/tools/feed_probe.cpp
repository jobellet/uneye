// eyelive_feed_probe <mailbox file> [seconds] [--saliency]: opens the engine's stimulus feed on a mailbox written by a script (PsychoPy, Python, MATLAB) and prints what the engine receives.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include "eyelive/engine.hpp"
using namespace eyelive;
int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <mailbox file> [seconds] [--saliency]\n", argv[0]); return 2; }
    StimulusFeedConfig cfg; cfg.path = argv[1]; const double secs = argc > 2 ? std::atof(argv[2]) : 5.0; for (int i = 2; i < argc; ++i) if (!std::strcmp(argv[i], "--saliency")) cfg.saliency = true;
    auto e = make_engine(); std::string err; for (int k = 0; k < 400 && !e->stimulus_feed_open(cfg, &err); ++k) std::this_thread::sleep_for(std::chrono::milliseconds(50));   // wait for the script to create the file
    if (!e->stimulus_feed_status().open) { std::fprintf(stderr, "cannot open %s: %s\n", cfg.path.c_str(), err.c_str()); return 1; }
    const auto t0 = std::chrono::steady_clock::now(); uint64_t last_updates = ~0ull;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
        const auto st = e->stimulus_feed_status();
        if (st.candidate_updates != last_updates) { last_updates = st.candidate_updates; const auto c = e->candidates();
            std::printf("t=%.2fs frames=%llu updates=%llu candidates=%zu explicit=%d size=%dx%d", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), (unsigned long long)st.frames, (unsigned long long)st.candidate_updates, c.size(), int(st.explicit_candidates), st.width, st.height);
            for (size_t i = 0; i < c.size() && i < 3; ++i) std::printf("  (%.1f, %.1f, w %.2f)", double(c[i].x), double(c[i].y), double(c[i].weight)); std::printf("\n"); std::fflush(stdout); }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const auto st = e->stimulus_feed_status(); std::printf("final: frames=%llu last frame age %.3f s\n", (unsigned long long)st.frames, st.last_frame_age_s); return 0;
}

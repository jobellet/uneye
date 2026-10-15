// Candidate-target selector and saliency peaks (pure functions, no engine).
#include <cmath>
#include <cstdio>
#include "eyelive/candidates.hpp"
using namespace eyelive;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
static Candidate cand(float x, float y, float w = 1.f) { Candidate c; c.x = x; c.y = y; c.weight = w; return c; }
int main() {
    // eye left (0,0), now at (2,0) after 10 ms, tables say 8 deg are still to come: the target is at about (10, 0)
    { std::vector<Candidate> c = {cand(10, 0), cand(0, 10), cand(-10, 0), cand(10, 6)}; auto r = pick_candidate(0, 0, 2, 0, 8, c.data(), c.size(), 1.0);
      CHECK(r.index == 0, "aligned candidate not chosen (%d)", r.index); CHECK(r.confidence > 0.7, "confidence %.2f", r.confidence); }
    // two candidates on the same ray: the one at the predicted amplitude wins; with a long predicted amplitude the far one wins
    { std::vector<Candidate> c = {cand(6, 0), cand(25, 0)}; CHECK(pick_candidate(0, 0, 1, 0, 5, c.data(), 2, 1.0).index == 0, "near candidate expected");
      CHECK(pick_candidate(0, 0, 3, 0, 20, c.data(), 2, 1.0).index == 1, "far candidate expected"); }
    // a prior weight breaks a tie between two equal candidates
    { std::vector<Candidate> c = {cand(10, 0.1f, 1.f), cand(10, -0.1f, 3.f)}; CHECK(pick_candidate(0, 0, 2, 0, 8, c.data(), 2, 1.0).index == 1, "weight ignored"); }
    // units: a stream in 'pixels' (0.05 deg each) gives the same choice
    { std::vector<Candidate> c = {cand(200, 0), cand(0, 200)}; CHECK(pick_candidate(0, 0, 2, 0, 8, c.data(), 2, 0.05).index == 0, "unit conversion"); }
    // nothing to decide: no candidates, no movement yet; refusal below min_confidence
    { CHECK(pick_candidate(0, 0, 2, 0, 8, nullptr, 0, 1.0).index == -1, "empty list"); std::vector<Candidate> c = {cand(10, 0)};
      CHECK(pick_candidate(1, 1, 1, 1, 8, c.data(), 1, 1.0).index == -1, "no direction yet");
      CandidateParams p; p.min_confidence = 0.9; std::vector<Candidate> two = {cand(10, 0.3f), cand(10, -0.3f)}; CHECK(pick_candidate(0, 0, 2, 0, 8, two.data(), 2, 1.0, p).index == -1, "ambiguous choice should be refused"); }
    // saliency map 40 x 20 covering [-20, 20] x [-10, 10], two gaussian bumps (the second weaker) and a faint one under the threshold
    { const int W = 40, H = 20; static float m[W * H]; auto bump = [&](double cx, double cy, double a) { for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) m[y * W + x] += float(a * std::exp(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) / 8.0)); };
      bump(10, 5, 1.0); bump(30, 14, 0.6); bump(20, 10, 0.1);
      auto pk = peaks_from_map(m, W, H, -20, -10, 20, 10); CHECK(pk.size() == 2, "expected 2 peaks, got %zu", pk.size());
      if (pk.size() == 2) { CHECK(std::fabs(pk[0].x - (-20 + 10.5f)) < 1.01 && std::fabs(pk[0].y - (-10 + 5.5f * 1.0f)) < 1.01, "first peak at (%.1f, %.1f)", pk[0].x, pk[0].y); CHECK(pk[0].weight == 1.f && pk[1].weight > 0.5f && pk[1].weight < 0.7f, "weights %.2f %.2f", pk[0].weight, pk[1].weight); }
      CHECK(peaks_from_map(m, W, H, -20, -10, 20, 10, 1).size() == 1, "max_peaks ignored"); }
    if (fails == 0) std::printf("candidates: OK\n");
    return fails ? 1 : 0;
}

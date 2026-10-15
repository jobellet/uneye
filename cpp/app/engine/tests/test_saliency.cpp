// PQFT saliency: static pop-out, motion pop-out, timestamps, speed, and the link with the candidate peaks.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include "eyelive/candidates.hpp"
#include "eyelive/saliency.hpp"
using namespace eyelive;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
static void argmax(const std::vector<float>& m, int n, int& bx, int& by) { float b = -1; bx = by = 0; for (int y = 0; y < n; ++y) for (int x = 0; x < n; ++x) if (m[size_t(y) * size_t(n) + size_t(x)] > b) { b = m[size_t(y) * size_t(n) + size_t(x)]; bx = x; by = y; } }
// 320 x 240 RGB frame: mid-gray with a fixed pseudo-random grain (+-6), optional red square of side 32 at (sx, sy)
static std::vector<uint8_t> frame(int sx, int sy, bool square) {
    std::vector<uint8_t> f(320 * 240 * 3); for (int y = 0; y < 240; ++y) for (int x = 0; x < 320; ++x) { const uint8_t v = uint8_t(110 + int((uint32_t(x) * 2654435761u ^ uint32_t(y) * 40503u * 977u) >> 13) % 13 - 6); uint8_t* p = &f[(size_t(y) * 320 + size_t(x)) * 3]; p[0] = p[1] = p[2] = v; }
    if (square) for (int y = sy; y < sy + 32; ++y) for (int x = sx; x < sx + 32; ++x) { uint8_t* p = &f[(size_t(y) * 320 + size_t(x)) * 3]; p[0] = 230; p[1] = 40; p[2] = 40; }
    return f;
}
int main() {
    SaliencyPQFT s(64);
    // 1. static pop-out: a red square (side 32) centred at (246, 66) of the 320 x 240 frame -> map pixel (49, 18)
    { auto f = frame(230, 50, true); s.push(f.data(), 320, 240, 3, 0.0); int bx, by; argmax(s.map(), 64, bx, by);
      CHECK(std::abs(bx - 49) <= 5 && std::abs(by - 18) <= 5, "static peak at (%d, %d), expected near (49, 18)", bx, by); CHECK(s.valid(), "not valid"); }
    // 2. motion: grainy scene without a square, then the same scene with the square jumped in at (66, 176): second frame 33 ms later
    { SaliencyPQFT m(64); auto a = frame(0, 0, false); auto b = frame(50, 160, true); m.push(a.data(), 320, 240, 3, 1.0); m.push(b.data(), 320, 240, 3, 1.0 + 1.0 / 30);
      int bx, by; argmax(m.map(), 64, bx, by); CHECK(std::abs(bx - 15) <= 6 && std::abs(by - 45) <= 6, "motion peak at (%d, %d), expected near (15, 45)", bx, by); }
    // 3. a gap of more than 0.5 s resets the motion channel: the map equals the static one of the same frame
    { SaliencyPQFT a(64), b(64); auto f0 = frame(0, 0, false), f1 = frame(50, 160, true); a.push(f0.data(), 320, 240, 3, 0.0); a.push(f1.data(), 320, 240, 3, 5.0); b.push(f1.data(), 320, 240, 3, 0.0);
      double d = 0; for (size_t i = 0; i < a.map().size(); ++i) d = std::max(d, double(std::fabs(a.map()[i] - b.map()[i]))); CHECK(d < 1e-5, "gap not treated as a new stream (max diff %.2g)", d); }
    // 4. speed: 64 x 64 map from a 640 x 480 RGB frame
    { SaliencyPQFT q(64); std::vector<uint8_t> big(640 * 480 * 3, 128); for (size_t i = 0; i < big.size(); i += 7) big[i] = uint8_t(i * 31); const auto t0 = std::chrono::steady_clock::now(); const int R = 200;
      for (int r = 0; r < R; ++r) q.push(big.data(), 640, 480, 3, r / 30.0); const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / R;
      std::printf("saliency: %.0f us per 640x480 RGB frame (64x64 map)\n", us); CHECK(us < 5000, "too slow: %.0f us", us); }
    // 5. peaks -> candidates in gaze units (a screen of 40 x 30 deg centred on 0): the first candidate lies on the square
    { auto f = frame(230, 50, true); SaliencyPQFT t(64); t.push(f.data(), 320, 240, 3, 0.0); auto c = peaks_from_map(t.map().data(), 64, 64, -20, 15, 20, -15, 8, 0.4);
      CHECK(!c.empty(), "no candidate"); if (!c.empty()) { const double ex = -20 + 40.0 * 246 / 320, ey = 15 - 30.0 * 66 / 240; CHECK(std::hypot(c[0].x - ex, c[0].y - ey) < 5.0, "first candidate (%.1f, %.1f), expected near (%.1f, %.1f)", c[0].x, c[0].y, ex, ey); } }
    if (fails == 0) std::printf("saliency: OK\n");
    return fails ? 1 : 0;
}

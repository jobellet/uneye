// Candidate targets of the ongoing saccade: the application knows where the eye may go (buttons, words, saliency peaks), not which one. The selector compares, for every candidate,
// its direction with the chord of the saccade so far and its distance with the amplitude predicted by the fixed tables, and keeps the best. Header-only, no allocation, no thread.
// Parameters tuned on GazeBase RAN (train split), foundation/landing_candidates_heuristic.py; on the held-out participants the true target was chosen 92 / 80 / 65 % of the time
// with 2 / 4 / 8 candidates (distractors from the same screen distribution), landing error 2.4 / 3.0 / 3.7 deg against 5.2 deg for the tables alone.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include "eyelive/engine.hpp"

namespace eyelive {

struct CandidateParams {
    double angle_sigma0_rad = 0.12;   // angular tolerance at the very beginning (it shrinks as the eye moves: sigma = sigma0 / (1 + d / d0))
    double d0_deg = 0.5;
    double amplitude_sigma = 1.0;     // tolerance on ln(|candidate| / predicted amplitude)
    double min_confidence = 0.0;      // below this share of the total score the choice is refused (index -1); 0 = always choose
    double min_amplitude_deg = 0.3;
};
struct CandidateChoice { int index = -1; double confidence = 0; };

// positions in degrees. p0 = position before the onset, cur = current position, remaining = amplitude still to come predicted by the tables (>= 0).
inline CandidateChoice pick_candidate(double p0x, double p0y, double curx, double cury, double remaining, const Candidate* c, size_t n, double unit_to_deg, const CandidateParams& prm = CandidateParams()) {
    CandidateChoice out; if (n == 0) return out;
    const double dx = curx - p0x, dy = cury - p0y, d = std::hypot(dx, dy);
    if (d < 1e-9) return out;                                          // no direction yet
    const double ux = dx / d, uy = dy / d, A = std::max(d + std::max(remaining, 0.0), prm.min_amplitude_deg), lnA = std::log(A);
    const double sth = prm.angle_sigma0_rad / (1.0 + d / prm.d0_deg);
    double best = -1, total = 0; int bi = -1;
    for (size_t k = 0; k < n; ++k) {
        const double cx = c[k].x * unit_to_deg - p0x, cy = c[k].y * unit_to_deg - p0y, cn = std::max(std::hypot(cx, cy), 1e-6);
        const double th = std::acos(std::min(1.0, std::max(-1.0, (cx * ux + cy * uy) / cn))), la = std::log(cn) - lnA;
        const double w = c[k].weight > 0 ? c[k].weight : 1e-6;
        const double s = w * std::exp(-th * th / (2 * sth * sth)) * std::exp(-la * la / (2 * prm.amplitude_sigma * prm.amplitude_sigma)) + 1e-12;
        total += s; if (s > best) { best = s; bi = int(k); }
    }
    out.confidence = total > 0 ? best / total : 0;
    out.index = out.confidence >= prm.min_confidence ? bi : -1;
    return out;
}

// A saliency map (row-major, w x h, larger = more salient) covering the stream-unit rectangle [x0, x1] x [y0, y1] (y0 = value of row 0): its local maxima become weighted candidates.
// A pixel is a peak when it is the maximum of its (2r+1)^2 neighbourhood and above rel_threshold * the global maximum; at most max_peaks of the strongest are kept.
inline std::vector<Candidate> peaks_from_map(const float* map, int w, int h, double x0, double y0, double x1, double y1, int max_peaks = 32, double rel_threshold = 0.3, int radius = 2) {
    std::vector<Candidate> out; if (!map || w < 1 || h < 1) return out;
    float mx = 0; for (int i = 0; i < w * h; ++i) mx = std::max(mx, map[i]);
    if (!(mx > 0)) return out;
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        const float v = map[y * w + x]; if (v < rel_threshold * mx) continue;
        bool peak = true;
        for (int yy = std::max(0, y - radius); yy <= std::min(h - 1, y + radius) && peak; ++yy)
            for (int xx = std::max(0, x - radius); xx <= std::min(w - 1, x + radius); ++xx) { const float u = map[yy * w + xx]; if (u > v || (u == v && (yy < y || (yy == y && xx < x)))) { peak = false; break; } }
        if (!peak) continue;
        Candidate cd; cd.x = float(x0 + (x + 0.5) * (x1 - x0) / w); cd.y = float(y0 + (y + 0.5) * (y1 - y0) / h); cd.weight = v / mx; out.push_back(cd);
    }
    std::sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) { return a.weight > b.weight; });
    if (int(out.size()) > max_peaks) out.resize(size_t(max_peaks));
    return out;
}

}  // namespace eyelive

// Pure helpers of the Acquire tab (no ImGui): the preallocation summary shown while the user types, so that it can be unit-tested without a window.
#pragma once
#include <cmath>
#include <cstdio>
#include <string>
#include "eyelive/recorder.hpp"

namespace eyelive { namespace gui {

inline std::string sizing_text(double rate_hz, double duration_s, double margin = 1.05) {
    if (!(rate_hz > 0) || !(duration_s > 0)) return "enter a rate and a duration";
    const RecorderSizing s = recorder_sizing(rate_hz, duration_s, margin); char b[160]; const double mb = double(s.bytes) / 1e6;
    if (mb >= 1000) std::snprintf(b, sizeof b, "%llu samples, %.2f GB preallocated", (unsigned long long)s.capacity_samples, mb / 1000.0); else std::snprintf(b, sizeof b, "%llu samples, %.1f MB preallocated", (unsigned long long)s.capacity_samples, mb);
    return b;
}
// degrees per stream unit from the screen geometry: the horizontal size of one pixel seen from the eye (small-angle exact formula)
inline double deg_per_pixel_from_geometry(double width_cm, double distance_cm, double res_x_px) {
    if (!(width_cm > 0) || !(distance_cm > 0) || !(res_x_px > 0)) return 0.0; return 2.0 * std::atan(0.5 * width_cm / res_x_px / distance_cm) * 180.0 / 3.14159265358979323846;
}
}}  // namespace eyelive::gui

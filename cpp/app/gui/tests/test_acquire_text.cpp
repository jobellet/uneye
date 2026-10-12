// Acquire tab helpers without a window: sizing text for 1000 Hz x 3600 s, geometry conversion.
#include <cmath>
#include <cstdio>
#include "eyelive/gui/tab_acquire.hpp"
int main() {
    int fails = 0; auto check = [&](bool ok, const char* m) { if (!ok) { ++fails; std::printf("FAIL %s\n", m); } };
    const std::string t = eyelive::gui::sizing_text(1000, 3600); std::printf("1000 Hz x 3600 s: '%s'\n", t.c_str());
    check(t.find("3781024 samples") != std::string::npos && t.find("78.2 MB") != std::string::npos, "sizing text for 1000 Hz x 3600 s");
    check(eyelive::gui::sizing_text(0, 10).find("enter") != std::string::npos, "empty input"); check(eyelive::gui::sizing_text(2000, 36000).find("GB") != std::string::npos, "gigabytes");
    const double d = eyelive::gui::deg_per_pixel_from_geometry(53.0, 57.0, 1920.0); std::printf("53 cm wide, 57 cm away, 1920 px: %.5f deg per pixel (%.2f px per degree)\n", d, 1.0 / d); check(std::fabs(1.0 / d - 36.2) < 0.5, "geometry: about 36 px per degree for a 53 cm / 1920 px screen at 57 cm");
    check(eyelive::gui::deg_per_pixel_from_geometry(0, 57, 1920) == 0.0, "invalid geometry");
    std::printf(fails ? "FAILED\n" : "acquire helper tests passed\n"); return fails ? 1 : 0;
}

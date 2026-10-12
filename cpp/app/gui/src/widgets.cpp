#include "eyelive/gui/widgets.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace eyelive { namespace gui {

static inline double hyp(double a, double b) { return std::sqrt(a * a + b * b); }       // eye positions are small numbers: no need for std::hypot's overflow handling (5-10x faster)

ImU32 class_color(int cls, float a) {
    static const ImVec4 c[5] = {{0.55f, 0.55f, 0.58f, 1}, {0.20f, 0.55f, 0.95f, 1}, {0.95f, 0.60f, 0.15f, 1}, {0.25f, 0.75f, 0.40f, 1}, {0.70f, 0.40f, 0.85f, 1}};
    ImVec4 v = c[std::max(0, std::min(cls, 4))]; v.w = a; return ImGui::ColorConvertFloat4ToU32(v);
}
const char* class_name(int cls) { static const char* n[5] = {"fixation", "saccade", "blink", "class 3", "class 4"}; return n[std::max(0, std::min(cls, 4))]; }
std::string format_bytes(double b) { char s[32]; if (b >= 1e9) std::snprintf(s, sizeof s, "%.2f GB", b / 1e9); else if (b >= 1e6) std::snprintf(s, sizeof s, "%.1f MB", b / 1e6); else if (b >= 1e3) std::snprintf(s, sizeof s, "%.1f kB", b / 1e3); else std::snprintf(s, sizeof s, "%.0f B", b); return s; }
std::string format_time(double t) { char s[32]; if (t >= 3600) std::snprintf(s, sizeof s, "%d:%02d:%02d", int(t / 3600), int(std::fmod(t, 3600) / 60), int(std::fmod(t, 60))); else std::snprintf(s, sizeof s, "%02d:%04.1f", int(t / 60), std::fmod(t, 60)); return s; }

PlotHit draw_traces(const char* id, ImVec2 size, const ITraceView& tr, TimeView& view, const std::vector<Event>& events, int selected, const PlotStyle& style) {
    PlotHit hit; ImDrawList* dl = ImGui::GetWindowDrawList(); const ImVec2 p0 = ImGui::GetCursorScreenPos(); if (size.x < 50) size.x = ImGui::GetContentRegionAvail().x; const ImVec2 p1(p0.x + size.x, p0.y + size.y);
    ImGui::InvisibleButton(id, size, ImGuiButtonFlags_MouseButtonLeft); const bool hov = ImGui::IsItemHovered(); const double rate = tr.info().rate_hz > 0 ? tr.info().rate_hz : 1000.0; const int64_t n = tr.size();
    const float left = 44.0f, bottom = 18.0f; const ImVec2 q0(p0.x + left, p0.y + 4), q1(p1.x - 4, p1.y - bottom); hit.frame.p0 = q0; hit.frame.p1 = q1;
    // interaction: wheel zoom around the cursor, drag pan
    if (style.interactive && hov) {
        const ImGuiIO& io = ImGui::GetIO(); if (io.MouseWheel != 0 && q1.x > q0.x) { const double tc = view.t0 + double(io.MousePos.x - q0.x) / double(q1.x - q0.x) * (view.t1 - view.t0); const double f = std::pow(0.85, double(io.MouseWheel)); view.t0 = tc - (tc - view.t0) * f; view.t1 = tc + (view.t1 - tc) * f; }
        if (io.MouseWheelH != 0 && q1.x > q0.x) { const double dt = -double(io.MouseWheelH) * 0.08 * (view.t1 - view.t0); view.t0 += dt; view.t1 += dt; }
        if (style.pan && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 6.0f) && q1.x > q0.x) { const double dt = -double(io.MouseDelta.x) / double(q1.x - q0.x) * (view.t1 - view.t0); view.t0 += dt; view.t1 += dt; }
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) { const double span = view.t1 - view.t0; view.t1 = double(n) / rate; view.t0 = std::max(0.0, view.t1 - span); }
    }
    const double tmax = std::max(double(n) / rate, 1e-3); if (view.t1 - view.t0 < 0.02) view.t1 = view.t0 + 0.02; if (view.t0 < -0.5 * (view.t1 - view.t0)) { const double s = view.t1 - view.t0; view.t0 = -0.5 * s; view.t1 = view.t0 + s; }
    if (view.t0 > tmax) { const double s = view.t1 - view.t0; view.t0 = std::max(0.0, tmax - 0.5 * s); view.t1 = view.t0 + s; } hit.frame.t0 = view.t0; hit.frame.t1 = view.t1;
    dl->AddRectFilled(p0, p1, IM_COL32(24, 25, 29, 255)); const float split = style.show_speed ? (q1.y - q0.y) * (1.0f - style.speed_fraction) : (q1.y - q0.y); const ImVec2 a1(q1.x, q0.y + split - 2), s0(q0.x, q0.y + split + 2);
    const int64_t i0 = std::max<int64_t>(0, int64_t(std::floor(view.t0 * rate))), i1 = std::min<int64_t>(n, int64_t(std::ceil(view.t1 * rate)) + 1);
    // events as bands
    const size_t k0 = size_t(std::lower_bound(events.begin(), events.end(), i0 - 3000, [](const Event& e, int64_t bound) { return e.onset < bound; }) - events.begin());     // events are sorted by onset; none lasts 3 s
    for (size_t k = k0; k < events.size(); ++k) { const Event& e = events[k]; if (e.onset > i1) { if (!e.provisional) break; else continue; } if (e.offset < i0) continue; const float xa = std::max(q0.x, hit.frame.sx(double(e.onset) / rate)), xb = std::min(q1.x, std::max(hit.frame.sx(double(e.offset + 1) / rate), hit.frame.sx(double(e.onset) / rate) + 1.0f));
        dl->AddRectFilled(ImVec2(xa, q0.y), ImVec2(xb, q1.y), class_color(e.cls, e.provisional ? 0.34f : 0.22f)); if (e.provisional) dl->AddLine(ImVec2(xa, q0.y + 1), ImVec2(xb, q0.y + 1), class_color(e.cls), 3.0f); if (int(k) == selected) { dl->AddRect(ImVec2(xa, q0.y), ImVec2(xb, q1.y), IM_COL32(255, 255, 255, 220), 0, 0, 1.5f); } }
    // y range of the position traces over the visible samples
    float lo = 1e30f, hi = -1e30f; if (i1 > i0) { const int64_t step = std::max<int64_t>(1, (i1 - i0) / 4000); for (int64_t i = i0; i < i1; i += step) { const float x = tr.x(i), y = tr.y(i); if (std::isfinite(x)) { lo = std::min(lo, x); hi = std::max(hi, x); } if (std::isfinite(y)) { lo = std::min(lo, y); hi = std::max(hi, y); } } }
    if (!(hi > lo)) { lo = -1; hi = 1; } const float pad = 0.08f * (hi - lo) + 0.05f; lo -= pad; hi += pad; auto yy = [&](float v) { return a1.y - (v - lo) / (hi - lo) * (a1.y - q0.y); };
    // grid + labels
    for (int g = 0; g <= 4; ++g) { const float v = lo + (hi - lo) * float(g) / 4.0f; const float y = yy(v); dl->AddLine(ImVec2(q0.x, y), ImVec2(q1.x, y), IM_COL32(60, 62, 70, 255)); char b[16]; std::snprintf(b, sizeof b, "%.1f", v); dl->AddText(ImVec2(p0.x + 2, y - 7), IM_COL32(170, 170, 180, 255), b); }
    const double span = view.t1 - view.t0; double tick = 1e-3; for (double c : {1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0}) { tick = c; if (span / c < 10) break; }
    for (double t = std::ceil(view.t0 / tick) * tick; t < view.t1; t += tick) { const float x = hit.frame.sx(t); dl->AddLine(ImVec2(x, q0.y), ImVec2(x, q1.y), IM_COL32(45, 47, 54, 255)); char tb[32]; if (t >= 3600) std::snprintf(tb, sizeof tb, "%d:%02d:%02d", int(t / 3600), int(std::fmod(t, 3600) / 60), int(std::fmod(t, 60))); else std::snprintf(tb, sizeof tb, "%02d:%04.1f", int(t / 60), std::fmod(t, 60)); dl->AddText(ImVec2(x + 2, p1.y - 16), IM_COL32(170, 170, 180, 255), tb); }
    // traces: min / max per pixel column
    const int cols = std::max(1, int(q1.x - q0.x)); auto series = [&](int which, ImU32 col, const ImVec2& r0, const ImVec2& r1, float vlo, float vhi) {
        if (n <= 0) return; static thread_local std::vector<ImVec2> pts; pts.clear(); pts.reserve(size_t(cols) * 2); const double dt = (view.t1 - view.t0) / cols;
        for (int c = 0; c < cols; ++c) {                                    // column c shows the samples whose TIME falls in [t0 + c dt, t0 + (c + 1) dt): exact even when the view starts before 0 or ends after the data
            const int64_t a = std::max<int64_t>(0, int64_t(std::floor((view.t0 + c * dt) * rate))), b = std::min<int64_t>(n, std::max<int64_t>(a + 1, int64_t(std::floor((view.t0 + (c + 1) * dt) * rate))));
            if (a >= n || b <= a || (view.t0 + (c + 1) * dt) * rate < 0) continue; float mn = 1e30f, mx = -1e30f;
            for (int64_t i = a; i < b; ++i) { float v; if (which == 0) v = tr.x(i); else if (which == 1) v = tr.y(i); else { const int64_t j0 = std::max<int64_t>(0, i - 1), j1 = std::min<int64_t>(n - 1, i + 1); v = float(hyp(tr.x(j1) - tr.x(j0), tr.y(j1) - tr.y(j0)) * rate / double(std::max<int64_t>(1, j1 - j0))); } if (std::isfinite(v)) { mn = std::min(mn, v); mx = std::max(mx, v); } }
            if (mn > mx) continue; const float x = r0.x + float(c); const float ya = r1.y - (mn - vlo) / (vhi - vlo) * (r1.y - r0.y), yb = r1.y - (mx - vlo) / (vhi - vlo) * (r1.y - r0.y); pts.push_back(ImVec2(x, std::max(r0.y, std::min(r1.y, ya)))); pts.push_back(ImVec2(x, std::max(r0.y, std::min(r1.y, yb)))); }
        if (pts.size() > 1) dl->AddPolyline(pts.data(), int(pts.size()), col, 0, 1.2f);
    };
    dl->PushClipRect(ImVec2(q0.x, q0.y), ImVec2(q1.x, q1.y), true); series(0, IM_COL32(90, 180, 255, 255), ImVec2(q0.x, q0.y), a1, lo, hi); series(1, IM_COL32(255, 170, 70, 255), ImVec2(q0.x, q0.y), a1, lo, hi);
    if (style.show_speed) { float sm = 10.0f; if (i1 > i0) { const int64_t step = std::max<int64_t>(1, (i1 - i0) / 4000); for (int64_t i = i0 + 1; i + 1 < i1; i += step) sm = std::max(sm, float(hyp(tr.x(i + 1) - tr.x(i - 1), tr.y(i + 1) - tr.y(i - 1)) * rate / 2.0)); } sm = std::min(sm * 1.05f, 1500.0f);
        dl->AddLine(ImVec2(q0.x, s0.y), ImVec2(q1.x, s0.y), IM_COL32(60, 62, 70, 255)); series(2, IM_COL32(120, 220, 140, 255), s0, q1, 0.0f, sm); char b[24]; std::snprintf(b, sizeof b, "%.0f deg/s", sm); dl->AddText(ImVec2(q0.x + 4, s0.y), IM_COL32(120, 220, 140, 200), b); }
    if (style.forecast && style.forecast->f && !style.forecast->f->empty()) {         // forecast lines (magenta = x, yellow = y), time-shifted so that each point is drawn where the eye will be
        const ForecastOverlay& fo = *style.forecast; auto line = [&](bool use20, bool is_x, ImU32 col, float th) {
            static thread_local std::vector<ImVec2> pts; pts.clear(); const double lead = use20 ? 0.020 : 0.010; pts.reserve(size_t(cols));
            for (int c = 0; c < cols; ++c) { const double tc = view.t0 + (c + 1) * (view.t1 - view.t0) / cols; const int64_t i = int64_t(std::floor((tc - lead) * rate)) - fo.first; if (i < 0 || i >= int64_t(fo.f->size())) continue; const ForecastSample& f = (*fo.f)[size_t(i)]; if (!f.valid) continue;
                const float v = use20 ? (is_x ? f.x20 : f.y20) : (is_x ? f.x10 : f.y10); pts.push_back(ImVec2(q0.x + float(c), std::max(q0.y, std::min(a1.y, yy(v))))); }
            if (pts.size() > 1) dl->AddPolyline(pts.data(), int(pts.size()), col, 0, th); };
        if (fo.show10) { line(false, true, IM_COL32(255, 120, 255, 150), 1.0f); line(false, false, IM_COL32(255, 235, 120, 150), 1.0f); }
        if (fo.show20) { line(true, true, IM_COL32(255, 60, 255, 255), 2.0f); line(true, false, IM_COL32(255, 230, 50, 255), 2.0f); }
        if (fo.band && fo.show20 && n > 0) { const int64_t i = n - 1 - fo.first; if (i >= 0 && i < int64_t(fo.f->size()) && (*fo.f)[size_t(i)].valid) { const ForecastSample& f = (*fo.f)[size_t(i)]; const float xe = hit.frame.sx(double(n - 1) / rate + 0.020);        // the newest forecast with its 80 % interval
            for (int ax = 0; ax < 2; ++ax) { const float v = ax == 0 ? f.x20 : f.y20, sc = 1.61f * (ax == 0 ? f.sx20 : f.sy20); dl->AddLine(ImVec2(xe, yy(v + sc)), ImVec2(xe, yy(v - sc)), ax == 0 ? IM_COL32(255, 60, 255, 255) : IM_COL32(255, 230, 50, 255), 2.0f); dl->AddCircleFilled(ImVec2(xe, yy(v)), 3.5f, ax == 0 ? IM_COL32(255, 60, 255, 255) : IM_COL32(255, 230, 50, 255)); } } }
    }
    dl->PopClipRect(); dl->AddText(ImVec2(q1.x - 70, q0.y + 2), IM_COL32(90, 180, 255, 255), "x"); dl->AddText(ImVec2(q1.x - 55, q0.y + 2), IM_COL32(255, 170, 70, 255), "y"); if (style.forecast && style.forecast->f && style.forecast->show20) { dl->AddText(ImVec2(q1.x - 190, q0.y + 2), IM_COL32(255, 60, 255, 255), "x+20ms"); dl->AddText(ImVec2(q1.x - 125, q0.y + 2), IM_COL32(255, 230, 50, 255), "y+20ms"); }
    // cursor
    if (hov) { const ImVec2 m = ImGui::GetIO().MousePos; if (m.x >= q0.x && m.x <= q1.x) { hit.hovered = true; hit.t = hit.frame.tx(m.x); hit.sample = std::max<int64_t>(0, std::min<int64_t>(n - 1, int64_t(hit.t * rate))); dl->AddLine(ImVec2(m.x, q0.y), ImVec2(m.x, q1.y), IM_COL32(255, 255, 255, 90)); hit.clicked = ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !ImGui::IsMouseDragging(ImGuiMouseButton_Left, 6.0f); } }
    return hit;
}

void draw_gaze_map(const char* id, ImVec2 size, const ITraceView& tr, int64_t i0, int64_t i1, const std::vector<Event>& events) {
    ImDrawList* dl = ImGui::GetWindowDrawList(); const ImVec2 p0 = ImGui::GetCursorScreenPos(); ImGui::InvisibleButton(id, size); const ImVec2 p1(p0.x + size.x, p0.y + size.y); dl->AddRectFilled(p0, p1, IM_COL32(24, 25, 29, 255)); dl->AddRect(p0, p1, IM_COL32(70, 72, 80, 255));
    i0 = std::max<int64_t>(0, i0); i1 = std::min<int64_t>(tr.size(), i1); if (i1 <= i0) return; float xl = 1e30f, xh = -1e30f, yl = 1e30f, yh = -1e30f; const int64_t step = std::max<int64_t>(1, (i1 - i0) / 6000);
    for (int64_t i = i0; i < i1; i += step) { const float x = tr.x(i), y = tr.y(i); if (!std::isfinite(x) || !std::isfinite(y)) continue; xl = std::min(xl, x); xh = std::max(xh, x); yl = std::min(yl, y); yh = std::max(yh, y); }
    if (!(xh > xl)) { xl -= 1; xh += 1; } if (!(yh > yl)) { yl -= 1; yh += 1; } const float half = 0.55f * std::max(xh - xl, yh - yl), cx = 0.5f * (xl + xh), cy = 0.5f * (yl + yh); const float s = 0.5f * std::min(size.x, size.y) / half;
    auto P = [&](float x, float y) { return ImVec2(p0.x + 0.5f * size.x + (x - cx) * s, p0.y + 0.5f * size.y - (y - cy) * s); };
    dl->PushClipRect(p0, p1, true); static thread_local std::vector<ImVec2> run; run.clear();
    for (int64_t i = i0; i < i1; i += step) { const float x = tr.x(i), y = tr.y(i); if (!std::isfinite(x) || !std::isfinite(y)) { if (run.size() > 1) dl->AddPolyline(run.data(), int(run.size()), IM_COL32(120, 150, 190, 140), 0, 1.0f); run.clear(); continue; } run.push_back(P(x, y)); }
    if (run.size() > 1) dl->AddPolyline(run.data(), int(run.size()), IM_COL32(120, 150, 190, 140), 0, 1.0f);
    const size_t k0 = size_t(std::lower_bound(events.begin(), events.end(), i0 - 3000, [](const Event& e, int64_t bound) { return e.onset < bound; }) - events.begin());
    for (size_t k = k0; k < events.size(); ++k) { const Event& e = events[k]; if (e.onset >= i1) { if (!e.provisional) break; continue; } if (e.offset < i0) continue; run.clear();
        for (int64_t i = std::max(e.onset, i0); i <= std::min<int64_t>(e.offset, i1 - 1); ++i) { const float x = tr.x(i), y = tr.y(i); if (std::isfinite(x) && std::isfinite(y)) run.push_back(P(x, y)); } if (run.size() > 1) dl->AddPolyline(run.data(), int(run.size()), class_color(e.cls), 0, 2.2f); }
    if (i1 > i0) { const ImVec2 q = P(tr.x(i1 - 1), tr.y(i1 - 1)); dl->AddCircleFilled(q, 4.0f, IM_COL32(255, 255, 255, 230)); } dl->PopClipRect();
}

void draw_curves(const char* id, ImVec2 size, const std::vector<std::vector<float>>& series, const std::vector<ImU32>& colors, const std::vector<std::string>& names) {
    ImDrawList* dl = ImGui::GetWindowDrawList(); const ImVec2 p0 = ImGui::GetCursorScreenPos(); ImGui::InvisibleButton(id, size); const ImVec2 p1(p0.x + size.x, p0.y + size.y); dl->AddRectFilled(p0, p1, IM_COL32(24, 25, 29, 255)); dl->AddRect(p0, p1, IM_COL32(70, 72, 80, 255));
    float hi = 1e-9f, lo = 1e30f; size_t n = 0; for (const auto& s : series) { n = std::max(n, s.size()); for (float v : s) { hi = std::max(hi, v); lo = std::min(lo, v); } } if (n < 2) { dl->AddText(ImVec2(p0.x + 8, p0.y + 8), IM_COL32(150, 150, 160, 255), "no data yet"); return; } lo = std::min(lo, 0.0f);
    for (size_t k = 0; k < series.size(); ++k) { std::vector<ImVec2> pts; for (size_t i = 0; i < series[k].size(); ++i) pts.push_back(ImVec2(p0.x + 6 + (size.x - 12) * float(i) / float(n - 1), p1.y - 6 - (series[k][i] - lo) / (hi - lo) * (size.y - 12))); if (pts.size() > 1) dl->AddPolyline(pts.data(), int(pts.size()), colors[k], 0, 1.8f); dl->AddText(ImVec2(p0.x + 8 + 110 * float(k), p0.y + 4), colors[k], k < names.size() ? names[k].c_str() : ""); }
}

}}  // namespace eyelive::gui

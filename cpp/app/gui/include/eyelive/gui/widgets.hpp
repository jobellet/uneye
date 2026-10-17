// Drawing helpers shared by the tabs (plain ImGui draw lists, no plotting library): the gaze trace plot with event shading, the gaze map, colours, formatting.
#pragma once
#include <algorithm>
#include <string>
#include <vector>
#include "imgui.h"
#include "eyelive/engine.hpp"
#include "eyelive/trace.hpp"
#include "eyelive/types.hpp"

namespace eyelive { namespace gui {

ImU32 class_color(int cls, float alpha = 1.0f);                 // 0 fixation (grey), 1 saccade (blue), 2 blink (orange), 3 and 4 (green, purple)
const char* class_name(int cls);
bool input_string(const char* label, std::string& value, size_t max_len = 512);   // InputText on a std::string; true when the text changed
void help_marker(const char* text);                              // a (?) that shows the text on hover
std::string format_bytes(double bytes);
std::string format_time(double seconds);                         // mm:ss.s or h:mm:ss

// The events of the session for a viewer: the finals already received are kept, only the new ones and the (few) provisional ones are fetched at each frame. Sorted by onset (the plot searches it).
class EventCache {
 public:
    const std::vector<Event>& get(IEngine& e) {
        const uint64_t nf = e.stats().events_final; if (nf < nf_ || (nf == 0 && !ev_.empty())) { ev_.clear(); nf_ = 0; }          // a new session starts again from zero
        ev_.resize(nf_); tail_.clear(); e.events_from(nf_, tail_); bool unsorted = false;
        for (const auto& x : tail_) { if (!x.provisional && nfinal_in_tail_ok(x)) { if (!ev_.empty() && x.onset < ev_.back().onset) unsorted = true; ev_.push_back(x); ++nf_; } }
        const size_t nfin = ev_.size(); for (const auto& x : tail_) if (x.provisional) ev_.push_back(x);
        if (unsorted) std::sort(ev_.begin(), ev_.begin() + long(nfin), [](const Event& a, const Event& b) { return a.onset < b.onset; });
        return ev_;
    }
 private:
    static bool nfinal_in_tail_ok(const Event&) { return true; }
    std::vector<Event> ev_, tail_; uint64_t nf_ = 0;
};

struct TimeView { double t0 = 0, t1 = 5; };                      // visible window of the plot, seconds

struct PlotFrame {                                               // mapping between screen and data of the last drawn plot
    ImVec2 p0, p1; double t0 = 0, t1 = 1;
    float sx(double t) const { return float(p0.x + (t - t0) / (t1 - t0) * (p1.x - p0.x)); }
    double tx(float x) const { return t0 + double(x - p0.x) / double(p1.x - p0.x) * (t1 - t0); }
};
struct PlotHit { bool hovered = false; double t = 0; int64_t sample = -1; bool clicked = false; PlotFrame frame; };

struct ForecastOverlay {                                  // the forecast line: the forecast made at sample i for i + 20 ms is drawn at time t_i + 20 ms, so it can be compared with the measurement at that time and it extends 20 ms beyond the last measurement
    const std::vector<ForecastSample>* f = nullptr; int64_t first = 0;   // f[k] is the forecast of sample first + k
    bool show20 = true, show10 = false, band = true;
};
struct PlotStyle { const ForecastOverlay* forecast = nullptr; bool show_speed = true; bool interactive = true; bool pan = true; float speed_fraction = 0.3f; };

// x and y position traces (degrees) with the events as shaded bands, optionally the speed below. `selected` highlights one event of `events` (-1 = none). Wheel = zoom around the cursor, drag = pan (when interactive).
PlotHit draw_traces(const char* id, ImVec2 size, const ITraceView& tr, TimeView& view, const std::vector<Event>& events, int selected, const PlotStyle& style = PlotStyle());
// x against y for samples [i0, i1), the events highlighted
void draw_gaze_map(const char* id, ImVec2 size, const ITraceView& tr, int64_t i0, int64_t i1, const std::vector<Event>& events);
// a small line plot of one or two series against their index (loss curves); series are drawn with their own colours and shared y range
void draw_curves(const char* id, ImVec2 size, const std::vector<std::vector<float>>& series, const std::vector<ImU32>& colors, const std::vector<std::string>& names);

}}  // namespace eyelive::gui

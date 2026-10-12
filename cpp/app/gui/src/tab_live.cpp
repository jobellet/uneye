// Live tab (order 20): scrolling position and speed traces with the events as bands, the gaze map, the latest events and the live statistics.
#include <algorithm>
#include <cmath>
#include "imgui.h"
#include "eyelive/gui/widgets.hpp"
#include "eyelive/gui_tab.hpp"

namespace eyelive {
namespace {
class LiveTab final : public ITab {
 public:
    const char* title() const override { return "Live"; }
    int order() const override { return 20; }
    bool wants_redraw() const override { return live_; }
    void draw(IEngine& engine) override {
        const ITraceView& tr = engine.trace(); const EngineStats st = engine.stats(); live_ = st.running; const double rate = tr.info().rate_hz > 0 ? tr.info().rate_hz : 1000.0; const int64_t n = tr.size();
        ImGui::Checkbox("follow", &follow_); ImGui::SameLine(); ImGui::SetNextItemWidth(160); ImGui::SliderFloat("window (s)", &span_, 0.2f, 60.0f, "%.1f", ImGuiSliderFlags_Logarithmic); ImGui::SameLine(); ImGui::Checkbox("speed", &show_speed_); if (engine.forecast_available()) { ImGui::SameLine(); ImGui::Checkbox("forecast +20 ms", &show_fc_); } ImGui::SameLine(); ImGui::SetNextItemWidth(120); ImGui::SliderInt("gaze map (ms)", &map_ms_, 50, 3000);
        ImGui::SameLine(); ImGui::TextDisabled("wheel = zoom, drag = pan, double click = jump to the end");
        if (follow_ && n > 0) { view_.t1 = double(n) / rate; view_.t0 = view_.t1 - double(span_); } else span_ = float(view_.t1 - view_.t0);
        const std::vector<Event>& events = ev_cache_.get(engine); const float avail_h = ImGui::GetContentRegionAvail().y; const float right_w = 330.0f; const float plot_w = std::max(200.0f, ImGui::GetContentRegionAvail().x - right_w - 12);
        ImGui::BeginGroup(); gui::PlotStyle ps; ps.show_speed = show_speed_; gui::ForecastOverlay fo; if (show_fc_ && engine.forecast_available()) { engine.forecast_enable(true); const int64_t f0 = std::max<int64_t>(0, int64_t(std::floor(view_.t0 * rate)) - 30), f1 = std::min<int64_t>(n, int64_t(std::ceil(view_.t1 * rate)) + 2); if (f1 > f0) { engine.forecasts(f0, f1, fc_); fo.f = &fc_; fo.first = f0; ps.forecast = &fo; } } gui::PlotHit h = gui::draw_traces("live_plot", ImVec2(plot_w, std::max(160.0f, avail_h - 6)), tr, view_, events, -1, ps); if (h.hovered && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 3.0f)) follow_ = false; ImGui::EndGroup();
        ImGui::SameLine(); ImGui::BeginGroup(); const int64_t i1 = n, i0 = std::max<int64_t>(0, n - int64_t(map_ms_ * rate / 1000.0)); ImGui::Text("gaze (last %d ms)", map_ms_); gui::draw_gaze_map("gaze_map", ImVec2(right_w, right_w), tr, i0, i1, events);
        ImGui::Spacing(); ImGui::Text("events: %llu final, %llu provisional", (unsigned long long)st.events_final, (unsigned long long)st.events_provisional); ImGui::Text("latency p50 %.1f ms, p95 %.1f ms", st.detect_latency_ms_p50, st.detect_latency_ms_p95);
        if (ImGui::BeginTable("ev", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp, ImVec2(right_w, std::max(100.0f, ImGui::GetContentRegionAvail().y - 4)))) { ImGui::TableSetupColumn("t (s)"); ImGui::TableSetupColumn("class"); ImGui::TableSetupColumn("amp (deg)"); ImGui::TableSetupColumn("ms"); ImGui::TableHeadersRow();
            for (size_t k = events.size(); k-- > 0 && events.size() - k <= 100;) { const Event& e = events[k]; ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::Text("%.3f", e.t_onset); ImGui::TableNextColumn(); ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(gui::class_color(e.cls)), "%s%s", gui::class_name(e.cls), e.provisional ? "*" : ""); ImGui::TableNextColumn(); ImGui::Text("%.2f", double(e.amplitude_deg)); ImGui::TableNextColumn(); ImGui::Text("%.0f", (e.t_offset - e.t_onset) * 1000.0); }
            ImGui::EndTable(); }
        ImGui::EndGroup();
    }
 private:
    gui::EventCache ev_cache_; std::vector<ForecastSample> fc_; bool show_fc_ = false; gui::TimeView view_{0, 5}; bool follow_ = true, show_speed_ = true, live_ = false; float span_ = 5.0f; int map_ms_ = 500;
};
static TabRegistrar reg([] { return std::make_unique<LiveTab>(); });
}  // namespace
}  // namespace eyelive

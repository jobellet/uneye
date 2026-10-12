// Log tab (order 80): engine warnings, statistics, trigger timing.
#include <algorithm>
#include <cmath>
#include "imgui.h"
#include "eyelive/gui/widgets.hpp"
#include "eyelive/gui_tab.hpp"
namespace eyelive {
namespace {
class LogTab final : public ITab {
 public:
    const char* title() const override { return "Log"; }
    int order() const override { return 80; }
    bool wants_redraw() const override { return true; }
    void draw(IEngine& engine) override {
        const EngineStats s = engine.stats(); ImGui::SeparatorText("Engine");
        ImGui::Text("running: %s", s.running ? "yes" : "no"); ImGui::Text("sampling rate: measured %.2f Hz, standard %.0f Hz", s.rate_hz_measured, s.rate_hz_standard); ImGui::Text("samples: %llu (%s of %s), file %s", (unsigned long long)s.samples, gui::format_time(s.recorded_s).c_str(), gui::format_time(s.capacity_s).c_str(), gui::format_bytes(double(s.bytes_allocated)).c_str());
        ImGui::Text("dropped packets: %llu, gaps: %llu", (unsigned long long)s.dropped_packets, (unsigned long long)s.gaps); ImGui::Text("events: %llu final, %llu provisional", (unsigned long long)s.events_final, (unsigned long long)s.events_provisional);
        ImGui::Text("detection latency: p50 %.2f ms, p95 %.2f ms; publish jitter p95 %.2f ms; cpu %.1f %%", s.detect_latency_ms_p50, s.detect_latency_ms_p95, s.publish_jitter_ms_p95, s.cpu_percent);
        ImGui::Text("unit scale: %.5g deg/unit (%s)%s", s.deg_per_unit, s.scale_estimated ? "estimated" : "given", s.scale_low_confidence ? ", LOW CONFIDENCE" : "");
        if (!s.warning.empty()) ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1), "warning: %s", s.warning.c_str()); else ImGui::TextDisabled("no warning");
        ImGui::SeparatorText("Triggers (requested versus fired host time)"); const auto log = engine.trigger_log(); std::vector<double> err; for (const auto& e : log) err.push_back((e.fired_host_s - e.requested_host_s) * 1000.0); std::sort(err.begin(), err.end());
        if (err.empty()) ImGui::TextDisabled("no trigger fired yet"); else ImGui::Text("%zu firings: median %.3f ms, p95 %.3f ms, max %.3f ms, late %d", err.size(), err[err.size() / 2], err[size_t(0.95 * double(err.size() - 1))], err.back(), int(std::count_if(log.begin(), log.end(), [](const TriggerLogEntry& e) { return e.late; })));
        if (ImGui::BeginTable("trig", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Borders, ImVec2(0, 220))) { ImGui::TableSetupColumn("rule"); ImGui::TableSetupColumn("event time (s)"); ImGui::TableSetupColumn("error (ms)"); ImGui::TableSetupColumn("late"); ImGui::TableHeadersRow();
            for (size_t i = log.size(); i-- > 0 && log.size() - i <= 200;) { ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::TextUnformatted(log[i].rule.c_str()); ImGui::TableNextColumn(); ImGui::Text("%.3f", log[i].anchor_event_t); ImGui::TableNextColumn(); ImGui::Text("%.3f", (log[i].fired_host_s - log[i].requested_host_s) * 1000.0); ImGui::TableNextColumn(); ImGui::TextUnformatted(log[i].late ? "late" : ""); }
            ImGui::EndTable(); }
    }
};
static TabRegistrar reg([] { return std::make_unique<LogTab>(); });
}  // namespace
}  // namespace eyelive

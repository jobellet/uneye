// Triggers tab (order 60): rules that fire a UDP message or a log line after an event, optionally delayed. The state block (Predict tab) is the interface for MATLAB / PsychoPy; no TTL.
#include <algorithm>
#include <cstdio>
#include <sstream>
#include <cstdlib>
#include "imgui.h"
#include "eyelive/gui/settings.hpp"
#include "eyelive/gui/widgets.hpp"
#include "eyelive/gui_tab.hpp"

namespace eyelive {
namespace {
struct Row { char name[48] = "rule"; int cls = 1; bool any = false; float amin = 0, amax = 1e9f; int anchor = 0; float delay = 0, refr = 0; char sink[160] = "stdout"; char payload[96] = "T"; bool enabled = true; };

class TriggersTab final : public ITab {
 public:
    TriggersTab() { auto& s = gui::Settings::instance(); const int n = int(s.get_number("trg.count", 0)); for (int i = 0; i < n; ++i) { Row r; std::stringstream ss(s.get("trg." + std::to_string(i))); std::string f; std::vector<std::string> v; while (std::getline(ss, f, '\t')) v.push_back(f); if (v.size() < 11) continue;
            std::snprintf(r.name, sizeof r.name, "%s", v[0].c_str()); r.cls = std::atoi(v[1].c_str()); r.any = v[2] == "1"; r.amin = float(std::atof(v[3].c_str())); r.amax = float(std::atof(v[4].c_str())); r.anchor = std::atoi(v[5].c_str()); r.delay = float(std::atof(v[6].c_str())); r.refr = float(std::atof(v[7].c_str())); std::snprintf(r.sink, sizeof r.sink, "%s", v[8].c_str()); std::snprintf(r.payload, sizeof r.payload, "%s", v[9].c_str()); r.enabled = v[10] == "1"; rows_.push_back(r); } }
    const char* title() const override { return "Triggers"; }
    int order() const override { return 60; }
    bool wants_redraw() const override { return true; }
    void draw(IEngine& engine) override {
        ImGui::TextWrapped("A rule fires when an event matches its class and amplitude range: 'onset (provisional)' fires 3-6 ms after the movement starts (no amplitude is known yet); 'onset (final)' and 'offset' use the final event. Sinks: udp:127.0.0.1:6000, log:<file>, stdout. Payload placeholders: {t} event time, {amp} amplitude, {cls} class.");
        bool changed = false; const char* anchors[3] = {"onset (provisional)", "onset (final)", "offset"}; const char* classes[5] = {"fixation", "saccade", "blink", "class 3", "class 4"};
        if (ImGui::BeginTable("rules", 11, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_ScrollY, ImVec2(0, 230))) {
            for (const char* h : {"on", "name", "any class", "class", "amp min", "amp max", "anchor", "delay ms", "refractory ms", "sink", "payload"}) ImGui::TableSetupColumn(h); ImGui::TableHeadersRow(); int remove = -1;
            for (size_t i = 0; i < rows_.size(); ++i) { Row& r = rows_[i]; ImGui::PushID(int(i)); ImGui::TableNextRow(); ImGui::TableNextColumn(); changed |= ImGui::Checkbox("##on", &r.enabled); ImGui::TableNextColumn(); ImGui::SetNextItemWidth(110); changed |= ImGui::InputText("##name", r.name, sizeof r.name); ImGui::TableNextColumn(); changed |= ImGui::Checkbox("##any", &r.any);
                ImGui::TableNextColumn(); ImGui::BeginDisabled(r.any); ImGui::SetNextItemWidth(90); if (ImGui::BeginCombo("##cls", classes[std::max(0, std::min(r.cls, 4))])) { for (int c = 0; c < 5; ++c) if (ImGui::Selectable(classes[c], c == r.cls)) { r.cls = c; changed = true; } ImGui::EndCombo(); } ImGui::EndDisabled();
                ImGui::TableNextColumn(); ImGui::SetNextItemWidth(60); changed |= ImGui::InputFloat("##amin", &r.amin, 0, 0, "%.2f"); ImGui::TableNextColumn(); ImGui::SetNextItemWidth(70); changed |= ImGui::InputFloat("##amax", &r.amax, 0, 0, "%.2f");
                ImGui::TableNextColumn(); ImGui::SetNextItemWidth(150); if (ImGui::BeginCombo("##anchor", anchors[r.anchor])) { for (int a = 0; a < 3; ++a) if (ImGui::Selectable(anchors[a], a == r.anchor)) { r.anchor = a; changed = true; } ImGui::EndCombo(); }
                ImGui::TableNextColumn(); ImGui::SetNextItemWidth(70); changed |= ImGui::InputFloat("##delay", &r.delay, 0, 0, "%.1f"); ImGui::TableNextColumn(); ImGui::SetNextItemWidth(70); changed |= ImGui::InputFloat("##refr", &r.refr, 0, 0, "%.0f");
                ImGui::TableNextColumn(); ImGui::SetNextItemWidth(220); changed |= ImGui::InputText("##sink", r.sink, sizeof r.sink); ImGui::TableNextColumn(); ImGui::SetNextItemWidth(130); changed |= ImGui::InputText("##payload", r.payload, sizeof r.payload); ImGui::SameLine(); if (ImGui::SmallButton("x")) remove = int(i);
                if (r.anchor == 0 && (r.amin > 0 || r.amax < 1e8f)) { ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::TableNextColumn(); ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1), "an amplitude range cannot match provisional events: use an onset (final) or offset anchor"); } ImGui::PopID(); }
            if (remove >= 0) { rows_.erase(rows_.begin() + remove); changed = true; } ImGui::EndTable(); }
        if (ImGui::Button("Add rule")) { rows_.push_back(Row()); changed = true; } ImGui::SameLine(); if (ImGui::Button("Add 'every saccade' example")) { Row r; std::snprintf(r.name, sizeof r.name, "saccade_udp"); std::snprintf(r.sink, sizeof r.sink, "udp:127.0.0.1:6000"); std::snprintf(r.payload, sizeof r.payload, "SACCADE {t} {amp}"); rows_.push_back(r); changed = true; }
        ImGui::SameLine(); if (ImGui::Button("Add microsaccade example")) { Row r; std::snprintf(r.name, sizeof r.name, "microsaccade"); r.anchor = 1; r.amax = 1.0f; std::snprintf(r.sink, sizeof r.sink, "udp:127.0.0.1:6000"); std::snprintf(r.payload, sizeof r.payload, "MICRO {t} {amp}"); rows_.push_back(r); changed = true; }
        if (changed || first_) { apply(engine); save(); first_ = false; }
        ImGui::SeparatorText("Firing log (requested versus actual host time)"); const auto log = engine.trigger_log(); std::vector<double> err; for (const auto& e : log) err.push_back((e.fired_host_s - e.requested_host_s) * 1000.0); std::sort(err.begin(), err.end());
        if (err.empty()) ImGui::TextDisabled("nothing fired yet"); else ImGui::Text("%zu firings; error median %.3f ms, p95 %.3f ms, max %.3f ms", err.size(), err[err.size() / 2], err[size_t(0.95 * double(err.size() - 1))], err.back());
    }
 private:
    void apply(IEngine& engine) { std::vector<TriggerRule> rules; for (const Row& r : rows_) { TriggerRule t; t.name = r.name; t.cls = uint8_t(r.cls); t.any_class = r.any; t.amp_min_deg = r.amin; t.amp_max_deg = r.amax; t.anchor = r.anchor == 0 ? TriggerAnchor::OnsetProvisional : (r.anchor == 1 ? TriggerAnchor::OnsetFinal : TriggerAnchor::Offset); t.delay_ms = r.delay; t.refractory_ms = r.refr; t.sink = r.sink; t.payload = r.payload; t.enabled = r.enabled; rules.push_back(t); } engine.set_trigger_rules(rules); }
    void save() const { auto& s = gui::Settings::instance(); s.set_number("trg.count", double(rows_.size())); for (size_t i = 0; i < rows_.size(); ++i) { const Row& r = rows_[i]; char b[512]; std::snprintf(b, sizeof b, "%s\t%d\t%d\t%g\t%g\t%d\t%g\t%g\t%s\t%s\t%d", r.name, r.cls, r.any ? 1 : 0, double(r.amin), double(r.amax), r.anchor, double(r.delay), double(r.refr), r.sink, r.payload, r.enabled ? 1 : 0); s.set("trg." + std::to_string(i), b); } }
    std::vector<Row> rows_; bool first_ = true;
};
static TabRegistrar reg([] { return std::make_unique<TriggersTab>(); });
}  // namespace
}  // namespace eyelive

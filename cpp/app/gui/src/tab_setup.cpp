// Setup tab (order 5): the device profile in one place (tracker, network, rate, units and screen, detector, landing network, trigger rules, duration), saved as a named profile, with the checks.
#include <cmath>
#include <cstdio>
#include "imgui.h"
#include "eyelive/gui/shell_ui.hpp"
#include "eyelive/gui/tab_acquire.hpp"
#include "eyelive/gui/widgets.hpp"
#include "eyelive/gui_tab.hpp"

namespace eyelive {
namespace {
class SetupTab final : public ITab {
 public:
    const char* title() const override { return "Setup"; }
    int order() const override { return 5; }
    void draw(IEngine& engine) override {
        using namespace gui; UiController& ui = shared_ui(); const bool running = engine.stats().running; DeviceProfile p = ui.profile(); bool changed = false; ImGui::PushItemWidth(std::min(520.0f, ImGui::GetContentRegionAvail().x * 0.6f));
        if (running) ImGui::TextColored(ImVec4(1, 0.75f, 0.2f, 1), "Recording: the setup can be edited after the Stop (the running session keeps its configuration).");
        ImGui::BeginDisabled(running);
        ImGui::SeparatorText("Profile"); changed |= input_string("name", p.name, 80); help_marker("A profile holds everything you set once. It is saved as a small text file in ~/.eyelive/profiles."); changed |= input_string("tracker (free text)", p.tracker, 80);
        ImGui::SeparatorText("Where the gaze comes from"); int src = p.source.kind == "sim" ? 0 : (p.source.kind == "udp" ? 1 : 2); if (ImGui::RadioButton("simulated eye", &src, 0) | (ImGui::SameLine(), ImGui::RadioButton("tracker over UDP", &src, 1)) | (ImGui::SameLine(), ImGui::RadioButton("replay a CSV file", &src, 2))) { p.source.kind = src == 0 ? "sim" : (src == 1 ? "udp" : "file"); changed = true; }
        if (src == 1) { changed |= input_string("listen on address", p.source.bind_address, 64); ImGui::SetNextItemWidth(140); changed |= ImGui::InputInt("UDP port", &p.source.port, 0, 0); }
        if (src == 2) { changed |= input_string("CSV file", p.source.path, 500); float sp = float(p.source.speed); ImGui::SetNextItemWidth(140); if (ImGui::InputFloat("speed (1 = real time, 0 = as fast as possible)", &sp, 0, 0, "%.2f")) { p.source.speed = sp; changed = true; } }
        if (src == 0) { float r = float(p.source.rate_hz); ImGui::SetNextItemWidth(140); if (ImGui::InputFloat("simulated rate (Hz)", &r, 0, 0, "%.0f")) { p.source.rate_hz = r; changed = true; } }
        { float r = float(p.rate_hz); ImGui::SetNextItemWidth(140); if (ImGui::InputFloat("declared sampling rate (Hz, 0 = measure it)", &r, 0, 0, "%.0f")) { p.rate_hz = r; changed = true; } }
        ImGui::SeparatorText("Units and screen"); int u = p.units == "degrees" ? 0 : (p.units == "pixels" ? 1 : 2); if (ImGui::RadioButton("degrees", &u, 0) | (ImGui::SameLine(), ImGui::RadioButton("screen pixels", &u, 1)) | (ImGui::SameLine(), ImGui::RadioButton("unknown (estimate)", &u, 2))) { p.units = u == 0 ? "degrees" : (u == 1 ? "pixels" : "auto"); changed = true; }
        if (u == 1) { double w = p.screen.width_cm, h = p.screen.height_cm, d = p.screen.distance_cm; ImGui::SetNextItemWidth(110); if (ImGui::InputDouble("width (cm)", &w, 0, 0, "%.1f")) { p.screen.width_cm = w; changed = true; } ImGui::SameLine(); ImGui::SetNextItemWidth(110); if (ImGui::InputDouble("height (cm)", &h, 0, 0, "%.1f")) { p.screen.height_cm = h; changed = true; } ImGui::SameLine(); ImGui::SetNextItemWidth(110); if (ImGui::InputDouble("distance (cm)", &d, 0, 0, "%.1f")) { p.screen.distance_cm = d; changed = true; }
            ImGui::SetNextItemWidth(110); changed |= ImGui::InputInt("resolution x", &p.screen.res_x, 0, 0); ImGui::SameLine(); ImGui::SetNextItemWidth(110); changed |= ImGui::InputInt("y", &p.screen.res_y, 0, 0); const double ppd = pixels_per_degree(p.screen); if (ppd > 0) ImGui::Text("= %.2f pixels per degree (%.4f degrees per pixel) at the centre of the screen", ppd, 1.0 / ppd); }
        ImGui::SeparatorText("Detection and prediction"); { const auto dets = engine.available_detectors(); std::string cur = p.detector_id; for (auto& d : dets) if (d.id == p.detector_id) cur = d.name; ImGui::SetNextItemWidth(420); if (ImGui::BeginCombo("detector", cur.c_str())) { for (auto& d : dets) if (ImGui::Selectable(d.name.c_str(), d.id == p.detector_id)) { p.detector_id = d.id; changed = true; } ImGui::EndCombo(); } }
        { const auto models = ui.landing_models(); const std::string cur = p.landing_model.empty() ? "none (fixed tables)" : p.landing_model; ImGui::SetNextItemWidth(420); if (ImGui::BeginCombo("landing network", cur.c_str())) { if (ImGui::Selectable("none (fixed tables)", p.landing_model.empty())) { p.landing_model.clear(); changed = true; } for (auto& m : models) if (ImGui::Selectable(m.c_str(), m == p.landing_model)) { p.landing_model = m; changed = true; } ImGui::EndCombo(); }
          help_marker("Estimates where a saccade will land from its first milliseconds. A network is valid for the kind of experiment it was trained on (landing_mlp_gazebase: target steps; landing_mlp_drone: free viewing of videos); the fixed tables work everywhere."); }
        ImGui::SeparatorText("Triggers and duration"); changed |= input_string("trigger rules file (empty = none)", p.triggers_file, 500); if (!p.triggers_file.empty()) { std::vector<TriggerRule> r; std::string e; if (load_trigger_rules(p.triggers_file, r, &e)) ImGui::TextColored(ImVec4(0.3f, 0.85f, 0.4f, 1), "%zu rule(s) read", r.size()); else ImGui::TextColored(ImVec4(1, 0.45f, 0.35f, 1), "%s", e.c_str()); }
        { float m = float(p.duration_min); ImGui::SetNextItemWidth(140); if (ImGui::InputFloat("planned duration (min)", &m, 0, 0, "%.0f")) { p.duration_min = m; changed = true; } const double rate = p.rate_hz > 0 ? p.rate_hz : (p.source.kind == "sim" ? p.source.rate_hz : 1000.0); ImGui::TextDisabled("preallocated session file: %s", sizing_text(rate, p.duration_min * 60.0).c_str()); }
        ImGui::EndDisabled(); ImGui::PopItemWidth(); if (changed) ui.set_profile(p);
        std::vector<std::string> problems; const bool ok = profile_problems(ui.profile(), &problems); for (const auto& s : problems) ImGui::TextColored(ok ? ImVec4(1, 0.75f, 0.2f, 1) : ImVec4(1, 0.45f, 0.35f, 1), "%s", s.c_str());
        ImGui::Spacing(); ImGui::BeginDisabled(running || !ok); if (ImGui::Button("Save profile")) { std::string e; if (ui.save_profile(&e)) message_ = "saved in " + ui.profile_dir(); else message_ = e; } ImGui::EndDisabled(); ImGui::SameLine(); if (ImGui::Button("Run checks")) checks_ = ui.checks(); ImGui::SameLine();
        ImGui::BeginDisabled(running); if (ImGui::Button("Delete this profile")) { std::string e; const std::string key = std::string(profile_file_name(ui.profile().name)).substr(0, profile_file_name(ui.profile().name).size() - 8); message_ = ui.delete_profile(key, &e) ? "deleted" : e; } ImGui::EndDisabled(); if (!message_.empty()) { ImGui::SameLine(); ImGui::TextDisabled("%s", message_.c_str()); }
        if (!checks_.empty()) { ImGui::SeparatorText("Checks"); for (const auto& c : checks_) { const ImVec4 col = c.level == CheckLevel::Ok ? ImVec4(0.3f, 0.85f, 0.4f, 1) : (c.level == CheckLevel::Warn ? ImVec4(1, 0.75f, 0.2f, 1) : ImVec4(1, 0.4f, 0.35f, 1)); ImGui::TextColored(col, "%s", c.level == CheckLevel::Ok ? "[ok]" : (c.level == CheckLevel::Warn ? "[check]" : "[problem]")); ImGui::SameLine(); ImGui::TextWrapped("%s: %s%s%s", c.what.c_str(), c.message.c_str(), c.hint.empty() ? "" : "  ->  ", c.hint.c_str()); } }
    }
    std::string message_; std::vector<CheckResult> checks_;
};
static TabRegistrar reg([] { return std::make_unique<SetupTab>(); });
}  // namespace
}  // namespace eyelive

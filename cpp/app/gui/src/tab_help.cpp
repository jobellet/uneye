// Help tab (order 90): the checks, where things are, the usual problems, and a diagnostic text to paste into a message to the support.
#include <ctime>
#include "imgui.h"
#include "eyelive/export.hpp"
#include "eyelive/gui/shell_ui.hpp"
#include "eyelive/gui/widgets.hpp"
#include "eyelive/gui_tab.hpp"
#include "eyelive/net_detectors.hpp"

namespace eyelive {
namespace {
class HelpTab final : public ITab {
 public:
    const char* title() const override { return "Help"; }
    int order() const override { return 90; }
    void draw(IEngine& engine) override {
        using namespace gui; UiController& ui = shared_ui(); if (checks_.empty() || ImGui::Button("Run the checks again")) checks_ = ui.checks();
        ImGui::SeparatorText(("Checks for the profile '" + ui.profile().name + "'").c_str());
        for (const auto& c : checks_) { const ImVec4 col = c.level == CheckLevel::Ok ? ImVec4(0.3f, 0.85f, 0.4f, 1) : (c.level == CheckLevel::Warn ? ImVec4(1, 0.75f, 0.2f, 1) : ImVec4(1, 0.4f, 0.35f, 1)); ImGui::TextColored(col, "%s", c.level == CheckLevel::Ok ? "[ok]" : (c.level == CheckLevel::Warn ? "[check]" : "[problem]")); ImGui::SameLine(); ImGui::TextWrapped("%s: %s%s%s", c.what.c_str(), c.message.c_str(), c.hint.empty() ? "" : "  ->  ", c.hint.c_str()); }
        ImGui::SeparatorText("Where things are");
        ImGui::Text("profiles:  %s", ui.profile_dir().c_str()); ImGui::Text("sessions:  %s", ui.session_dir().c_str()); ImGui::Text("models:    %s", models_dir().empty() ? "(not found: set --models-dir)" : models_dir().c_str()); ImGui::Text("version:   %s", engine_version().c_str()); ImGui::TextDisabled("Next to each session file: <session>.provenance.json (how it was recorded) and <session>.report.html (quality report).");
        ImGui::SeparatorText("The usual problems");
        ImGui::TextWrapped("No sample arrives: the tracker must stream to the address and port of the profile; allow the program through the firewall (private networks); with the tracker on the same computer use 127.0.0.1.");
        ImGui::TextWrapped("Amplitudes look 10 to 40 times too large: the stream is in pixels while the profile says degrees; set the units and the screen in the Setup tab.");
        ImGui::TextWrapped("Many gaps or lost packets: close other programs, use a wired network, lower the planned rate of the tracker's streaming, check the cable.");
        ImGui::TextWrapped("The main sequence in the report is weak or there are no events: wrong scale, noisy tracking, or a detector that does not suit the recording (try another one in Setup, then 'eyelive_tool redetect' on the finished sessions).");
        ImGui::SeparatorText("Diagnostic text"); if (ImGui::Button("Copy to the clipboard")) ImGui::SetClipboardText(diagnostic(engine, ui).c_str()); ImGui::SameLine(); ImGui::TextDisabled("version, profile, checks and statistics; no gaze data.");
    }
    static std::string diagnostic(IEngine& engine, gui::UiController& ui) {
        const EngineStats s = engine.stats(); std::string o = "EyeLive " + engine_version() + "\n" + serialize_profile(ui.profile()) + "--- checks\n";
        for (const auto& c : ui.checks()) o += std::string(c.level == CheckLevel::Ok ? "ok    " : (c.level == CheckLevel::Warn ? "check " : "FAIL  ")) + c.what + ": " + c.message + "\n";
        char b[400]; std::snprintf(b, sizeof b, "--- engine\nrunning=%d samples=%llu rate=%.2f dropped=%llu gaps=%llu latency_p95=%.2f ms cpu=%.1f%% scale=%.5g warning='%s'\n", int(s.running), (unsigned long long)s.samples, s.rate_hz_measured, (unsigned long long)s.dropped_packets, (unsigned long long)s.gaps, s.detect_latency_ms_p95, s.cpu_percent, s.deg_per_unit, s.warning.c_str()); return o + b;
    }
    std::vector<CheckResult> checks_;
};
static TabRegistrar reg([] { return std::make_unique<HelpTab>(); });
}  // namespace
}  // namespace eyelive

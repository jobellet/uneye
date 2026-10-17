#include <cstdio>
#include <memory>
#include "imgui.h"
#include "eyelive/gui/shell_ui.hpp"
#include "eyelive/gui/widgets.hpp"

namespace eyelive { namespace gui {
namespace {
std::string g_profile_dir, g_session_dir;
}
void configure_shared_ui(const std::string& profile_dir, const std::string& session_dir) { g_profile_dir = profile_dir; g_session_dir = session_dir; }
UiController& shared_ui() { static UiController ui(Settings::instance(), g_profile_dir.empty() ? default_profile_dir() : g_profile_dir, g_session_dir); return ui; }

ImVec4 health_color(HealthLevel l) { switch (l) { case HealthLevel::Ok: return ImVec4(0.30f, 0.85f, 0.40f, 1); case HealthLevel::Warn: return ImVec4(1.0f, 0.75f, 0.20f, 1); case HealthLevel::Bad: return ImVec4(1.0f, 0.35f, 0.30f, 1); default: return ImVec4(0.62f, 0.62f, 0.66f, 1); } }
const char* health_word(HealthLevel l) { switch (l) { case HealthLevel::Ok: return "OK"; case HealthLevel::Warn: return "CHECK"; case HealthLevel::Bad: return "PROBLEM"; default: return "idle"; } }

bool draw_strip(IEngine& engine, UiController& ui, std::string& error, bool* open_wizard) {
    bool mode_changed = false; const EngineStats st = engine.stats(); const bool running = st.running;
    // line 1: the modes, and on the right the health light
    for (Mode m : {Mode::Record, Mode::Review, Mode::Analyze, Mode::Help, Mode::Advanced}) { const bool sel = ui.mode() == m; if (sel) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive)); if (ImGui::Button(mode_name(m), ImVec2(0, 0))) { if (!sel) { ui.set_mode(m); mode_changed = true; } } if (sel) ImGui::PopStyleColor(); ImGui::SameLine(); }
    const Health h = ui.health(engine); const std::string label = std::string(health_word(h.level)) + (h.message.empty() || h.level == HealthLevel::Idle ? "" : ": " + h.message); const float w = ImGui::CalcTextSize(label.c_str()).x + 34; ImGui::SameLine(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowWidth() - w - 12));
    { const ImVec2 p = ImGui::GetCursorScreenPos(); const float r = ImGui::GetTextLineHeight() * 0.38f; ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(p.x + r + 2, p.y + ImGui::GetTextLineHeight() * 0.5f), r, ImGui::ColorConvertFloat4ToU32(health_color(h.level))); ImGui::Dummy(ImVec2(r * 2 + 8, ImGui::GetTextLineHeight())); ImGui::SameLine(); ImGui::TextColored(health_color(h.level), "%s", label.c_str()); }
    // line 2: profile, Start / Stop, elapsed time
    ImGui::BeginDisabled(running); { std::string cur = ui.profile().name; for (const auto& e : ui.profiles()) if (e.key == ui.profile().name) cur = e.label; ImGui::SetNextItemWidth(260); if (ImGui::BeginCombo("##profile", ("Profile: " + cur).c_str())) { for (const auto& e : ui.profiles()) { std::string txt = e.label + (e.builtin ? "  (built-in)" : ""); if (ImGui::Selectable(txt.c_str(), e.label == ui.profile().name)) { std::string err; if (!ui.select_profile(e.key, &err)) error = err; else error.clear(); } } ImGui::EndCombo(); } }
    ImGui::SameLine(); if (ImGui::Button("Setup wizard") && open_wizard) *open_wizard = true; ImGui::EndDisabled(); ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Button, running ? ImVec4(0.72f, 0.22f, 0.20f, 1) : ImVec4(0.16f, 0.58f, 0.28f, 1)); ImGui::PushStyleColor(ImGuiCol_ButtonHovered, running ? ImVec4(0.85f, 0.30f, 0.26f, 1) : ImVec4(0.20f, 0.70f, 0.34f, 1));
    if (ImGui::Button(running ? "  Stop  " : "  Start recording  ", ImVec2(190, ImGui::GetFrameHeight() * 1.5f))) { if (running) { ui.stop(engine); error.clear(); } else { error.clear(); std::string e; if (!ui.start(engine, &e)) error = e; } }
    ImGui::PopStyleColor(2); ImGui::SameLine(); if (running) ImGui::Text("%s", format_time(ui.elapsed_s()).c_str()); else if (!ui.last_session().empty()) ImGui::TextDisabled("last: %s", ui.last_session().c_str());
    if (!error.empty()) ImGui::TextColored(ImVec4(1, 0.45f, 0.35f, 1), "%s", error.c_str());
    ImGui::Separator(); return mode_changed;
}

bool draw_wizard(UiController& ui, WizardState& w, bool open_now) {
    if (open_now) ImGui::OpenPopup("Welcome to EyeLive");
    const ImVec2 c = ImGui::GetMainViewport()->GetCenter(); ImGui::SetNextWindowPos(c, ImGuiCond_Always, ImVec2(0.5f, 0.5f)); ImGui::SetNextWindowSize(ImVec2(std::min(760.0f, ImGui::GetMainViewport()->Size.x - 40.0f), 0));
    bool open = true; if (!ImGui::BeginPopupModal("Welcome to EyeLive", &open, ImGuiWindowFlags_AlwaysAutoResize)) return false;
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 680); ImGui::TextDisabled("Four questions, then one button to record. You can change everything later in the Setup tab."); ImGui::PopTextWrapPos();
    ImGui::Separator(); ImGui::Text("%s", WizardState::step_title(std::min(w.step, 4))); ImGui::Spacing();
    switch (w.step) {
        case WizardState::Tracker:
            ImGui::RadioButton("Let me try the software first (a simulated eye, nothing to connect)", &w.tracker_choice, 0); ImGui::RadioButton("My eye tracker sends the gaze over the network (UDP) to this computer", &w.tracker_choice, 1); ImGui::RadioButton("Replay a recorded file (CSV: time, x, y)", &w.tracker_choice, 2);
            if (w.tracker_choice == 1) { input_string("tracker (free text, for the reports)", w.tracker_label, 80); } if (w.tracker_choice == 2) { input_string("CSV file", w.file, 500); help_marker("One line per sample: t,x,y or t,x,y,valid. The first line may be a header."); } break;
        case WizardState::Signal:
            if (w.tracker_choice == 0) { ImGui::TextWrapped("Nothing to set for the simulated eye (1000 Hz, degrees)."); break; }
            if (w.tracker_choice == 1) { input_string("listen on address", w.bind, 64); help_marker("127.0.0.1 = the tracker computer is this computer. 0.0.0.0 = any computer on the network."); ImGui::SetNextItemWidth(140); ImGui::InputInt("UDP port", &w.port, 0, 0); ImGui::TextDisabled("Set the same address and port in the streaming settings of your tracker."); }
            ImGui::Spacing(); ImGui::TextUnformatted("Sampling rate:"); ImGui::RadioButton("measure it", &w.rate_choice, 0); ImGui::SameLine(); ImGui::RadioButton("500 Hz", &w.rate_choice, 1); ImGui::SameLine(); ImGui::RadioButton("1000 Hz", &w.rate_choice, 2); ImGui::SameLine(); ImGui::RadioButton("2000 Hz", &w.rate_choice, 3);
            ImGui::Spacing(); ImGui::TextUnformatted("Positions are in:"); ImGui::RadioButton("degrees", &w.units_choice, 0); ImGui::SameLine(); ImGui::RadioButton("screen pixels", &w.units_choice, 1); ImGui::SameLine(); ImGui::RadioButton("I do not know", &w.units_choice, 2); help_marker("Pixels need the screen size and the viewing distance (next question). 'I do not know': the scale is estimated from the first minute of saccades; fine for detection, not for exact amplitudes."); break;
        case WizardState::Screen: {
            ImGui::SetNextItemWidth(120); ImGui::InputDouble("screen width (cm)", &w.width_cm, 0, 0, "%.1f"); ImGui::SameLine(); ImGui::SetNextItemWidth(120); ImGui::InputDouble("height (cm)", &w.height_cm, 0, 0, "%.1f"); ImGui::SetNextItemWidth(120); ImGui::InputDouble("viewing distance (cm)", &w.distance_cm, 0, 0, "%.1f");
            ImGui::SetNextItemWidth(120); ImGui::InputInt("horizontal resolution (pixels)", &w.res_x, 0, 0); ImGui::SameLine(); ImGui::SetNextItemWidth(120); ImGui::InputInt("vertical", &w.res_y, 0, 0); ScreenGeometry g; g.width_cm = w.width_cm; g.distance_cm = w.distance_cm; g.res_x = w.res_x; const double ppd = pixels_per_degree(g);
            if (ppd > 0) ImGui::Text("= %.2f pixels per degree at the centre of the screen", ppd); break; }
        case WizardState::Goals: {
            ImGui::Checkbox("Fire a trigger (to MATLAB, PsychoPy, a log file...) when a saccade starts", &w.want_triggers); if (w.want_triggers) { input_string("trigger rules file", w.triggers_file, 500); help_marker("One rule per line, for example: name=sacc;cls=1;anchor=provisional;delay_ms=10;sink=udp:127.0.0.1:6000;payload=SACC {t} {amp}. See the manual."); if (!w.triggers_file.empty()) { std::vector<TriggerRule> r; std::string e; if (load_trigger_rules(w.triggers_file, r, &e)) ImGui::TextColored(ImVec4(0.3f, 0.85f, 0.4f, 1), "%zu rule(s) read", r.size()); else ImGui::TextColored(ImVec4(1, 0.45f, 0.35f, 1), "%s", e.c_str()); } }
            ImGui::Spacing(); ImGui::TextUnformatted("Estimate where a saccade will land (for gaze-contingent displays):"); ImGui::RadioButton("fixed tables (no network)", &w.landing_choice, 0); ImGui::RadioButton("network for target-step tasks (GazeBase-like: large saccades to known targets)", &w.landing_choice, 1); ImGui::RadioButton("network for free viewing of videos (small saccades)", &w.landing_choice, 2);
            ImGui::TextDisabled("A network is valid for the kind of experiment it was trained on; check it on your own data before relying on it."); break; }
        case WizardState::Name: input_string("name of this setup", w.name, 80); { const DeviceProfile p = w.to_profile(); std::vector<std::string> pr; const bool ok = profile_problems(p, &pr); ImGui::Spacing(); ImGui::TextDisabled("Summary"); ImGui::BeginChild("summary", ImVec2(0, 150), ImGuiChildFlags_Borders); ImGui::TextUnformatted(serialize_profile(p).c_str()); ImGui::EndChild(); for (const auto& s : pr) ImGui::TextColored(ok ? ImVec4(1, 0.75f, 0.2f, 1) : ImVec4(1, 0.45f, 0.35f, 1), "%s", s.c_str()); } break;
        default: break;
    }
    for (const auto& e : w.errors) ImGui::TextColored(ImVec4(1, 0.45f, 0.35f, 1), "%s", e.c_str());
    ImGui::Separator(); ImGui::BeginDisabled(w.step == WizardState::Tracker); if (ImGui::Button("Back")) w.back(); ImGui::EndDisabled(); ImGui::SameLine();
    const bool last = w.step == WizardState::Name; static std::string finish_err; if (ImGui::Button(last ? "Finish" : "Next", ImVec2(120, 0))) { if (w.next() && w.finished()) { finish_err.clear(); if (ui.finish_wizard(w, &finish_err)) { ImGui::CloseCurrentPopup(); open = false; } else { w.step = WizardState::Name; } } }
    ImGui::SameLine(); if (ImGui::Button("Skip: use the simulation")) { std::string e; ui.select_profile("Simulation (demo)", &e); Settings::instance().set("ui.wizard_done", "1"); ImGui::CloseCurrentPopup(); open = false; }
    if (!finish_err.empty()) ImGui::TextColored(ImVec4(1, 0.45f, 0.35f, 1), "%s", finish_err.c_str());
    ImGui::EndPopup(); return open;
}

}}  // namespace eyelive::gui

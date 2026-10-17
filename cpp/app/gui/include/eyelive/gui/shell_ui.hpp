// ImGui drawing of the simplified shell: the control strip (modes, profile, big Start / Stop button, health light) and the first-launch wizard. The logic is in ui_logic.hpp.
#pragma once
#include "eyelive/engine.hpp"
#include "eyelive/gui/ui_logic.hpp"

namespace eyelive { namespace gui {

UiController& shared_ui();                                    // one controller for the whole program (settings of the user, ~/.eyelive/profiles, ~/EyeLive)
void configure_shared_ui(const std::string& profile_dir, const std::string& session_dir);        // before the first use (tests, --profiles-dir, --session-dir)
// draws the strip at the current cursor position; returns true when the mode changed (the shell then selects the first tab of the mode)
bool draw_strip(IEngine& engine, UiController& ui, std::string& error, bool* open_wizard);
// draws the wizard as a modal window; returns true while it is open
bool draw_wizard(UiController& ui, WizardState& w, bool open_now);
ImVec4 health_color(HealthLevel l);
const char* health_word(HealthLevel l);                       // "idle", "OK", "CHECK", "PROBLEM": the colour is never the only information

}}  // namespace eyelive::gui

// Predict tab (order 70): landing-point prediction is not implemented in this version; the tab explains how MATLAB / PsychoPy read the live state block (the real-time interface of the application).
#include <cstdio>
#include "imgui.h"
#include "eyelive/gui/settings.hpp"
#include "eyelive/gui_tab.hpp"

namespace eyelive {
namespace {
class PredictTab final : public ITab {
 public:
    const char* title() const override { return "Predict"; }
    int order() const override { return 70; }
    bool wants_redraw() const override { return false; }
    void draw(IEngine&) override {
        const std::string path = gui::Settings::instance().get("acq.last_session", "session.eyelive");
        ImGui::SeparatorText("Saccade landing prediction"); ImGui::TextWrapped("Not implemented in this version (planned after the detection and annotation workflow). The state block already reserves the fields for the predicted landing point and its uncertainty.");
        ImGui::SeparatorText("Live state for MATLAB / Psychtoolbox and PsychoPy (the session file is the interface)"); ImGui::TextWrapped("While a session records, the first 512 bytes of the session file hold a header and a 256-byte state block at offset 256: the last gaze sample, the speed, whether an event is in progress, counters of provisional and final events and the last event (time, amplitude, peak speed, class). Map the file read-only and read it with the seqlock protocol; no driver, no TTL. Session file of the last start:");
        ImGui::TextColored(ImVec4(0.6f, 0.85f, 1, 1), "%s", path.c_str());
        ImGui::SeparatorText("MATLAB"); ImGui::TextUnformatted("m = memmapfile('<session file>', 'Offset', 256, 'Format', 'uint8', 'Repeat', 256, 'Writable', false);\ns = eyelive_state(m.Data);          % cpp/app/readers/matlab/eyelive_state.m  (retries until the sequence number is even and unchanged)\nfprintf('%d final events, last gaze (%.2f, %.2f)\\n', s.event_counter, s.x, s.y);");
        ImGui::SeparatorText("Python / PsychoPy"); ImGui::TextUnformatted("from eyelive_reader import StateReader     # cpp/app/readers/python/eyelive_reader.py\nr = StateReader('<session file>')\ns = r.latest()                          # dict: x, y, speed_deg_s, in_event, event_counter, ...\nfor ev in r.follow_events(): ...        # one dict per new final event");
        ImGui::TextDisabled("Documentation: cpp/app/readers/README.md and docs/eyelive/.");
    }
};
static TabRegistrar reg([] { return std::make_unique<PredictTab>(); });
}  // namespace
}  // namespace eyelive

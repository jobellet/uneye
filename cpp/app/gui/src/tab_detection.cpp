// Detection tab (order 30): the available detectors with their properties, hot swap of the active one, convention calibration to the expert's corrections, the unit scale in use.
#include <algorithm>
#include <cstdio>
#include "imgui.h"
#include "eyelive/gui/widgets.hpp"
#include "eyelive/gui_tab.hpp"

namespace eyelive {
namespace {
class DetectionTab final : public ITab {
 public:
    const char* title() const override { return "Detection"; }
    int order() const override { return 30; }
    bool wants_redraw() const override { return false; }
    void draw(IEngine& engine) override {
        const EngineStats st = engine.stats(); const auto dets = engine.available_detectors();
        ImGui::SeparatorText("Detectors"); ImGui::TextWrapped("Pick the detector that labels the stream. 'hmm_accel' needs no labels at all; the neural detectors were trained on the owner's datasets; a model you train in the Train tab appears here as user:<name>. Changing the detector while recording loses no sample and no event.");
        if (ImGui::BeginTable("dets", 6, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchProp)) { ImGui::TableSetupColumn("detector"); ImGui::TableSetupColumn("id"); ImGui::TableSetupColumn("classes"); ImGui::TableSetupColumn("decision delay"); ImGui::TableSetupColumn("provisional events"); ImGui::TableSetupColumn(""); ImGui::TableHeadersRow();
            for (const auto& d : dets) { ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::TextUnformatted(d.name.c_str()); ImGui::TableNextColumn(); ImGui::TextDisabled("%s", d.id.c_str()); ImGui::TableNextColumn(); ImGui::Text("%d", d.n_classes); ImGui::TableNextColumn(); ImGui::Text("%.0f ms%s", d.decision_delay_ms, d.causal ? " (causal)" : ""); ImGui::TableNextColumn(); if (d.provisional_latency_ms >= 0) ImGui::Text("after %.0f ms", d.provisional_latency_ms); else ImGui::TextDisabled("no");
                ImGui::TableNextColumn(); ImGui::PushID(d.id.c_str()); if (ImGui::SmallButton("Use now")) { std::string err; if (engine.set_detector(d.id, &err)) { active_ = d.id; msg_ = "active detector: " + d.name; } else msg_ = err; } ImGui::PopID(); }
            ImGui::EndTable(); }
        if (!msg_.empty()) ImGui::TextColored(ImVec4(0.6f, 0.9f, 0.6f, 1), "%s", msg_.c_str());
        ImGui::SeparatorText("Unit scale"); ImGui::Text("degrees per stream unit: %.5g (%s)%s", st.deg_per_unit, st.scale_estimated ? "estimated from the data" : "given", st.scale_low_confidence ? " - LOW CONFIDENCE" : "");
        if (st.scale_estimated) ImGui::Text("agreement between the network and the unit-free detector at this scale: %.2f", st.scale_agreement);
        ImGui::TextDisabled("To enter the scale by hand (pixels per degree, or the screen geometry), choose it in the Acquire tab before starting a session.");
        ImGui::SeparatorText("Calibrate to my corrections"); ImGui::TextWrapped("Learns how you report events (smallest amplitude kept, how early the onset is placed, how late the offset, the merging distance) from the events you corrected in the Review tab, and applies the same convention to the detector's output. 3 to 10 annotated stretches are enough.");
        if (ImGui::Button("Fit the convention to my corrections")) { std::string err; calib_ = engine.fit_calibration_to_expert(1, &err); have_calib_ = err.empty(); calib_msg_ = err; }
        if (have_calib_) { ImGui::Text("minimum amplitude %.2f deg | onset shift %+d samples | offset shift %+d samples | merge gap %d samples", double(calib_.amp_min_deg), calib_.onset_shift, calib_.offset_shift, calib_.merge_gap);
            if (ImGui::Button("Show the calibrated events in the Review tab")) { const ITraceView& tr = engine.trace(); const auto cal = apply_calibration(tr, engine.events(), calib_); engine.expert().load_prelabels(cal, 0, tr.size()); calib_msg_ = "calibrated events loaded as pre-labels"; } }
        if (!calib_msg_.empty()) ImGui::TextColored(ImVec4(1, 0.7f, 0.4f, 1), "%s", calib_msg_.c_str());
    }
 private:
    std::string active_, msg_, calib_msg_; CalibParams calib_; bool have_calib_ = false;
};
static TabRegistrar reg([] { return std::make_unique<DetectionTab>(); });
}  // namespace
}  // namespace eyelive

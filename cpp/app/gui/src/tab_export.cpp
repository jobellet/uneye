// Export tab (order 75): the last (or any) session to CSV, BIDS-style TSV + JSON, MAT-file, or a one-page report; the summary of the session is shown here.
#include "imgui.h"
#include "eyelive/export.hpp"
#include "eyelive/gui/shell_ui.hpp"
#include "eyelive/gui/widgets.hpp"
#include "eyelive/gui_tab.hpp"

namespace eyelive {
namespace {
class ExportTab final : public ITab {
 public:
    const char* title() const override { return "Export"; }
    int order() const override { return 75; }
    void draw(IEngine& engine) override {
        using namespace gui; UiController& ui = shared_ui(); if (path_.empty() || (!engine.stats().running && path_ != ui.last_session() && !edited_)) { path_ = ui.last_session(); summary_.clear(); }
        ImGui::TextWrapped("Write what the session contains in the formats other programs read. Nothing here changes the session file.");
        if (input_string("session file", path_, 600)) { edited_ = true; summary_.clear(); } if (engine.stats().running) ImGui::TextColored(ImVec4(1, 0.75f, 0.2f, 1), "A session is being recorded: export it after the Stop.");
        ImGui::BeginDisabled(engine.stats().running);
        auto button = [&](const char* label, ExportKind k, const char* what) { if (ImGui::Button(label, ImVec2(230, 0))) { std::string err; const std::string out = UiController::default_export_path(path_, k); if (ui.export_session(path_, k, out, &err)) { message_ = std::string(what) + " written: " + out; ok_ = true; } else { message_ = err; ok_ = false; } } ImGui::SameLine(); ImGui::TextDisabled("%s", what); };
        ImGui::SeparatorText("Export"); button("Events (CSV)", ExportKind::EventsCsv, "one row per saccade / blink: times, amplitude, peak speed, direction"); ImGui::NewLine(); button("BIDS-style (TSV + JSON)", ExportKind::Bids, "events and samples with their JSON descriptions"); ImGui::NewLine(); button("MAT-file", ExportKind::Mat, "t, x, y, valid, label, expert, events (MATLAB, Octave, scipy)"); ImGui::NewLine(); button("Report (HTML)", ExportKind::Report, "one page: quality of the recording, main sequence, warnings");
        ImGui::NewLine(); if (ImGui::Button("Show the summary", ImVec2(230, 0))) { std::string err; auto rec = Recorder::open_readonly(path_, &err); if (rec) { summary_ = report_text(summarize(*rec)); message_.clear(); } else { message_ = err; ok_ = false; summary_.clear(); } }
        ImGui::EndDisabled(); if (!message_.empty()) ImGui::TextColored(ok_ ? ImVec4(0.3f, 0.85f, 0.4f, 1) : ImVec4(1, 0.45f, 0.35f, 1), "%s", message_.c_str());
        if (!summary_.empty()) { ImGui::SeparatorText("Summary"); ImGui::BeginChild("summary", ImVec2(0, 0), ImGuiChildFlags_Borders); ImGui::TextUnformatted(summary_.c_str()); ImGui::EndChild(); }
    }
    std::string path_, message_, summary_; bool ok_ = false, edited_ = false;
};
static TabRegistrar reg([] { return std::make_unique<ExportTab>(); });
}  // namespace
}  // namespace eyelive

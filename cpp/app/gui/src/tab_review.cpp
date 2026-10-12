// Review tab (order 40): the expert corrects pre-labelled events (presence, onset, offset, class), with undo / redo, reviewed and excluded ranges, an uncertain-first queue and the agreement with the detector.
//  mouse: click an event = select, drag its edges = move onset / offset, drag in "add" mode = new event; keys: Delete remove, 1-5 class, Z undo, Y redo, S split at the cursor, M merge with the next, R mark the window reviewed, X exclude it, [ ] previous / next uncertain.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <future>
#include <memory>
#include "imgui.h"
#include "eyelive/gui/widgets.hpp"
#include "eyelive/gui_tab.hpp"

namespace eyelive {
namespace {
class ReviewTab final : public ITab {
 public:
    const char* title() const override { return "Review"; }
    int order() const override { return 40; }
    bool wants_redraw() const override { return job_.valid(); }
    void draw(IEngine& engine) override {
        ExpertLabels& ex = engine.expert(); const ITraceView& tr = engine.trace(); const double rate = tr.info().rate_hz > 0 ? tr.info().rate_hz : 1000.0; const int64_t n = tr.size();
        poll_job(); const bool busy = job_.valid();
        if (busy) { ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1), "pre-labelling..."); ImGui::BeginDisabled(); }
        // ---- toolbar
        const auto dets = engine.available_detectors(); ImGui::SetNextItemWidth(300);
        if (ImGui::BeginCombo("pre-labeler", name_of(dets, detector_).c_str())) { for (auto& d : dets) if (ImGui::Selectable(d.name.c_str(), d.id == detector_)) detector_ = d.id; ImGui::EndCombo(); }
        ImGui::SameLine(); if (ImGui::Button("Pre-label all")) start_job(engine, 0, n); ImGui::SameLine(); if (ImGui::Button("Pre-label window")) start_job(engine, std::max<int64_t>(0, int64_t(view_.t0 * rate)), std::min<int64_t>(n, int64_t(view_.t1 * rate)));
        ImGui::SameLine(); ImGui::SetNextItemWidth(140); const int nc = ex.n_classes(); const char* names[5] = {"fixation", "saccade", "blink", "class 3", "class 4"}; add_class_ = std::max(1, std::min(add_class_, nc - 1)); if (ImGui::BeginCombo("class", names[add_class_])) { for (int c = 1; c < nc; ++c) if (ImGui::Selectable(names[c], c == add_class_)) add_class_ = c; ImGui::EndCombo(); }
        ImGui::SameLine(); ImGui::Checkbox("add mode (drag)", &add_mode_); ImGui::SameLine(); if (ImGui::Button("Undo (Z)")) ex.undo(); ImGui::SameLine(); if (ImGui::Button("Redo (Y)")) ex.redo();
        if (ImGui::Button("Mark window reviewed (R)")) mark_window(ex, rate, n, true); ImGui::SameLine(); if (ImGui::Button("Exclude window (X)")) { ex.exclude_range(int64_t(view_.t0 * rate), std::min<int64_t>(n, int64_t(view_.t1 * rate)), true); }
        ImGui::SameLine(); if (ImGui::Button("Next uncertain (])")) goto_uncertain(ex, rate, +1);
        const double rev_s = ex.reviewed_seconds(rate); ImGui::Text("reviewed: %.1f s (default training target 50 s)", rev_s); ImGui::SameLine(); ImGui::ProgressBar(float(std::min(1.0, rev_s / 50.0)), ImVec2(140, 0)); const AgreementStats ag = agreement_cached(engine); ImGui::SameLine(); ImGui::Text("| detector vs expert: F1 %.2f, kappa %.2f (tp %d, fp %d, fn %d)", ag.event_f1, ag.kappa, ag.tp, ag.fp, ag.fn);
        if (!msg_.empty()) { ImGui::SameLine(); ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1), "%s", msg_.c_str()); }
        // ---- plot with editing
        const std::vector<Event>& ev = ex.events(); const float avail = ImGui::GetContentRegionAvail().y; const float list_w = 380.0f; const float plot_w = std::max(200.0f, ImGui::GetContentRegionAvail().x - list_w - 10);
        if (!inited_ && n > 0) { view_.t0 = 0; view_.t1 = std::min(6.0, double(n) / rate); inited_ = true; }
        ImGui::BeginGroup(); gui::PlotStyle ps; ps.pan = !dragging_ && !(add_mode_ && ImGui::IsMouseDown(ImGuiMouseButton_Left)); gui::PlotHit h = gui::draw_traces("review_plot", ImVec2(plot_w, std::max(200.0f, avail - 6)), tr, view_, ev, selected_, ps); frame_ = h.frame;
        ImDrawList* dl = ImGui::GetWindowDrawList(); const ImGuiIO& io = ImGui::GetIO();
        if (selected_ >= 0 && selected_ < int(ev.size())) { const Event& e = ev[size_t(selected_)]; const float xa = frame_.sx(double(e.onset) / rate), xb = frame_.sx(double(e.offset + 1) / rate); for (float x : {xa, xb}) dl->AddLine(ImVec2(x, frame_.p0.y), ImVec2(x, frame_.p1.y), IM_COL32(255, 255, 255, 230), 2.5f); }
        const ImGuiID id = ImGui::GetID("review_plot"); (void)id;
        // mouse logic (after the plot's own invisible button)
        if (!busy && h.hovered) {
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                press_t_ = h.t; moved_ = false; dragging_ = false; drag_edge_ = 0; creating_ = false;
                if (add_mode_) creating_ = true;
                else if (selected_ >= 0 && selected_ < int(ev.size())) { const Event& e = ev[size_t(selected_)]; if (std::fabs(io.MousePos.x - frame_.sx(double(e.onset) / rate)) < 7) { drag_edge_ = -1; dragging_ = true; } else if (std::fabs(io.MousePos.x - frame_.sx(double(e.offset + 1) / rate)) < 7) { drag_edge_ = +1; dragging_ = true; } }
            }
        }
        if (!busy && (dragging_ || creating_) && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            const int64_t cur = std::max<int64_t>(0, std::min<int64_t>(n - 1, int64_t(frame_.tx(io.MousePos.x) * rate))); moved_ = true;
            if (creating_) { const int64_t a = std::min(cur, int64_t(press_t_ * rate)), b = std::max(cur, int64_t(press_t_ * rate)); dl->AddRectFilled(ImVec2(frame_.sx(double(a) / rate), frame_.p0.y), ImVec2(frame_.sx(double(b + 1) / rate), frame_.p1.y), gui::class_color(add_class_, 0.3f)); }
            else dl->AddLine(ImVec2(io.MousePos.x, frame_.p0.y), ImVec2(io.MousePos.x, frame_.p1.y), IM_COL32(255, 255, 0, 200), 2.0f);
        }
        if (!busy && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            const int64_t cur = std::max<int64_t>(0, std::min<int64_t>(n - 1, int64_t(frame_.tx(io.MousePos.x) * rate)));
            if (creating_ && moved_) { const int64_t a = std::min(cur, int64_t(press_t_ * rate)), b = std::max(cur, int64_t(press_t_ * rate)); if (b - a >= 2) { if (!ex.add(a, b, uint8_t(add_class_))) msg_ = "could not add: the event overlaps another one"; else { msg_.clear(); select_at(ex, (a + b) / 2); } } }
            else if (dragging_ && moved_ && selected_ >= 0 && selected_ < int(ev.size())) { const bool ok = drag_edge_ < 0 ? ex.move_onset(size_t(selected_), cur) : ex.move_offset(size_t(selected_), std::max<int64_t>(cur - 1, 0)); msg_ = ok ? "" : "cannot move the edge there"; }
            else if (!moved_ && h.hovered) select_at(ex, int64_t(h.t * rate));
            dragging_ = creating_ = false; drag_edge_ = 0;
        }
        keys(ex, rate, n, ev, h);
        ImGui::EndGroup(); ImGui::SameLine();
        // ---- event list
        ImGui::BeginGroup(); ImGui::Text("%zu events (%zu from the pre-labeler)", ev.size(), ex.prelabels().size());
        if (ImGui::BeginTable("evl", 6, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_Borders, ImVec2(list_w, std::max(100.0f, ImGui::GetContentRegionAvail().y - 4)))) {
            ImGui::TableSetupColumn("t (s)"); ImGui::TableSetupColumn("ms"); ImGui::TableSetupColumn("deg"); ImGui::TableSetupColumn("class"); ImGui::TableSetupColumn("conf"); ImGui::TableSetupColumn("by"); ImGui::TableHeadersRow();
            ImGuiListClipper clip; clip.Begin(int(ev.size())); while (clip.Step()) for (int k = clip.DisplayStart; k < clip.DisplayEnd; ++k) { const Event& e = ev[size_t(k)]; ImGui::TableNextRow(); ImGui::TableNextColumn(); char lab[32]; std::snprintf(lab, sizeof lab, "%.3f##%d", double(e.onset) / rate, k);
                if (ImGui::Selectable(lab, k == selected_, ImGuiSelectableFlags_SpanAllColumns)) { selected_ = k; center_on(e, rate); } ImGui::TableNextColumn(); ImGui::Text("%.0f", double(e.offset - e.onset + 1) * 1000.0 / rate); ImGui::TableNextColumn(); ImGui::Text("%.2f", double(e.amplitude_deg));
                ImGui::TableNextColumn(); ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(gui::class_color(e.cls)), "%s", gui::class_name(e.cls)); ImGui::TableNextColumn(); ImGui::Text("%.2f", double(e.confidence)); ImGui::TableNextColumn(); ImGui::TextUnformatted(e.origin == EventOrigin::Expert ? "you" : "model"); }
            ImGui::EndTable(); }
        ImGui::EndGroup(); if (busy) ImGui::EndDisabled();
    }
 private:
    static std::string name_of(const std::vector<DetectorInfo>& d, const std::string& id) { for (auto& i : d) if (i.id == id) return i.name; return id; }
    void start_job(IEngine& engine, int64_t a, int64_t b) { if (job_.valid()) return; msg_.clear(); const std::string id = detector_; IEngine* e = &engine; job_ = std::async(std::launch::async, [e, id, a, b]() -> std::string { std::string err; return e->prelabel(id, a, b, &err) ? std::string() : (err.empty() ? std::string("pre-labelling failed") : err); }); }
    void poll_job() { if (job_.valid() && job_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) { msg_ = job_.get(); selected_ = -1; ag_time_ = -1e9; } }
    AgreementStats agreement_cached(IEngine& engine) { const double now = ImGui::GetTime(); if (now - ag_time_ > 1.5) { ag_ = engine.agreement_with_expert(1); ag_time_ = now; } return ag_; }
    void select_at(ExpertLabels& ex, int64_t sample) { selected_ = -1; const auto& ev = ex.events(); for (size_t k = 0; k < ev.size(); ++k) if (sample >= ev[k].onset && sample <= ev[k].offset) { selected_ = int(k); return; } int best = -1; int64_t bd = 1 << 30; for (size_t k = 0; k < ev.size(); ++k) { const int64_t d = std::min(std::llabs(ev[k].onset - sample), std::llabs(ev[k].offset - sample)); if (d < bd) { bd = d; best = int(k); } } if (best >= 0 && bd < 40) selected_ = best; }
    void center_on(const Event& e, double rate) { const double c = 0.5 * double(e.onset + e.offset) / rate, half = 0.5 * (view_.t1 - view_.t0); view_.t0 = c - half; view_.t1 = c + half; }
    void mark_window(ExpertLabels& ex, double rate, int64_t n, bool on) { ex.mark_reviewed(std::max<int64_t>(0, int64_t(view_.t0 * rate)), std::min<int64_t>(n, int64_t(view_.t1 * rate)), on); ag_time_ = -1e9; }
    void goto_uncertain(ExpertLabels& ex, double rate, int dir) {
        const auto& ev = ex.events(); std::vector<int> q; for (size_t k = 0; k < ev.size(); ++k) if (ev[k].origin == EventOrigin::Detector) q.push_back(int(k)); if (q.empty()) { msg_ = "no unreviewed event left"; return; }
        std::sort(q.begin(), q.end(), [&](int a, int b) { return ev[size_t(a)].confidence < ev[size_t(b)].confidence; }); queue_pos_ = (queue_pos_ + (dir > 0 ? 1 : -1) + int(q.size())) % int(q.size()); selected_ = q[size_t(queue_pos_)]; center_on(ev[size_t(selected_)], rate); msg_.clear();
    }
    void keys(ExpertLabels& ex, double rate, int64_t n, const std::vector<Event>& ev, const gui::PlotHit& h) {
        if (job_.valid() || ImGui::GetIO().WantTextInput || !ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) return;
        if (ImGui::IsKeyPressed(ImGuiKey_Z)) ex.undo(); if (ImGui::IsKeyPressed(ImGuiKey_Y)) ex.redo(); if (ImGui::IsKeyPressed(ImGuiKey_R)) mark_window(ex, rate, n, true); if (ImGui::IsKeyPressed(ImGuiKey_X)) ex.exclude_range(int64_t(view_.t0 * rate), std::min<int64_t>(n, int64_t(view_.t1 * rate)), true);
        if (ImGui::IsKeyPressed(ImGuiKey_RightBracket)) goto_uncertain(ex, rate, +1); if (ImGui::IsKeyPressed(ImGuiKey_LeftBracket)) goto_uncertain(ex, rate, -1);
        if (selected_ >= 0 && selected_ < int(ev.size())) {
            if (ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_Backspace)) { ex.remove(size_t(selected_)); selected_ = -1; return; }
            for (int c = 1; c <= 4; ++c) if (ImGui::IsKeyPressed(ImGuiKey(ImGuiKey_0 + c)) && c < ex.n_classes()) ex.set_class(size_t(selected_), uint8_t(c));
            if (ImGui::IsKeyPressed(ImGuiKey_S) && h.hovered) { const bool ok = ex.split(size_t(selected_), std::max<int64_t>(0, int64_t(h.t * rate))); msg_ = ok ? "" : "cannot split there"; }
            if (ImGui::IsKeyPressed(ImGuiKey_M) && selected_ + 1 < int(ev.size())) { const bool ok = ex.merge(size_t(selected_), size_t(selected_ + 1)); msg_ = ok ? "" : "can merge adjacent events of the same class only"; }
        }
    }
    std::future<std::string> job_; std::string msg_, detector_ = "hmm_accel"; gui::TimeView view_{0, 6}; gui::PlotFrame frame_; int selected_ = -1, add_class_ = 1, drag_edge_ = 0, queue_pos_ = -1; bool add_mode_ = false, dragging_ = false, creating_ = false, moved_ = false, inited_ = false; double press_t_ = 0, ag_time_ = -1e9; AgreementStats ag_;
};
static TabRegistrar reg([] { return std::make_unique<ReviewTab>(); });
}  // namespace
}  // namespace eyelive

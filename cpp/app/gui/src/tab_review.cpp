// Review tab (order 40): the expert corrects pre-labelled events (presence, onset, offset, class), with undo / redo, reviewed and excluded ranges, an uncertain-first queue and the agreement with the detector.
//  mouse: click an event = select it (white outline, its edges can be dragged); "Add event": click the START then click the END (or drag across); Esc cancels. Buttons and keys: Delete / Backspace remove the selected event, 1-4 its class,
//  S split at the cursor, M merge with the next, Z / Y undo / redo, A add mode, R mark the window reviewed, X exclude it, [ ] previous / next uncertain event. Wheel = zoom, horizontal scroll or drag = pan.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <memory>
#include "imgui.h"
#include "eyelive/gui/settings.hpp"
#include "eyelive/gui/widgets.hpp"
#include "eyelive/gui_tab.hpp"

namespace eyelive {
namespace {
class ReviewTab final : public ITab {
 public:
    const char* title() const override { return "Review"; }
    int order() const override { return 40; }
    bool wants_redraw() const override { return job_.valid() || redraw_ > 0 || al_.training || al_.labeling || al_.retrain_in_s >= 0; }
    void draw(IEngine& engine) override {
        ExpertLabels& ex = engine.expert(); const ITraceView& tr = engine.trace(); const double rate = tr.info().rate_hz > 0 ? tr.info().rate_hz : 1000.0; const int64_t n = tr.size();
        if (redraw_ > 0) --redraw_; poll_job(); const bool busy = job_.valid();
        if (!auto_inited_ && n > 0) { auto& st = gui::Settings::instance(); auto_cfg_.enabled = st.get_number("auto.enabled", 1) != 0; auto_cfg_.retrain_delay_s = st.get_number("auto.delay", 5.0); auto_cfg_.pretrained_start = st.get_number("auto.pretrained", 0) != 0; auto_cfg_.expected_epochs = int(st.get_number("auto.epochs", 50)); engine.autolearn_set(auto_cfg_); auto_inited_ = true; }
        if (auto_inited_) { const bool was_training = al_.training; al_ = engine.autolearn_poll(edited_); edited_ = false; if (was_training != al_.training || al_.labeling) touch(); }
        if (busy) { ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1), "pre-labelling..."); ImGui::BeginDisabled(); }
        draw_auto_panel(engine, rate);
        // ---- toolbar, line 1: pre-labelling
        const auto dets = engine.available_detectors(); ImGui::SetNextItemWidth(300);
        if (ImGui::BeginCombo("pre-labeler", name_of(dets, detector_).c_str())) { for (auto& d : dets) if (ImGui::Selectable(d.name.c_str(), d.id == detector_)) detector_ = d.id; ImGui::EndCombo(); }
        ImGui::SameLine(); if (ImGui::Button("Pre-label all")) start_job(engine, 0, n); ImGui::SameLine(); if (ImGui::Button("Pre-label window")) start_job(engine, std::max<int64_t>(0, int64_t(view_.t0 * rate)), std::min<int64_t>(n, int64_t(view_.t1 * rate)));
        ImGui::SameLine(); if (ImGui::Button("Undo (Z)")) { ex.undo(); touch(); } ImGui::SameLine(); if (ImGui::Button("Redo (Y)")) { ex.redo(); touch(); }
        // ---- line 2: adding events and reviewed ranges
        const int nc = ex.n_classes(); const char* names[5] = {"fixation", "saccade", "blink", "class 3", "class 4"}; add_class_ = std::max(1, std::min(add_class_, nc - 1));
        if (add_mode_) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.45f, 0.1f, 1)); if (ImGui::Button(add_mode_ ? "Adding: click START then END (Esc = stop)" : "Add event (A)")) { add_mode_ = !add_mode_; pending_ = -1; } if (add_mode_) ImGui::PopStyleColor();
        ImGui::SameLine(); ImGui::SetNextItemWidth(120); if (ImGui::BeginCombo("new event class", names[add_class_])) { for (int c = 1; c < nc; ++c) if (ImGui::Selectable(names[c], c == add_class_)) add_class_ = c; ImGui::EndCombo(); }
        ImGui::SameLine(); if (ImGui::Button("Mark window reviewed (R)")) mark_window(ex, rate, n, true); ImGui::SameLine(); if (ImGui::Button("Exclude window (X)")) { ex.exclude_range(int64_t(view_.t0 * rate), std::min<int64_t>(n, int64_t(view_.t1 * rate)), true); touch(); }
        ImGui::SameLine(); if (ImGui::Button("Next uncertain (])")) goto_uncertain(ex, rate, +1);
        // ---- line 3: the selected event
        const std::vector<Event>& ev = ex.events(); if (selected_ >= int(ev.size())) selected_ = -1;
        if (selected_ >= 0) { const Event& e = ev[size_t(selected_)]; ImGui::TextColored(ImVec4(1, 1, 1, 1), "selected: %.3f s, %.0f ms, %.2f deg, %s", double(e.onset) / rate, double(e.offset - e.onset + 1) * 1000.0 / rate, double(e.amplitude_deg), names[std::min<int>(e.cls, 4)]);
            ImGui::SameLine(); ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.75f, 0.2f, 0.2f, 1)); if (ImGui::Button("Delete selected (Del)")) delete_selected(ex, rate); ImGui::PopStyleColor();
            for (int c = 1; c < nc; ++c) { ImGui::SameLine(); char l[40]; std::snprintf(l, sizeof l, "-> %s (%d)", names[c], c); if (ImGui::SmallButton(l)) { const Event e0 = ex.events()[size_t(selected_)]; ex.set_class(size_t(selected_), uint8_t(c)); edited(ex, e0.onset, e0.offset, rate); } }
            ImGui::SameLine(); if (ImGui::SmallButton("Split at cursor (S)")) split_at_cursor(ex, rate); ImGui::SameLine(); if (ImGui::SmallButton("Merge with next (M)")) merge_next(ex, rate); }
        else ImGui::TextDisabled("click an event in the plot or in the list to select it");
        const double rev_s = ex.reviewed_seconds(rate); ImGui::Text("reviewed: %.1f s of the 50 s target", rev_s); ImGui::SameLine(); ImGui::ProgressBar(float(std::min(1.0, rev_s / 50.0)), ImVec2(120, 0)); const AgreementStats ag = agreement_cached(engine); ImGui::SameLine(); ImGui::Text("| detector vs your corrections (reviewed parts): F1 %.2f, kappa %.2f (tp %d, fp %d, fn %d)", ag.event_f1, ag.kappa, ag.tp, ag.fp, ag.fn);
        if (!msg_.empty()) { ImGui::SameLine(); ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1), "%s", msg_.c_str()); }
        // ---- plot
        const float avail = ImGui::GetContentRegionAvail().y; const float list_w = 380.0f; const float plot_w = std::max(200.0f, ImGui::GetContentRegionAvail().x - list_w - 10);
        if (!inited_ && n > 0) { view_.t0 = 0; view_.t1 = std::min(6.0, double(n) / rate); inited_ = true; }
        ImGui::BeginGroup(); gui::PlotStyle ps; ps.pan = !dragging_ && !add_mode_; gui::PlotHit h = gui::draw_traces("review_plot", ImVec2(plot_w, std::max(200.0f, avail - 6)), tr, view_, ev, selected_, ps); frame_ = h.frame;
        ImDrawList* dl = ImGui::GetWindowDrawList(); const ImGuiIO& io = ImGui::GetIO(); const auto sample_at = [&](float x) { return std::max<int64_t>(0, std::min<int64_t>(n - 1, int64_t(frame_.tx(x) * rate))); };
        if (selected_ >= 0) { const Event& e = ev[size_t(selected_)]; const float xa = frame_.sx(double(e.onset) / rate), xb = frame_.sx(double(e.offset + 1) / rate); for (float x : {xa, xb}) { dl->AddLine(ImVec2(x, frame_.p0.y), ImVec2(x, frame_.p1.y), IM_COL32(255, 255, 255, 230), 2.5f); dl->AddTriangleFilled(ImVec2(x - 6, frame_.p0.y), ImVec2(x + 6, frame_.p0.y), ImVec2(x, frame_.p0.y + 10), IM_COL32(255, 255, 255, 230)); } }
        if (pending_ >= 0) { const float xs = frame_.sx(double(pending_) / rate); dl->AddLine(ImVec2(xs, frame_.p0.y), ImVec2(xs, frame_.p1.y), IM_COL32(255, 160, 40, 255), 2.5f); if (h.hovered) { const float xm = io.MousePos.x; dl->AddRectFilled(ImVec2(std::min(xs, xm), frame_.p0.y), ImVec2(std::max(xs, xm), frame_.p1.y), gui::class_color(add_class_, 0.25f)); dl->AddLine(ImVec2(xm, frame_.p0.y), ImVec2(xm, frame_.p1.y), IM_COL32(255, 160, 40, 200), 2.0f); } }
        if (add_mode_ && h.hovered && pending_ < 0) dl->AddLine(ImVec2(io.MousePos.x, frame_.p0.y), ImVec2(io.MousePos.x, frame_.p1.y), IM_COL32(255, 160, 40, 160), 1.5f);
        if (!busy && h.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            press_x_ = io.MousePos.x; press_t_ = h.t; moved_ = false; dragging_ = false; drag_edge_ = 0; pressing_ = true;       // the press started on the plot
            if (!add_mode_ && selected_ >= 0) { const Event& e = ev[size_t(selected_)]; if (std::fabs(io.MousePos.x - frame_.sx(double(e.onset) / rate)) < 8) { drag_edge_ = -1; dragging_ = true; } else if (std::fabs(io.MousePos.x - frame_.sx(double(e.offset + 1) / rate)) < 8) { drag_edge_ = +1; dragging_ = true; } }
        }
        if (!busy && pressing_ && ImGui::IsMouseDown(ImGuiMouseButton_Left) && std::fabs(io.MousePos.x - press_x_) > 5.0f) moved_ = true;
        if (!busy && pressing_ && dragging_ && moved_ && ImGui::IsMouseDown(ImGuiMouseButton_Left)) dl->AddLine(ImVec2(io.MousePos.x, frame_.p0.y), ImVec2(io.MousePos.x, frame_.p1.y), IM_COL32(255, 255, 0, 220), 2.0f);
        if (!busy && pressing_ && add_mode_ && moved_ && ImGui::IsMouseDown(ImGuiMouseButton_Left) && pending_ < 0) dl->AddRectFilled(ImVec2(std::min(press_x_, io.MousePos.x), frame_.p0.y), ImVec2(std::max(press_x_, io.MousePos.x), frame_.p1.y), gui::class_color(add_class_, 0.3f));
        if (!busy && pressing_ && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            const int64_t cur = sample_at(io.MousePos.x);
            if (dragging_ && moved_ && selected_ >= 0) { const bool ok = drag_edge_ < 0 ? ex.move_onset(size_t(selected_), cur) : ex.move_offset(size_t(selected_), std::max<int64_t>(cur - 1, 0)); msg_ = ok ? "" : "cannot move the edge there (it would overlap another event)"; if (ok && selected_ >= 0) { const Event& m = ex.events()[size_t(selected_)]; edited(ex, m.onset, m.offset, rate); } else touch(); }
            else if (add_mode_ && moved_ && pending_ < 0) create(ex, sample_at(press_x_), cur, rate);                   // drag across
            else if (add_mode_ && !moved_ && h.hovered) { if (pending_ < 0) pending_ = cur; else { create(ex, pending_, cur, rate); pending_ = -1; } }       // click start, click end
            else if (!add_mode_ && !moved_ && h.hovered) { select_at(ex, int64_t(h.t * rate)); touch(); }
            dragging_ = false; drag_edge_ = 0; moved_ = false; pressing_ = false;
        }
        keys(ex, rate, n, h);
        ImGui::EndGroup(); ImGui::SameLine();
        // ---- event list
        ImGui::BeginGroup(); ImGui::Text("%zu events (%zu from the pre-labeler)", ev.size(), ex.prelabels().size());
        if (ImGui::BeginTable("evl", 6, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_Borders, ImVec2(list_w, std::max(100.0f, ImGui::GetContentRegionAvail().y - 4)))) {
            ImGui::TableSetupColumn("t (s)"); ImGui::TableSetupColumn("ms"); ImGui::TableSetupColumn("deg"); ImGui::TableSetupColumn("class"); ImGui::TableSetupColumn("conf"); ImGui::TableSetupColumn("by"); ImGui::TableHeadersRow();
            const std::vector<Event>& ev2 = ex.events(); ImGuiListClipper clip; clip.Begin(int(ev2.size())); while (clip.Step()) for (int k = clip.DisplayStart; k < clip.DisplayEnd; ++k) { const Event& e = ev2[size_t(k)]; ImGui::TableNextRow(); ImGui::TableNextColumn(); char lab[32]; std::snprintf(lab, sizeof lab, "%.3f##%d", double(e.onset) / rate, k);
                if (ImGui::Selectable(lab, k == selected_, ImGuiSelectableFlags_SpanAllColumns)) { selected_ = k; center_on(e, rate); touch(); } ImGui::TableNextColumn(); ImGui::Text("%.0f", double(e.offset - e.onset + 1) * 1000.0 / rate); ImGui::TableNextColumn(); ImGui::Text("%.2f", double(e.amplitude_deg));
                ImGui::TableNextColumn(); ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(gui::class_color(e.cls)), "%s", gui::class_name(e.cls)); ImGui::TableNextColumn(); ImGui::Text("%.2f", double(e.confidence)); ImGui::TableNextColumn(); ImGui::TextUnformatted(e.origin == EventOrigin::Expert ? "you" : "model"); }
            ImGui::EndTable(); }
        ImGui::EndGroup(); if (busy) ImGui::EndDisabled();
    }
 private:
    void touch() { redraw_ = 3; ag_time_ = -1e9; }
    // an edit makes its neighbourhood "labelled by the expert" (+-0.5 s, silently: one Undo reverts the edit and this range) and tells the assisted-annotation loop
    void edited(ExpertLabels& ex, int64_t on, int64_t off, double rate) { const int64_t w = int64_t(0.5 * rate); ex.mark_reviewed(std::max<int64_t>(0, on - w), off + w, true, false); edited_ = true; touch(); }                                               // the edit changed what must be drawn: keep rendering for a few frames (render-on-demand shell)
    void draw_auto_panel(IEngine& engine, double rate) {
        (void)rate; if (!auto_inited_) return; bool en = auto_cfg_.enabled;
        if (ImGui::Checkbox("Assisted annotation", &en)) { auto_cfg_.enabled = en; gui::Settings::instance().set_number("auto.enabled", en ? 1 : 0); engine.autolearn_set(auto_cfg_); touch(); }
        if (!auto_cfg_.enabled) { ImGui::SameLine(); ImGui::TextDisabled("off: use Pre-label and train by hand in the Train tab"); return; }
        ImGui::SameLine(); ImGui::Text("| labelled so far by: %s (model #%d) | %.1f s labelled", al_.labeler.c_str(), al_.generation, al_.labelled_s);
        if (al_.training) { ImGui::SameLine(); ImGui::Text("| training model #%d: epoch %d / ~%d", al_.generation + 1, al_.epoch, al_.expected_epochs); ImGui::SameLine(); char ov[32]; std::snprintf(ov, sizeof ov, "%d", al_.epoch); ImGui::ProgressBar(float(al_.progress), ImVec2(150, 0), ov); ImGui::SameLine(); ImGui::Text("val F1 %.2f", al_.val_f1); }
        else if (al_.retrain_in_s >= 0) { ImGui::SameLine(); ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1), "| a new model is trained %.0f s after your last correction", al_.retrain_in_s + 0.5); }
        else { ImGui::SameLine(); ImGui::TextDisabled("| correct an event: a model is trained from your corrections"); }
        ImGui::SameLine(); if (ImGui::SmallButton("Train now")) { std::string err; if (!engine.autolearn_train_now(&err)) msg_ = err; touch(); }
        ImGui::SameLine(); ImGui::SetNextItemWidth(70); float d = float(auto_cfg_.retrain_delay_s); if (ImGui::InputFloat("delay (s)", &d, 0, 0, "%.0f") && d >= 0 && d < 600) { auto_cfg_.retrain_delay_s = d; gui::Settings::instance().set_number("auto.delay", d); engine.autolearn_set(auto_cfg_); }
        if (!al_.message.empty()) { ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.9f, 0.6f, 1), "%s", al_.message.c_str()); }
        if (ImGui::TreeNode("Quality of each labeler (on the windows it labelled and you reviewed)")) {
            ImGui::TextWrapped("When a model agrees with your final labels as well as you need on windows it labelled BEFORE you looked at them, you can stop annotating. 'reviewed' = seconds of windows you went through; agreement F1 / kappa = its labels against your final labels there.");
            if (ImGui::BeginTable("gens", 8, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchProp)) { for (const char* h : {"#", "labeler", "reviewed (s)", "events", "agreement F1", "kappa", "val F1 (training)", "epochs"}) ImGui::TableSetupColumn(h); ImGui::TableHeadersRow();
                for (const auto& g : al_.history) { ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::Text("%d", g.id); ImGui::TableNextColumn(); ImGui::TextUnformatted(g.labeler.c_str()); ImGui::TableNextColumn(); ImGui::Text("%.1f", g.reviewed_s); ImGui::TableNextColumn(); ImGui::Text("%d", g.n_events); ImGui::TableNextColumn(); if (g.reviewed_s > 0) ImGui::Text("%.2f", g.f1); else ImGui::TextDisabled("-"); ImGui::TableNextColumn(); if (g.reviewed_s > 0) ImGui::Text("%.2f", g.kappa); else ImGui::TextDisabled("-"); ImGui::TableNextColumn(); if (g.id > 0) ImGui::Text("%.2f", g.val_f1); else ImGui::TextDisabled("-"); ImGui::TableNextColumn(); if (g.id > 0) ImGui::Text("%d", g.epochs); else ImGui::TextDisabled("-"); }
                ImGui::EndTable(); }
            ImGui::TreePop(); }
    }
    static std::string name_of(const std::vector<DetectorInfo>& d, const std::string& id) { for (auto& i : d) if (i.id == id) return i.name; return id; }
    void start_job(IEngine& engine, int64_t a, int64_t b) { if (job_.valid()) return; msg_.clear(); const std::string id = detector_; IEngine* e = &engine; job_ = std::async(std::launch::async, [e, id, a, b]() -> std::string { std::string err; return e->prelabel(id, a, b, &err) ? std::string() : (err.empty() ? std::string("pre-labelling failed") : err); }); }
    void poll_job() { if (job_.valid() && job_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) { msg_ = job_.get(); selected_ = -1; touch(); } }
    AgreementStats agreement_cached(IEngine& engine) { const double now = ImGui::GetTime(); if (now - ag_time_ > 1.5) { ag_ = engine.agreement_with_expert(1); ag_time_ = now; } return ag_; }
    void create(ExpertLabels& ex, int64_t a, int64_t b, double rate) { if (a > b) std::swap(a, b); if (b - a < 2) { msg_ = "event too short (click the start, then the end)"; return; } if (!ex.add(a, b, uint8_t(add_class_))) { msg_ = "could not add: it overlaps another event"; touch(); } else { msg_.clear(); select_at(ex, (a + b) / 2); edited(ex, a, b, rate); } }
    void center_if_outside(ExpertLabels&, double) {}
    void delete_selected(ExpertLabels& ex, double rate) { if (selected_ < 0 || selected_ >= int(ex.events().size())) return; const Event e = ex.events()[size_t(selected_)]; ex.remove(size_t(selected_)); selected_ = -1; msg_.clear(); edited(ex, e.onset, e.offset, rate); }
    void split_at_cursor(ExpertLabels& ex, double rate) { if (selected_ < 0) return; const Event& e = ex.events()[size_t(selected_)]; const int64_t at = std::max<int64_t>(e.onset + 1, std::min<int64_t>(e.offset, int64_t(cursor_t_ * rate))); const bool ok = ex.split(size_t(selected_), at); msg_ = ok ? "" : "cannot split there"; if (ok) edited(ex, e.onset, e.offset, rate); else touch(); }
    void merge_next(ExpertLabels& ex, double rate) { if (selected_ < 0 || selected_ + 1 >= int(ex.events().size())) return; const Event a = ex.events()[size_t(selected_)], b = ex.events()[size_t(selected_ + 1)]; const bool ok = ex.merge(size_t(selected_), size_t(selected_ + 1)); msg_ = ok ? "" : "can merge adjacent events of the same class only"; if (ok) edited(ex, a.onset, b.offset, rate); else touch(); }
    void select_at(ExpertLabels& ex, int64_t sample) { selected_ = -1; const auto& ev = ex.events(); for (size_t k = 0; k < ev.size(); ++k) if (sample >= ev[k].onset && sample <= ev[k].offset) { selected_ = int(k); return; } int best = -1; int64_t bd = 1 << 30; for (size_t k = 0; k < ev.size(); ++k) { const int64_t d = std::min(std::llabs(ev[k].onset - sample), std::llabs(ev[k].offset - sample)); if (d < bd) { bd = d; best = int(k); } } if (best >= 0 && bd < 60) selected_ = best; }
    void center_on(const Event& e, double rate) { const double c = 0.5 * double(e.onset + e.offset) / rate, half = 0.5 * (view_.t1 - view_.t0); view_.t0 = c - half; view_.t1 = c + half; }
    void mark_window(ExpertLabels& ex, double rate, int64_t n, bool on) { ex.mark_reviewed(std::max<int64_t>(0, int64_t(view_.t0 * rate)), std::min<int64_t>(n, int64_t(view_.t1 * rate)), on); edited_ = true; touch(); }
    void goto_uncertain(ExpertLabels& ex, double rate, int dir) {
        const auto& ev = ex.events(); std::vector<int> q; for (size_t k = 0; k < ev.size(); ++k) if (ev[k].origin == EventOrigin::Detector) q.push_back(int(k)); if (q.empty()) { msg_ = "no unreviewed event left"; return; }
        std::sort(q.begin(), q.end(), [&](int a, int b) { return ev[size_t(a)].confidence < ev[size_t(b)].confidence; }); queue_pos_ = (queue_pos_ + (dir > 0 ? 1 : -1) + int(q.size())) % int(q.size()); selected_ = q[size_t(queue_pos_)]; center_on(ev[size_t(selected_)], rate); msg_.clear(); touch();
    }
    void keys(ExpertLabels& ex, double rate, int64_t n, const gui::PlotHit& h) {
        if (h.hovered) cursor_t_ = h.t;
        if (job_.valid() || ImGui::GetIO().WantTextInput || !ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) return;
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) { pending_ = -1; add_mode_ = false; touch(); }
        if (ImGui::IsKeyPressed(ImGuiKey_A)) { add_mode_ = !add_mode_; pending_ = -1; touch(); }
        if (ImGui::IsKeyPressed(ImGuiKey_Z)) { ex.undo(); touch(); } if (ImGui::IsKeyPressed(ImGuiKey_Y)) { ex.redo(); touch(); } if (ImGui::IsKeyPressed(ImGuiKey_R)) mark_window(ex, rate, n, true);
        if (ImGui::IsKeyPressed(ImGuiKey_X)) { ex.exclude_range(int64_t(view_.t0 * rate), std::min<int64_t>(n, int64_t(view_.t1 * rate)), true); touch(); }
        if (ImGui::IsKeyPressed(ImGuiKey_RightBracket)) goto_uncertain(ex, rate, +1); if (ImGui::IsKeyPressed(ImGuiKey_LeftBracket)) goto_uncertain(ex, rate, -1);
        if (selected_ >= 0 && selected_ < int(ex.events().size())) {
            if (ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_Backspace)) { delete_selected(ex, rate); return; }
            for (int c = 1; c <= 4; ++c) if (ImGui::IsKeyPressed(ImGuiKey(ImGuiKey_0 + c)) && c < ex.n_classes()) { const Event e0 = ex.events()[size_t(selected_)]; ex.set_class(size_t(selected_), uint8_t(c)); edited(ex, e0.onset, e0.offset, rate); }
            if (ImGui::IsKeyPressed(ImGuiKey_S)) split_at_cursor(ex, rate); if (ImGui::IsKeyPressed(ImGuiKey_M)) merge_next(ex, rate);
        }
    }
    AutoLearnConfig auto_cfg_; AutoLearnStatus al_; bool auto_inited_ = false, edited_ = false;
    std::future<std::string> job_; std::string msg_, detector_ = "hmm_accel"; gui::TimeView view_{0, 6}; gui::PlotFrame frame_; int selected_ = -1, add_class_ = 1, drag_edge_ = 0, queue_pos_ = -1, redraw_ = 0; int64_t pending_ = -1;
    bool add_mode_ = false, dragging_ = false, moved_ = false, pressing_ = false, inited_ = false; double press_t_ = 0, cursor_t_ = 0, ag_time_ = -1e9; float press_x_ = 0; AgreementStats ag_;
};
static TabRegistrar reg([] { return std::make_unique<ReviewTab>(); });
}  // namespace
}  // namespace eyelive

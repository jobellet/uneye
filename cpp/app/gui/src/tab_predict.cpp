// Predict tab (order 70): the forecast of the eye position 10 and 20 ms ahead, drawn over the measurement in real time, with the error it makes measured on the last seconds, and how to read the live state in MATLAB / PsychoPy.
// The forecast comes from the gaze engine (two causal networks + landing tables of the saccades); it needs a 1 kHz stream.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include "imgui.h"
#include "eyelive/gui/settings.hpp"
#include "eyelive/gui/widgets.hpp"
#include "eyelive/gui_tab.hpp"

namespace eyelive {
namespace {
class PredictTab final : public ITab {
 public:
    const char* title() const override { return "Predict"; }
    int order() const override { return 70; }
    bool wants_redraw() const override { return live_; }
    void draw(IEngine& engine) override {
        const ITraceView& tr = engine.trace(); const EngineStats st = engine.stats(); live_ = st.running; const double rate = tr.info().rate_hz > 0 ? tr.info().rate_hz : 1000.0; const int64_t n = tr.size(); const double sc = st.deg_per_unit > 0 ? st.deg_per_unit : 1.0;
        ImGui::SeparatorText("Forecast of the eye position");
        if (!engine.forecast_available()) { ImGui::TextWrapped("No forecast: it needs a 1 kHz stream and the two causal networks (causal.bin, tcn_l10.bin in the models folder). Start a session with a 1 kHz source (the simulator, a replayed 1 kHz file or a 1 kHz tracker)."); }
        else {
            ImGui::TextWrapped("Magenta (x) and yellow (y): where the eye will be 20 ms after each sample, drawn at that future time. Where the line meets the measurement (blue, orange) the forecast was right; at the right edge it extends 20 ms beyond the last measurement, with its 80 %% interval. During a saccade the forecast follows the movement because the landing tables use the speed profile of saccades; during fixation it stays close to the current position.");
            ImGui::SetNextItemWidth(160); ImGui::SliderFloat("window (s)", &span_, 0.2f, 5.0f, "%.1f"); ImGui::SameLine(); ImGui::Checkbox("+10 ms line", &show10_); ImGui::SameLine(); ImGui::Checkbox("interval", &band_);
            view_.t1 = double(n) / rate + 0.03; view_.t0 = view_.t1 - double(span_);
            const int64_t f0 = std::max<int64_t>(0, int64_t(view_.t0 * rate) - 30), f1 = n; if (f1 > f0) engine.forecasts(f0, f1, fc_); gui::ForecastOverlay fo; fo.f = &fc_; fo.first = f0; fo.show10 = show10_; fo.band = band_; gui::PlotStyle ps; ps.show_speed = true; ps.forecast = &fo; ps.interactive = false;
            gui::draw_traces("predict_plot", ImVec2(0, std::max(220.0f, ImGui::GetContentRegionAvail().y - 250.0f)), tr, view_, engine.events(), -1, ps);
            // error of the forecast over the last 10 s, against what the eye really did (degrees), and against the trivial forecast "the eye stays where it is"
            const int64_t a = std::max<int64_t>(30, n - int64_t(10 * rate)), b = n - 21; std::vector<double> e10, e20, s20, e20m, s20m, e10m, s10m, e10f, e20f; int valid = 0, total = 0;
            if (b > a) { engine.forecasts(a, b + 1, tmp_); for (int64_t i = a; i <= b; ++i) { ++total; const ForecastSample& f = tmp_[size_t(i - a)]; if (!f.valid) continue; ++valid; const double xi = tr.x(i), yi = tr.y(i), x10 = tr.x(i + 10), y10 = tr.y(i + 10), x20 = tr.x(i + 20), y20 = tr.y(i + 20);
                const double ea = std::hypot(f.x10 - x10, f.y10 - y10) * sc, eb = std::hypot(f.x20 - x20, f.y20 - y20) * sc, sa = std::hypot(xi - x10, yi - y10) * sc, sb = std::hypot(xi - x20, yi - y20) * sc; const bool moving = sb > 0.5;
                e10.push_back(ea); e20.push_back(eb); s20.push_back(sb); if (moving) { e20m.push_back(eb); s20m.push_back(sb); e10m.push_back(ea); s10m.push_back(sa); } else { e20f.push_back(eb); e10f.push_back(ea); } } }
            auto med = [](std::vector<double> v) { if (v.empty()) return -1.0; std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end()); return v[v.size() / 2]; }; auto mean = [](const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return v.empty() ? -1.0 : s / double(v.size()); };
            ImGui::SeparatorText("Quality of the forecast over the last 10 s (error in degrees)"); ImGui::Text("forecast available for %d of %d samples", valid, total);
            if (ImGui::BeginTable("fq", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchProp)) { for (const char* h : {"samples", "n", "forecast +10 ms", "forecast +20 ms", "'stays where it is' +20 ms"}) ImGui::TableSetupColumn(h); ImGui::TableHeadersRow();
                auto row = [&](const char* name, size_t cnt, double m10, double m20, double ms) { ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::TextUnformatted(name); ImGui::TableNextColumn(); ImGui::Text("%zu", cnt); ImGui::TableNextColumn(); if (m10 >= 0) ImGui::Text("%.3f", m10); else ImGui::TextDisabled("-"); ImGui::TableNextColumn(); if (m20 >= 0) ImGui::Text("%.3f", m20); else ImGui::TextDisabled("-"); ImGui::TableNextColumn(); if (ms >= 0) ImGui::Text("%.3f", ms); else ImGui::TextDisabled("-"); };
                row("all (median)", e20.size(), med(e10), med(e20), med(s20)); row("fixation (median)", e20f.size(), med(e10f), med(e20f), -1.0); row("during movements > 0.5 deg in 20 ms (mean)", e20m.size(), mean(e10m), mean(e20m), mean(s20m)); ImGui::EndTable(); }
            ImGui::TextDisabled("A forecast is useful where it beats 'stays where it is': during movements. In fixation there is little to gain and the two are close.");
        }
        if (ImGui::TreeNode("Live state for MATLAB / Psychtoolbox and PsychoPy (the session file is the interface)")) {
            const std::string path = gui::Settings::instance().get("acq.last_session", "session.eyelive"); ImGui::TextWrapped("While a session records, the first 512 bytes of the session file hold a header and a 256-byte state block at offset 256: the last gaze sample, the speed, the event counters, the last event and the latest forecast (position 20 ms ahead and its uncertainty). Map the file read-only and read it with the seqlock protocol; no driver, no TTL. Session file of the last start:");
            ImGui::TextColored(ImVec4(0.6f, 0.85f, 1, 1), "%s", path.c_str());
            ImGui::TextUnformatted("MATLAB:  m = memmapfile('<session file>', 'Offset', 256, 'Format', 'uint8', 'Repeat', 256, 'Writable', false);\n         s = eyelive_state(m.Data);      % cpp/app/readers/matlab/eyelive_state.m\nPython:  from eyelive_reader import StateReader      # cpp/app/readers/python/eyelive_reader.py\n         s = StateReader('<session file>').latest()   # x, y, speed_deg_s, event_counter, pred_x, pred_y, ...");
            ImGui::TreePop(); }
    }
 private:
    std::vector<ForecastSample> fc_, tmp_; gui::TimeView view_{0, 1}; float span_ = 1.5f; bool show10_ = false, band_ = true, live_ = false;
};
static TabRegistrar reg([] { return std::make_unique<PredictTab>(); });
}  // namespace
}  // namespace eyelive

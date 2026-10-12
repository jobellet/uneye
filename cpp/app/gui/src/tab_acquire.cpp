// Acquire tab (order 10): source, rate, duration, units, session file, detector at start, Start / Stop with the error text of start().
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <cstdlib>
#include "imgui.h"
#include "eyelive/gui/settings.hpp"
#include "eyelive/gui/tab_acquire.hpp"
#include "eyelive/gui/widgets.hpp"
#include "eyelive/gui_tab.hpp"

namespace eyelive {
namespace {
using gui::Settings;
std::string default_session_path() {
    const char* h = std::getenv("HOME"); if (!h) h = std::getenv("USERPROFILE"); const std::time_t t = std::time(nullptr); char b[48]; std::strftime(b, sizeof b, "session_%Y%m%d_%H%M%S.eyelive", std::localtime(&t));
    return std::string(h ? h : ".") + "/EyeLive/" + b;
}

class AcquireTab final : public ITab {
 public:
    AcquireTab() {
        auto& s = Settings::instance(); source_ = int(s.get_number("acq.source", 0)); std::snprintf(addr_, sizeof addr_, "%s", s.get("acq.addr", "127.0.0.1").c_str()); port_ = int(s.get_number("acq.port", 5005)); std::snprintf(file_, sizeof file_, "%s", s.get("acq.file", "").c_str());
        speed_ = float(s.get_number("acq.speed", 1.0)); rate_ = float(s.get_number("acq.rate", 0)); minutes_ = float(s.get_number("acq.minutes", 60)); units_ = int(s.get_number("acq.units", 0)); ppd_ = float(s.get_number("acq.ppd", 36)); width_cm_ = float(s.get_number("acq.width", 53)); dist_cm_ = float(s.get_number("acq.dist", 57)); res_x_ = float(s.get_number("acq.resx", 1920));
        std::snprintf(path_, sizeof path_, "%s", default_session_path().c_str()); sim_rate_ = float(s.get_number("acq.simrate", 1000));
    }
    const char* title() const override { return "Acquire"; }
    int order() const override { return 10; }
    bool wants_redraw() const override { return running_; }
    void draw(IEngine& engine) override {
        const EngineStats st = engine.stats(); running_ = st.running; ImGui::BeginDisabled(running_);
        ImGui::SeparatorText("Source"); ImGui::RadioButton("Simulator", &source_, 0); ImGui::SameLine(); ImGui::RadioButton("File replay", &source_, 1); ImGui::SameLine(); ImGui::RadioButton("UDP stream", &source_, 2);
        if (source_ == 0) { ImGui::SetNextItemWidth(160); ImGui::InputFloat("sampling rate (Hz)", &sim_rate_, 0, 0, "%.0f"); ImGui::SetNextItemWidth(160); ImGui::InputFloat("speed (1 = real time, 0 = as fast as possible)", &speed_, 0, 0, "%.1f"); }
        if (source_ == 1) { ImGui::SetNextItemWidth(520); ImGui::InputText("CSV file (t,x,y[,valid])", file_, sizeof file_); ImGui::SetNextItemWidth(160); ImGui::InputFloat("speed (1 = real time, 0 = as fast as possible)", &speed_, 0, 0, "%.1f"); }
        if (source_ == 2) { ImGui::SetNextItemWidth(220); ImGui::InputText("bind address", addr_, sizeof addr_); ImGui::SetNextItemWidth(120); ImGui::InputInt("UDP port", &port_, 0, 0); ImGui::TextDisabled("protocol: docs/eyelive/ (binary v1 \"EYE1\" or text lines \"t x y [valid]\")"); }
        ImGui::SeparatorText("Sampling rate and duration"); ImGui::SetNextItemWidth(160); ImGui::InputFloat("declared rate (Hz, 0 = measure it)", &rate_, 0, 0, "%.0f"); ImGui::SetNextItemWidth(160); ImGui::InputFloat("planned duration (minutes)", &minutes_, 0, 0, "%.1f");
        const double rate_for_size = rate_ > 0 ? rate_ : (source_ == 0 ? sim_rate_ : 1000.0); const std::string sizing = gui::sizing_text(rate_for_size, double(minutes_) * 60.0); ImGui::Text("preallocation: %s%s", sizing.c_str(), rate_ > 0 ? "" : " (rate assumed; it is measured at start)");
        refresh_space(); if (space_bytes_ > 0) { const double need = double(recorder_sizing(rate_for_size, double(minutes_) * 60.0).bytes); ImGui::Text("free disk space: %s", gui::format_bytes(space_bytes_).c_str()); if (need > space_bytes_) ImGui::TextColored(ImVec4(1, 0.4f, 0.3f, 1), "not enough free space for this session"); }
        ImGui::SeparatorText("Units of the stream"); ImGui::RadioButton("degrees", &units_, 0); ImGui::SameLine(); ImGui::RadioButton("pixels, pixels per degree known", &units_, 1); ImGui::SameLine(); ImGui::RadioButton("pixels, screen geometry", &units_, 2); ImGui::SameLine(); ImGui::RadioButton("pixels, find it automatically", &units_, 3);
        if (units_ == 1) { ImGui::SetNextItemWidth(160); ImGui::InputFloat("pixels per degree", &ppd_, 0, 0, "%.2f"); }
        if (units_ == 2) { ImGui::SetNextItemWidth(120); ImGui::InputFloat("screen width (cm)", &width_cm_, 0, 0, "%.1f"); ImGui::SameLine(); ImGui::SetNextItemWidth(120); ImGui::InputFloat("distance (cm)", &dist_cm_, 0, 0, "%.1f"); ImGui::SameLine(); ImGui::SetNextItemWidth(120); ImGui::InputFloat("horizontal resolution (px)", &res_x_, 0, 0, "%.0f"); const double d = gui::deg_per_pixel_from_geometry(width_cm_, dist_cm_, res_x_); if (d > 0) ImGui::Text("= %.2f pixels per degree", 1.0 / d); }
        if (units_ == 3) ImGui::TextWrapped("The scale is estimated from the first minute of data (the network and the unit-free detector must agree) and shown in the Log tab; an approximate value is enough for detection. You can override it later by choosing 'pixels per degree known'.");
        ImGui::SeparatorText("Session"); ImGui::SetNextItemWidth(620); ImGui::InputText("session file", path_, sizeof path_); const auto dets = detectors(engine); ImGui::SetNextItemWidth(420);
        if (ImGui::BeginCombo("detector at start", current_name(dets).c_str())) { for (size_t i = 0; i < dets.size(); ++i) if (ImGui::Selectable(dets[i].name.c_str(), dets[i].id == detector_)) detector_ = dets[i].id; ImGui::EndCombo(); }
        ImGui::EndDisabled(); ImGui::Spacing();
        if (!running_) { if (ImGui::Button("Start", ImVec2(140, 0))) start(engine); } else if (ImGui::Button("Stop", ImVec2(140, 0))) engine.stop();
        if (!error_.empty()) { ImGui::SameLine(); ImGui::TextColored(ImVec4(1, 0.45f, 0.35f, 1), "%s", error_.c_str()); }
        if (running_ || st.samples > 0) { ImGui::SeparatorText("Stream"); ImGui::Text("rate %.2f Hz (standard %.0f) | %llu samples | dropped packets %llu | gaps %llu | recorded %s of %s", st.rate_hz_measured, st.rate_hz_standard, (unsigned long long)st.samples, (unsigned long long)st.dropped_packets, (unsigned long long)st.gaps, gui::format_time(st.recorded_s).c_str(), gui::format_time(st.capacity_s).c_str()); }
    }
 private:
    std::vector<DetectorInfo> detectors(IEngine& e) { const auto now = std::chrono::steady_clock::now(); if (dets_.empty() || now - dets_time_ > std::chrono::seconds(3)) { dets_ = e.available_detectors(); dets_time_ = now; } return dets_; }
    std::string current_name(const std::vector<DetectorInfo>& d) const { for (auto& i : d) if (i.id == detector_) return i.name; return detector_; }
    void refresh_space() { const auto now = std::chrono::steady_clock::now(); if (now - space_time_ < std::chrono::seconds(5)) return; space_time_ = now; std::error_code ec; auto dir = std::filesystem::path(path_).parent_path(); while (!dir.empty() && !std::filesystem::exists(dir, ec)) dir = dir.parent_path(); if (dir.empty()) dir = "."; const auto sp = std::filesystem::space(dir, ec); space_bytes_ = ec ? 0.0 : double(sp.available); }
    void start(IEngine& engine) {
        SessionConfig c; c.source.kind = source_ == 0 ? "sim" : (source_ == 1 ? "file" : "udp"); c.source.bind_address = addr_; c.source.port = port_; c.source.path = file_; c.source.speed = speed_; c.source.rate_hz = source_ == 0 ? sim_rate_ : (rate_ > 0 ? rate_ : 1000.0); c.session_path = path_; c.duration_s = double(minutes_) * 60.0; c.rate_hz = rate_ > 0 ? rate_ : 0;
        if (source_ == 0) c.rate_hz = sim_rate_; c.detector_id = detector_; c.n_classes = 3;
        if (units_ == 0) c.deg_per_unit = 1.0; else if (units_ == 1) c.deg_per_unit = ppd_ > 0 ? 1.0 / ppd_ : 0.0; else if (units_ == 2) c.deg_per_unit = gui::deg_per_pixel_from_geometry(width_cm_, dist_cm_, res_x_); else c.deg_per_unit = 0.0;
        if (units_ != 0 && units_ != 3 && c.deg_per_unit <= 0) { error_ = "the pixels-per-degree value or the screen geometry is not valid"; return; }
        std::error_code ec; std::filesystem::create_directories(std::filesystem::path(path_).parent_path(), ec); error_.clear(); std::string err; if (!engine.start(c, &err)) { error_ = err; return; }
        auto& s = Settings::instance(); s.set_number("acq.source", source_); s.set("acq.addr", addr_); s.set_number("acq.port", port_); s.set("acq.file", file_); s.set_number("acq.speed", speed_); s.set_number("acq.rate", rate_); s.set_number("acq.minutes", minutes_); s.set_number("acq.units", units_); s.set_number("acq.ppd", ppd_); s.set_number("acq.width", width_cm_); s.set_number("acq.dist", dist_cm_); s.set_number("acq.resx", res_x_); s.set_number("acq.simrate", sim_rate_); s.set("acq.last_session", c.session_path);
        std::snprintf(path_, sizeof path_, "%s", default_session_path().c_str());        // the next session gets a new name
    }
    int source_ = 0, port_ = 5005, units_ = 0; char addr_[64] = "127.0.0.1", file_[512] = "", path_[512] = ""; float speed_ = 1, rate_ = 0, minutes_ = 60, ppd_ = 36, width_cm_ = 53, dist_cm_ = 57, res_x_ = 1920, sim_rate_ = 1000; bool running_ = false; std::string error_, detector_ = "hmm_accel";
    std::vector<DetectorInfo> dets_; std::chrono::steady_clock::time_point dets_time_{}, space_time_{}; double space_bytes_ = 0;
};
static TabRegistrar reg([] { return std::make_unique<AcquireTab>(); });
}  // namespace
}  // namespace eyelive

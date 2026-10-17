#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <set>
#include "eyelive/export.hpp"
#include "eyelive/gui/ui_logic.hpp"
#include "eyelive/net_detectors.hpp"

namespace eyelive { namespace gui {
namespace fs = std::filesystem;

const char* mode_name(Mode m) { switch (m) { case Mode::Record: return "Record"; case Mode::Review: return "Review"; case Mode::Analyze: return "Analyze"; case Mode::Help: return "Help"; case Mode::Advanced: return "Advanced"; } return "Record"; }
Mode mode_from_name(const std::string& s, Mode fallback) { for (Mode m : {Mode::Record, Mode::Review, Mode::Analyze, Mode::Help, Mode::Advanced}) if (s == mode_name(m)) return m; return fallback; }
bool tab_in_mode(Mode m, const std::string& t) {
    if (m == Mode::Advanced) return true;
    static const std::set<std::string> rec = {"Setup", "Live", "Triggers"}, rev = {"Review", "Detection", "Train"}, ana = {"Predict", "Export", "Log"}, hlp = {"Help"};
    switch (m) { case Mode::Record: return rec.count(t) > 0; case Mode::Review: return rev.count(t) > 0; case Mode::Analyze: return ana.count(t) > 0; case Mode::Help: return hlp.count(t) > 0; default: return true; }
}

// ---- wizard
const char* WizardState::step_title(int s) { static const char* t[] = {"1. Your tracker", "2. The signal", "3. The screen", "4. What you want", "5. Name"}; return s >= 0 && s <= 4 ? t[s] : ""; }
bool WizardState::next() {
    errors.clear();
    if (step == Tracker) { if (tracker_choice == 2 && file.empty()) errors.push_back("choose the file to replay"); }
    else if (step == Signal) { if (tracker_choice == 1) { if (port < 1 || port > 65535) errors.push_back("the port must be between 1 and 65535"); if (bind.empty()) errors.push_back("the address to listen on is empty (127.0.0.1 = this computer only, 0.0.0.0 = any computer)"); } }
    else if (step == Screen) { if (!(width_cm > 0 && distance_cm > 0 && res_x > 0)) errors.push_back("give the screen width, the viewing distance and the horizontal resolution"); else if (distance_cm < 20 || distance_cm > 300) errors.push_back("a viewing distance of " + std::to_string(int(distance_cm)) + " cm is not plausible"); }
    else if (step == Goals) { if (want_triggers && triggers_file.empty()) errors.push_back("choose the trigger rules file, or untick the triggers"); else if (want_triggers) { std::vector<TriggerRule> r; std::string e; if (!load_trigger_rules(triggers_file, r, &e)) errors.push_back(e); } }
    else if (step == Name) { if (name.empty() || name.find_first_not_of(" \t") == std::string::npos) errors.push_back("give the profile a name"); }
    if (!errors.empty()) return false;
    ++step; if (step == Screen && units_choice != 1) ++step; return true;                     // no screen question unless the tracker sends pixels
}
bool WizardState::back() { errors.clear(); if (step == Tracker || step == Done) return false; --step; if (step == Screen && units_choice != 1) --step; return true; }
DeviceProfile WizardState::to_profile() const {
    DeviceProfile p; p.name = name; p.tracker = tracker_choice == 0 ? "simulated eye" : (tracker_label.empty() ? (tracker_choice == 2 ? "recording" : "generic") : tracker_label);
    p.source.kind = tracker_choice == 0 ? "sim" : (tracker_choice == 1 ? "udp" : "file"); p.source.bind_address = bind; p.source.port = port; p.source.path = file; p.source.speed = 1.0;
    static const double rates[] = {0, 500, 1000, 2000}; p.rate_hz = tracker_choice == 0 ? 0 : rates[std::clamp(rate_choice, 0, 3)]; p.source.rate_hz = tracker_choice == 0 ? 1000 : (p.rate_hz > 0 ? p.rate_hz : 1000);
    p.units = tracker_choice == 0 ? "degrees" : (units_choice == 0 ? "degrees" : (units_choice == 1 ? "pixels" : "auto")); if (p.units == "pixels") { p.screen.width_cm = width_cm; p.screen.height_cm = height_cm; p.screen.distance_cm = distance_cm; p.screen.res_x = res_x; p.screen.res_y = res_y; }
    p.triggers_file = want_triggers ? triggers_file : ""; p.landing_model = landing_choice == 1 ? "landing_mlp_gazebase.bin" : (landing_choice == 2 ? "landing_mlp_drone.bin" : ""); p.duration_min = tracker_choice == 0 ? 10 : 60; return p;
}

// ---- controller
UiController::UiController(Settings& s, std::string profile_dir, std::string session_dir) : settings_(s), profile_dir_(std::move(profile_dir)), session_dir_(std::move(session_dir)) {
    if (session_dir_.empty()) { const char* h = std::getenv("HOME"); if (!h) h = std::getenv("USERPROFILE"); session_dir_ = s.get("ui.session_dir", std::string(h ? h : ".") + "/EyeLive"); }
    mode_ = mode_from_name(s.get("ui.mode", "Record")); last_session_ = s.get("ui.last_session", ""); reload_profiles(); std::string e; const std::string want = s.get("ui.profile", ""); profile_ = builtin_profiles()[0];
    if (want.empty() || !select_profile(want, &e)) { if (!cache_.empty()) select_profile(cache_[0].key, &e); }
}
void UiController::reload_profiles() {
    cache_.clear(); std::set<std::string> seen;
    for (const auto& key : list_profiles(profile_dir_)) { DeviceProfile p; std::string e; const bool ok = load_profile((fs::path(profile_dir_) / (key + ".profile")).string(), p, &e); cache_.push_back({key, ok && !p.name.empty() ? p.name : key, false}); seen.insert(key); }
    for (const auto& b : builtin_profiles()) if (!seen.count(fs::path(profile_file_name(b.name)).stem().string())) cache_.push_back({b.name, b.name, true});
}
static bool safe_key(const std::string& k) { return !k.empty() && k.find('/') == std::string::npos && k.find('\\') == std::string::npos && k.find("..") == std::string::npos; }      // a profile key is a file name, never a path
bool UiController::select_profile(const std::string& key, std::string* error) {
    if (!safe_key(key)) { if (error) *error = "invalid profile name"; return false; }
    DeviceProfile p; std::error_code ec; const fs::path saved = fs::path(profile_dir_) / (key + ".profile");
    if (fs::exists(saved, ec)) { if (!load_profile(saved.string(), p, error)) return false; }
    else { bool found = false; for (const auto& b : builtin_profiles()) if (b.name == key) { p = b; found = true; } if (!found) { if (error) *error = "no profile '" + key + "'"; return false; } }
    profile_ = p; settings_.set("ui.profile", key); return true;
}
bool UiController::save_profile(std::string* error) { std::vector<std::string> pr; if (!profile_problems(profile_, &pr)) { if (error) *error = pr.empty() ? "invalid profile" : pr[0]; return false; } if (!save_profile_to(error)) return false; settings_.set("ui.profile", fs::path(profile_file_name(profile_.name)).stem().string()); reload_profiles(); return true; }
bool UiController::delete_profile(const std::string& name, std::string* error) {
    if (!safe_key(name)) { if (error) *error = "invalid profile name"; return false; }
    std::error_code ec; const std::string path = (fs::path(profile_dir_) / (name + ".profile")).string(); if (!fs::exists(path, ec)) { if (error) *error = "'" + name + "' is a built-in profile or does not exist"; return false; }
    if (!fs::remove(path, ec) || ec) { if (error) *error = "cannot delete " + path; return false; } if (fs::path(profile_file_name(profile_.name)).stem().string() == name) { profile_ = builtin_profiles()[0]; settings_.set("ui.profile", profile_.name); } reload_profiles(); return true;
}
bool UiController::wizard_needed() const { return settings_.get("ui.wizard_done", "0") != "1" && list_profiles(profile_dir_).empty(); }
bool UiController::finish_wizard(const WizardState& w, std::string* error) {
    DeviceProfile p = w.to_profile(); std::vector<std::string> pr; if (!profile_problems(p, &pr)) { if (error) *error = pr.empty() ? "invalid profile" : pr[0]; return false; }
    if (!eyelive::save_profile(profile_dir_, p, error)) return false; profile_ = p; settings_.set("ui.profile", fs::path(profile_file_name(p.name)).stem().string()); settings_.set("ui.wizard_done", "1"); reload_profiles(); mode_ = Mode::Record; settings_.set("ui.mode", "Record"); return true;
}
void UiController::set_mode(Mode m) { mode_ = m; settings_.set("ui.mode", mode_name(m)); }

std::string UiController::next_session_path() const {
    const std::time_t t = std::time(nullptr); char b[32]; std::tm tmv{}; 
#ifdef _WIN32
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    std::strftime(b, sizeof b, "%Y%m%d_%H%M%S", &tmv); std::string stem = fs::path(profile_file_name(profile_.name)).stem().string(); const std::string base = (fs::path(session_dir_) / (stem + "_" + b)).string(); std::string path = base + ".eyelive"; std::error_code ec;
    for (int k = 2; fs::exists(path, ec) && k < 1000; ++k) path = base + "_" + std::to_string(k) + ".eyelive"; return path;     // never overwrites a session
}
bool UiController::start(IEngine& engine, std::string* error) {
    std::vector<std::string> pr; if (!profile_problems(profile_, &pr)) { if (error) *error = pr.empty() ? "the profile is not valid" : pr[0]; return false; }
    std::vector<TriggerRule> rules; if (!profile_.triggers_file.empty() && !load_trigger_rules(profile_.triggers_file, rules, error)) return false;       // a wrong rules file stops the start, not the middle of the experiment
    std::error_code ec; fs::create_directories(session_dir_, ec); if (ec) { if (error) *error = "cannot create the session folder " + session_dir_ + ": " + ec.message(); return false; }
    const std::string path = next_session_path(); SessionConfig c = session_config(profile_, path); c.command_line = "eyelive (graphical interface), profile '" + profile_.name + "'";
    engine.set_trigger_rules(rules); if (!engine.start(c, error)) return false;
    running_ = true; started_ = std::chrono::steady_clock::now(); set_last_session(path); return true;
}
void UiController::stop(IEngine& engine) { engine.stop(); running_ = false; }
double UiController::elapsed_s() const { return running_ ? std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count() : 0.0; }
Health UiController::health(IEngine& engine) const { return assess_health(engine.stats(), elapsed_s()); }

std::string UiController::default_export_path(const std::string& s, ExportKind k) {
    const std::string base = s.size() > 8 && s.compare(s.size() - 8, 8, ".eyelive") == 0 ? s.substr(0, s.size() - 8) : s;
    switch (k) { case ExportKind::EventsCsv: return base + ".events.csv"; case ExportKind::Bids: return base + "_bids"; case ExportKind::Mat: return base + ".mat"; case ExportKind::Report: return base + ".report.html"; } return base;
}
bool UiController::export_session(const std::string& path, ExportKind k, const std::string& out, std::string* error) const {
    std::error_code ec; if (path.empty() || !fs::exists(path, ec)) { if (error) *error = "choose a session file (nothing was recorded yet?)"; return false; }
    auto rec = Recorder::open_readonly(path, error); if (!rec) return false;
    switch (k) { case ExportKind::EventsCsv: return export_events_csv(*rec, out, error); case ExportKind::Bids: return export_bids(*rec, out, error); case ExportKind::Mat: return export_mat(*rec, out, error); case ExportKind::Report: return write_report(summarize(*rec), out, error); } return false;
}
std::vector<std::string> UiController::landing_models() const {
    std::vector<std::string> v; std::error_code ec; const std::string dir = models_dir(); if (dir.empty()) return v;
    for (const auto& e : fs::directory_iterator(dir, ec)) { const std::string n = e.path().filename().string(); if (n.rfind("landing_mlp_", 0) == 0 && e.path().extension() == ".bin") v.push_back(n); } std::sort(v.begin(), v.end()); return v;
}

}}  // namespace eyelive::gui

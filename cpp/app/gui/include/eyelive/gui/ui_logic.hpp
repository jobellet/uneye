// Logic of the simplified interface (no ImGui here: it is tested without a window). Three jobs: the four-question wizard of the first launch, the modes that group the tabs
// (Record / Review / Analyze / Help, Advanced shows everything), and the controller behind the big Start / Stop button (profiles, session file naming, trigger rules, health light, checks, export of the last session).
#pragma once
#include <chrono>
#include <string>
#include <vector>
#include "eyelive/engine.hpp"
#include "eyelive/gui/settings.hpp"
#include "eyelive/setup.hpp"

namespace eyelive { namespace gui {

enum class Mode { Record, Review, Analyze, Help, Advanced };
const char* mode_name(Mode m);
Mode mode_from_name(const std::string& s, Mode fallback = Mode::Record);
bool tab_in_mode(Mode m, const std::string& tab_title);                    // Setup, Live, Triggers | Review, Detection, Train | Predict, Export, Log | Help | everything

// ---- first-launch wizard: tracker, signal, screen (only for pixels), what to do, name
struct WizardState {
    enum Step { Tracker = 0, Signal = 1, Screen = 2, Goals = 3, Name = 4, Done = 5 };
    int step = Tracker;
    int tracker_choice = 0;                  // 0 simulation (try the software), 1 tracker streaming UDP to this computer, 2 replay a recorded file
    std::string tracker_label;               // "EyeLink 1000 Plus", ...
    std::string bind = "127.0.0.1"; int port = 5005; std::string file;
    int rate_choice = 0;                     // 0 measure it, 1 = 500 Hz, 2 = 1000 Hz, 3 = 2000 Hz
    int units_choice = 0;                    // 0 degrees, 1 screen pixels, 2 I do not know (estimate)
    double width_cm = 53, height_cm = 30, distance_cm = 60; int res_x = 1920, res_y = 1080;
    bool want_triggers = false; std::string triggers_file; int landing_choice = 0;   // 0 tables only, 1 GazeBase-type network (target steps), 2 free-viewing network (videos)
    std::string name = "My experiment";
    std::vector<std::string> errors;         // why next() refused the current step
    bool next();                             // validates the current step, moves on (the screen step is skipped unless units are pixels); false and errors filled when invalid
    bool back();
    bool finished() const { return step == Done; }
    DeviceProfile to_profile() const;        // valid after finished()
    static const char* step_title(int step);
};

// ---- controller
enum class ExportKind { EventsCsv, Bids, Mat, Report };
struct ProfileEntry { std::string key;        // what select_profile() takes: the file name without extension, or the name of a built-in profile
                      std::string label;      // name shown to the user
                      bool builtin = false; };

class UiController {
 public:
    UiController(Settings& settings, std::string profile_dir, std::string session_dir);
    // profiles
    const std::vector<ProfileEntry>& profiles() const { return cache_; }   // saved profiles first, then the built-in ones
    void reload_profiles();
    bool select_profile(const std::string& name, std::string* error);
    const DeviceProfile& profile() const { return profile_; }
    void set_profile(const DeviceProfile& p) { profile_ = p; }             // edited in the Setup tab, not saved yet
    bool save_profile(std::string* error);                                 // saves the current profile (built-in names are saved under their own name as a user profile)
    bool delete_profile(const std::string& name, std::string* error);
    bool wizard_needed() const;
    bool finish_wizard(const WizardState& w, std::string* error);         // saves the profile, selects it, remembers that the wizard was done
    // modes
    Mode mode() const { return mode_; }
    void set_mode(Mode m);
    // session
    bool start(IEngine& engine, std::string* error);
    void stop(IEngine& engine);
    std::string next_session_path() const;
    const std::string& last_session() const { return last_session_; }
    void set_last_session(const std::string& p) { last_session_ = p; settings_.set("ui.last_session", p); }
    double elapsed_s() const;
    Health health(IEngine& engine) const;
    std::vector<CheckResult> checks() const { return run_checks(profile_, session_dir_); }
    bool export_session(const std::string& session_path, ExportKind kind, const std::string& out_path, std::string* error) const;
    static std::string default_export_path(const std::string& session_path, ExportKind kind);
    const std::string& session_dir() const { return session_dir_; }
    const std::string& profile_dir() const { return profile_dir_; }
    std::vector<std::string> landing_models() const;                       // landing_mlp_*.bin of the models folder
 private:
    bool save_profile_to(std::string* error) { return eyelive::save_profile(profile_dir_, profile_, error); }
    std::vector<ProfileEntry> cache_;
    Settings& settings_; std::string profile_dir_, session_dir_, last_session_; DeviceProfile profile_; Mode mode_ = Mode::Record; bool running_ = false; std::chrono::steady_clock::time_point started_;
};

}}  // namespace eyelive::gui

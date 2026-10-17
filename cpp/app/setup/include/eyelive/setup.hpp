// Device profiles, screen geometry, environment checks and the health light of the GUI. A profile is a small text file (~/.eyelive/profiles/<name>.profile) that holds everything a laboratory sets once
// (tracker, port, rate, screen size and distance, detector, landing network, trigger rules) so that starting a session is one click.
#pragma once
#include <map>
#include <string>
#include <vector>
#include "eyelive/engine.hpp"

namespace eyelive {

struct ScreenGeometry {
    double width_cm = 0, height_cm = 0, distance_cm = 0;      // viewing distance from the eye to the screen centre; height optional (used only to check the pixel aspect)
    int res_x = 0, res_y = 0;
    bool complete() const { return width_cm > 0 && distance_cm > 0 && res_x > 0; }
};
// pixels per degree at the screen centre (the stream is converted linearly, the error is below 1 % up to 8 deg and 4 % at 15 deg); 0 when the geometry is incomplete
double pixels_per_degree(const ScreenGeometry& g);
double degrees_per_pixel(const ScreenGeometry& g);              // 0 when incomplete

struct DeviceProfile {
    std::string name = "default";
    std::string tracker = "generic";                          // free label shown in the reports: "EyeLink 1000 Plus", "Tobii Pro Spectrum", "Pupil Core", ...
    SourceSpec source;                                         // kind (udp | file | sim), bind address, port, replay file, simulation rate
    double rate_hz = 0;                                        // 0 = measured from the first 500 ms
    std::string units = "degrees";                             // what the tracker sends: "degrees", "pixels" (converted with the geometry below, the file keeps the pixels) or "auto" (the scale is estimated from the data)
    ScreenGeometry screen;
    std::string detector_id = "hmm_accel";
    std::string landing_model;                                 // "" = tables only; a file name of cpp/models (landing_mlp_gazebase.bin) or a path
    std::string triggers_file;
    double duration_min = 60;                                  // planned duration: sizes the preallocated session file
    std::string notes;
    std::map<std::string, std::string> extra;                  // unknown keys of the file are kept
};

std::string profile_file_name(const std::string& name);       // "Lab A / EyeLink" -> "Lab_A_EyeLink.profile" (letters, digits, . _ - only; never a path)
std::string serialize_profile(const DeviceProfile& p);
bool parse_profile(const std::string& text, DeviceProfile& out, std::string* error);       // error: "line 7: bad number for rate_hz"
bool profile_problems(const DeviceProfile& p, std::vector<std::string>* problems);         // true = usable; problems may hold warnings even then (prefix "warning: ")
bool save_profile(const std::string& dir, const DeviceProfile& p, std::string* error);
bool load_profile(const std::string& path, DeviceProfile& out, std::string* error);
std::vector<std::string> list_profiles(const std::string& dir);                            // profile names (file name without extension), sorted
std::string default_profile_dir();                             // ~/.eyelive/profiles
std::vector<DeviceProfile> builtin_profiles();                 // simulation, generic UDP (degrees, pixels), replay of a file

// SessionConfig of a profile: deg_per_unit = 1 (degrees), degrees per pixel of the geometry (pixels; the session file keeps the raw pixels) or 0 (auto: estimated from the data)
SessionConfig session_config(const DeviceProfile& p, const std::string& session_path);

// ---- trigger rules file (one rule per line, '#' starts a comment): name=sacc;cls=1;anchor=provisional;delay_ms=10;sink=udp:127.0.0.1:6000;payload=SACC {t} {amp}
bool parse_trigger_rule(const std::string& line, TriggerRule& out, std::string* error);        // strict: unknown keys, bad numbers, negative delays and bad sinks are errors
bool load_trigger_rules(const std::string& path, std::vector<TriggerRule>& out, std::string* error);   // error: "rules.txt:3: unknown key 'dely_ms'"

// ---- checks run before a session and by the Help tab
enum class CheckLevel { Ok, Warn, Fail };
struct CheckResult { std::string what; CheckLevel level = CheckLevel::Ok; std::string message, hint; };
bool udp_port_free(const std::string& bind_address, int port, std::string* error);
double free_disk_bytes(const std::string& dir);                // -1 when unknown
std::vector<CheckResult> run_checks(const DeviceProfile& p, const std::string& session_dir);

// ---- health light
enum class HealthLevel { Idle, Ok, Warn, Bad };
struct Health { HealthLevel level = HealthLevel::Idle; std::string message; };
Health assess_health(const EngineStats& s, double seconds_since_start);

}  // namespace eyelive

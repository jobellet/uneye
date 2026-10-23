// Logic of the simplified interface: modes, wizard, profiles, start / stop of a session through the controller (real engine, simulated tracker), export of the last session.
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include "eyelive/gui/ui_logic.hpp"
#include "eyelive/net_detectors.hpp"
using namespace eyelive;
using namespace eyelive::gui;
namespace fs = std::filesystem;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
static bool has(const std::vector<std::string>& v, const std::string& part) { for (auto& s : v) if (s.find(part) != std::string::npos) return true; return false; }

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : "."; set_models_dir(root + "/models");
    const std::string dir = (fs::temp_directory_path() / "eyelive_ui_logic_test").string(); fs::remove_all(dir); fs::create_directories(dir);
    // ---- modes
    for (Mode m : {Mode::Record, Mode::Review, Mode::Analyze, Mode::Help, Mode::Advanced}) CHECK(mode_from_name(mode_name(m)) == m, "mode name round trip");
    CHECK(mode_from_name("nonsense", Mode::Review) == Mode::Review, "unknown mode name");
    { const char* all[] = {"Setup", "Acquire", "Live", "Detection", "Review", "Train", "Triggers", "Predict", "Export", "Log", "Help"}; int shown_in_simple[11] = {0};
      for (int i = 0; i < 11; ++i) { CHECK(tab_in_mode(Mode::Advanced, all[i]), "advanced shows %s", all[i]); for (Mode m : {Mode::Record, Mode::Review, Mode::Analyze, Mode::Help}) shown_in_simple[i] += tab_in_mode(m, all[i]); }
      for (int i = 0; i < 11; ++i) CHECK(shown_in_simple[i] == (std::string(all[i]) == "Acquire" ? 0 : 1), "%s is in %d simple modes", all[i], shown_in_simple[i]); }
    CHECK(tab_in_mode(Mode::Record, "Live") && !tab_in_mode(Mode::Record, "Review") && tab_in_mode(Mode::Review, "Train") && tab_in_mode(Mode::Analyze, "Export") && tab_in_mode(Mode::Help, "Help") && !tab_in_mode(Mode::Help, "Log"), "mode contents");
    // ---- wizard
    { WizardState w; CHECK(w.step == WizardState::Tracker && std::string(WizardState::step_title(0)) == "1. Your tracker", "start"); w.tracker_choice = 2; CHECK(!w.next() && has(w.errors, "file") && w.step == WizardState::Tracker, "replay without a file");
      w.file = dir + "/rec.csv"; CHECK(w.next() && w.step == WizardState::Signal, "to the signal step"); CHECK(w.back() && w.step == WizardState::Tracker && !w.back(), "back, and no back from the first step"); w.next();
      w.tracker_choice = 1; w.port = 0; CHECK(!w.next() && has(w.errors, "port"), "port 0"); w.port = 5005; w.bind = ""; CHECK(!w.next() && has(w.errors, "listen"), "empty address"); w.bind = "127.0.0.1"; w.units_choice = 0; CHECK(w.next() && w.step == WizardState::Goals, "degrees skip the screen step");
      CHECK(w.back() && w.step == WizardState::Signal, "back skips the screen step too"); w.units_choice = 1; CHECK(w.next() && w.step == WizardState::Screen, "pixels ask for the screen"); w.width_cm = 0; CHECK(!w.next() && has(w.errors, "width"), "no width"); w.width_cm = 53; w.distance_cm = 5; CHECK(!w.next() && has(w.errors, "plausible"), "distance 5 cm"); w.distance_cm = 60; CHECK(w.next() && w.step == WizardState::Goals, "screen ok");
      w.want_triggers = true; CHECK(!w.next() && has(w.errors, "rules"), "triggers without a file"); w.triggers_file = dir + "/none.txt"; CHECK(!w.next() && has(w.errors, "none.txt"), "missing rules file named in the error: %s", w.errors.empty() ? "" : w.errors[0].c_str());
      std::ofstream(dir + "/rules.txt") << "name=a;sink=stdout\n"; w.triggers_file = dir + "/rules.txt"; w.landing_choice = 2; CHECK(w.next() && w.step == WizardState::Name, "goals ok"); w.name = "  "; CHECK(!w.next() && has(w.errors, "name"), "blank name"); w.name = "Lab A"; w.rate_choice = 2; w.tracker_label = "EyeLink 1000 Plus";
      CHECK(w.next() && w.finished(), "finished"); const DeviceProfile p = w.to_profile(); std::vector<std::string> pr; CHECK(profile_problems(p, &pr), "the wizard builds a valid profile: %s", pr.empty() ? "" : pr[0].c_str());
      CHECK(p.name == "Lab A" && p.tracker == "EyeLink 1000 Plus" && p.source.kind == "udp" && p.source.port == 5005 && p.rate_hz == 1000 && p.units == "pixels" && p.screen.width_cm == 53 && p.triggers_file == dir + "/rules.txt" && p.landing_model == "landing_mlp_drone.bin", "profile fields");
      WizardState s; s.tracker_choice = 0; s.units_choice = 1; s.name = "Try"; for (int i = 0; i < 4; ++i) s.next(); const DeviceProfile q = s.to_profile(); CHECK(q.source.kind == "sim" && q.units == "degrees" && q.rate_hz == 0 && q.landing_model.empty() && q.triggers_file.empty(), "simulation profile ignores the screen answers"); }
    // ---- controller
    Settings& st = Settings::instance(); st.set_path(dir + "/settings.txt"); const std::string pdir = dir + "/profiles", sdir = dir + "/sessions";
    {
        UiController ui(st, pdir, sdir); CHECK(ui.wizard_needed(), "first launch needs the wizard"); CHECK(ui.profiles().size() == 4 && ui.profiles()[0].builtin && ui.mode() == Mode::Record, "built-in profiles %zu", ui.profiles().size());
        WizardState w; w.tracker_choice = 1; w.name = "Lab A"; w.units_choice = 0; for (int i = 0; i < 5; ++i) CHECK(w.next() || i >= 4, "wizard step %d", i); std::string err; CHECK(ui.finish_wizard(w, &err), "finish: %s", err.c_str()); CHECK(!ui.wizard_needed() && ui.profile().name == "Lab A" && fs::exists(pdir + "/Lab_A.profile"), "wizard saved the profile");
        CHECK(ui.profiles()[0].key == "Lab_A" && ui.profiles()[0].label == "Lab A" && !ui.profiles()[0].builtin, "saved profile listed first, with its real name"); WizardState bad; bad.tracker_choice = 1; bad.port = 70000; bad.name = "Bad"; CHECK(!ui.finish_wizard(bad, &err) && !err.empty() && !fs::exists(pdir + "/Bad.profile"), "invalid wizard result saved: %s", err.c_str());
        ui.set_mode(Mode::Review); st.flush();
    }
    {   // a new launch sees the saved state
        Settings::instance().set_path(dir + "/settings.txt"); UiController ui(Settings::instance(), pdir, sdir); CHECK(!ui.wizard_needed() && ui.mode() == Mode::Review && ui.profile().name == "Lab A", "state restored: mode %s profile %s", mode_name(ui.mode()), ui.profile().name.c_str());
        std::string err; CHECK(ui.select_profile("Simulation (demo)", &err) && ui.profile().source.kind == "sim", "select a built-in profile"); CHECK(!ui.select_profile("nothing", &err) && err.find("no profile") != std::string::npos, "unknown profile"); CHECK(!ui.select_profile("../../etc/passwd", &err) && err == "invalid profile name" && !ui.delete_profile("a/b", &err) && !ui.select_profile("", &err), "path-like profile keys refused");
        DeviceProfile e = ui.profile(); e.name = "Quick sim"; e.source.speed = 0; e.source.rate_hz = 1000; e.duration_min = 1; ui.set_profile(e); CHECK(ui.save_profile(&err) && fs::exists(pdir + "/Quick_sim.profile") && ui.profiles().size() == 6, "save an edited profile: %s", err.c_str());
        DeviceProfile inv = e; inv.source.kind = "udp"; inv.source.port = 0; ui.set_profile(inv); CHECK(!ui.save_profile(&err) && err.find("port") != std::string::npos, "invalid profile refused: %s", err.c_str()); ui.set_profile(e);
        CHECK(!ui.delete_profile("Simulation (demo)", &err), "a built-in profile cannot be deleted"); DeviceProfile tmp = e; tmp.name = "Temp"; ui.set_profile(tmp); CHECK(ui.save_profile(&err) && ui.delete_profile("Temp", &err) && !fs::exists(pdir + "/Temp.profile") && ui.profiles().size() == 6, "delete a saved profile: %s", err.c_str());
        // landing networks found in the models folder
        { const auto m = ui.landing_models(); bool g = false, d = false; for (auto& n : m) { g |= n == "landing_mlp_gazebase.bin"; d |= n == "landing_mlp_drone.bin"; } CHECK(g && d, "landing models listed (%zu)", m.size()); }
        // session names never collide
        ui.select_profile("Quick_sim", &err); const std::string a = ui.next_session_path(); CHECK(a.find(sdir + "/Quick_sim_") == 0 && a.size() > 20 && a.rfind(".eyelive") == a.size() - 8, "session path: %s", a.c_str()); fs::create_directories(sdir); std::ofstream(a) << "x"; const std::string b = ui.next_session_path(); CHECK(b != a && b.find("_2.eyelive") != std::string::npos, "second path: %s", b.c_str()); fs::remove(a);
        // refusals before anything starts
        auto eng = make_engine(); DeviceProfile u = e; u.source.kind = "udp"; u.source.port = 70000; ui.set_profile(u); CHECK(!ui.start(*eng, &err) && err.find("port") != std::string::npos && !eng->stats().running, "invalid profile starts nothing: %s", err.c_str());
        DeviceProfile t = e; t.triggers_file = dir + "/missing_rules.txt"; ui.set_profile(t); CHECK(!ui.start(*eng, &err) && err.find("missing_rules.txt") != std::string::npos && !eng->stats().running, "missing rules file: %s", err.c_str());
        std::ofstream(dir + "/bad_rules.txt") << "name=a;sink=nope\n"; t.triggers_file = dir + "/bad_rules.txt"; ui.set_profile(t); CHECK(!ui.start(*eng, &err) && err.find("bad_rules.txt:1") != std::string::npos, "bad rules file: %s", err.c_str());
        // a real session
        std::ofstream(dir + "/ok_rules.txt") << "name=sacc;cls=1;anchor=final;sink=log:" << dir << "/trig.log;payload=S {t}\n"; DeviceProfile r = e; r.triggers_file = dir + "/ok_rules.txt"; r.landing_model = "landing_mlp_drone.bin"; ui.set_profile(r);
        CHECK(ui.health(*eng).level == HealthLevel::Idle, "idle before the start"); CHECK(ui.start(*eng, &err), "start: %s", err.c_str()); CHECK(fs::path(ui.last_session()).parent_path() == fs::path(sdir) && ui.last_session().find("Quick_sim_") != std::string::npos, "last session %s", ui.last_session().c_str());
        for (int i = 0; i < 600 && eng->stats().samples < 20000; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(20)); const Health h = ui.health(*eng); CHECK(h.level == HealthLevel::Ok || h.level == HealthLevel::Warn, "health while recording: %s", h.message.c_str()); CHECK(ui.elapsed_s() > 0, "elapsed");
        ui.stop(*eng); CHECK(!eng->stats().running && ui.elapsed_s() == 0 && ui.health(*eng).level == HealthLevel::Idle, "stopped"); CHECK(fs::exists(ui.last_session()) && fs::exists(ui.last_session() + ".provenance.json") && fs::exists(ui.last_session() + ".report.html"), "session, journal and report written");
        { std::ifstream f(ui.last_session() + ".provenance.json"); std::stringstream ss; ss << f.rdbuf(); CHECK(ss.str().find("\"profile\": \"Quick sim\"") != std::string::npos && ss.str().find("landing_mlp_drone.bin") != std::string::npos, "the journal knows the profile and the landing network"); }
        { std::ifstream f(dir + "/trig.log"); CHECK(bool(f) && f.peek() != std::ifstream::traits_type::eof(), "the trigger rules of the profile fired"); }
        // exports
        for (ExportKind k : {ExportKind::EventsCsv, ExportKind::Mat, ExportKind::Report}) { const std::string o = UiController::default_export_path(ui.last_session(), k); CHECK(ui.export_session(ui.last_session(), k, o, &err) && fs::exists(o) && fs::file_size(o) > 100, "export kind %d: %s", int(k), err.c_str()); }
        { const std::string o = UiController::default_export_path(ui.last_session(), ExportKind::Bids); CHECK(ui.export_session(ui.last_session(), ExportKind::Bids, o, &err) && fs::exists(o + "_eyetrack.tsv") && fs::exists(o + "_events.tsv"), "bids export: %s", err.c_str()); }
        CHECK(!ui.export_session("", ExportKind::Report, dir + "/x.html", &err) && err.find("choose a session") != std::string::npos, "export without a session"); CHECK(!ui.export_session(ui.last_session(), ExportKind::Mat, "/nonexistent_dir/x.mat", &err), "unwritable export path");
        CHECK(UiController::default_export_path("/a/b/s.eyelive", ExportKind::Report) == "/a/b/s.report.html" && UiController::default_export_path("/a/b/s.eyelive", ExportKind::EventsCsv) == "/a/b/s.events.csv" && UiController::default_export_path("/a/b/s", ExportKind::Mat) == "/a/b/s.mat", "default export paths");
        // checks
        { bool any = false; for (auto& c : ui.checks()) any |= c.what == "disk space"; CHECK(any, "checks include the disk space"); }
    }
    fs::remove_all(dir); if (fails == 0) std::printf("ui logic: OK\n"); return fails ? 1 : 0;
}

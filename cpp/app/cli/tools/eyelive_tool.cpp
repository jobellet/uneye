// eyelive_tool: everything you do with a session AFTER the recording, without the GUI.
//   eyelive_tool info <session>                                   one-page summary of a session (same as the report)
//   eyelive_tool export <session> [--csv F] [--bids BASE] [--mat F] [--report F]      events CSV, BIDS-style TSV + JSON, MAT-file, report (.html or text)
//   eyelive_tool redetect <session|folder> --detector ID --out DIR [--recursive]      label the samples again with another detector / model; events CSV + report per session, the sessions are not modified
//   eyelive_tool check [--profile NAME|FILE] [--session-dir DIR]  the checks of the Help tab (port free, disk space, scale, ...)
//   eyelive_tool profiles [list | show NAME | new NAME --from BUILTIN]
// Exit code: 0 success, 1 failure (message on stderr), 2 usage.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include "eyelive/engine.hpp"
#include "eyelive/export.hpp"
#include "eyelive/net_detectors.hpp"
#include "eyelive/setup.hpp"
using namespace eyelive;
namespace fs = std::filesystem;

static int usage() { std::fprintf(stderr, "usage:\n  eyelive_tool info <session>\n  eyelive_tool export <session> [--csv F] [--bids BASE] [--mat F] [--report F]\n  eyelive_tool redetect <session|folder> --detector ID --out DIR [--recursive]\n  eyelive_tool check [--profile NAME|FILE] [--session-dir DIR]\n  eyelive_tool profiles [list | show NAME | new NAME --from BUILTIN]\n"); return 2; }
static int fail(const std::string& m) { std::fprintf(stderr, "error: %s\n", m.c_str()); return 1; }
static std::unique_ptr<Recorder> open_session(const std::string& path, std::string* err) { std::error_code ec; if (!fs::exists(path, ec)) { *err = "no such file: " + path; return nullptr; } return Recorder::open_readonly(path, err); }

static void set_models_dir_near(const char* argv0) {
    if (!models_dir().empty()) return; std::error_code ec; fs::path d = fs::absolute(argv0, ec).parent_path();
    for (int up = 0; up < 6 && !d.empty(); ++up, d = d.parent_path()) for (const char* cand : {"models", "cpp/models"}) if (fs::is_directory(d / cand, ec)) { set_models_dir((d / cand).string()); return; }
}

static int cmd_info(int argc, char** argv) {
    if (argc < 1) return usage(); std::string err; auto rec = open_session(argv[0], &err); if (!rec) return fail(err); std::fputs(report_text(summarize(*rec)).c_str(), stdout); return 0;
}

static int cmd_export(int argc, char** argv) {
    if (argc < 1) return usage(); std::string err, csv, bids, mat, rep; for (int i = 1; i < argc; ++i) { const std::string a = argv[i]; auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--csv") csv = next(); else if (a == "--bids") bids = next(); else if (a == "--mat") mat = next(); else if (a == "--report") rep = next(); else return usage(); }
    if (csv.empty() && bids.empty() && mat.empty() && rep.empty()) return fail("nothing to export: give --csv, --bids, --mat or --report");
    auto rec = open_session(argv[0], &err); if (!rec) return fail(err);
    if (!csv.empty() && !export_events_csv(*rec, csv, &err)) return fail(err); if (!bids.empty() && !export_bids(*rec, bids, &err)) return fail(err); if (!mat.empty() && !export_mat(*rec, mat, &err)) return fail(err); if (!rep.empty() && !write_report(summarize(*rec), rep, &err)) return fail(err);
    std::printf("exported %lld samples, %zu events\n", (long long)rec->size(), final_events(*rec).size()); return 0;
}

static int redetect_one(const std::string& path, const std::string& detector, const fs::path& out_dir) {
    std::string err; auto rec = open_session(path, &err); if (!rec) { std::fprintf(stderr, "%s: %s\n", path.c_str(), err.c_str()); return 1; }
    auto det = DetectorRegistry::instance().create(detector, &err); if (!det) { std::fprintf(stderr, "detector %s: %s\n", detector.c_str(), err.c_str()); return 1; }
    const SessionInfo si = rec->info(); det->reset(si); det->set_scale(si.deg_per_unit > 0 ? si.deg_per_unit : 1.0); std::vector<Event> ev; det->detect_all(*rec, ev);
    std::stable_sort(ev.begin(), ev.end(), [](const Event& a, const Event& b) { return a.onset < b.onset; }); std::string tag = detector; for (char& c : tag) if (!std::isalnum((unsigned char)c) && c != '-' && c != '_' && c != '+') c = '_';
    const std::string stem = (out_dir / fs::path(path).stem()).string() + "." + tag; if (!write_events_csv(ev, si, stem + ".events.csv", &err) || !write_report(summarize(*rec, ev, path + " (re-detected with " + detector + ")"), stem + ".report.html", &err)) { std::fprintf(stderr, "%s: %s\n", path.c_str(), err.c_str()); return 1; }
    size_t n = 0; for (auto& e : ev) n += !e.provisional; std::printf("%s: %zu events with %s -> %s.events.csv\n", path.c_str(), n, detector.c_str(), stem.c_str()); return 0;
}
static int cmd_redetect(int argc, char** argv) {
    if (argc < 1) return usage(); std::string detector, out; bool recursive = false; for (int i = 1; i < argc; ++i) { const std::string a = argv[i]; auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--detector") detector = next(); else if (a == "--out") out = next(); else if (a == "--recursive") recursive = true; else return usage(); }
    if (detector.empty() || out.empty()) return fail("--detector and --out are required"); std::error_code ec; fs::create_directories(out, ec); if (ec) return fail("cannot create " + out + ": " + ec.message());
    std::vector<std::string> files; if (fs::is_directory(argv[0], ec)) { auto add = [&](const fs::directory_entry& e) { if (e.is_regular_file(ec) && e.path().extension() == ".eyelive") files.push_back(e.path().string()); };
        if (recursive) for (const auto& e : fs::recursive_directory_iterator(argv[0], ec)) add(e); else for (const auto& e : fs::directory_iterator(argv[0], ec)) add(e); std::sort(files.begin(), files.end()); } else files.push_back(argv[0]);
    if (files.empty()) return fail(std::string("no .eyelive session in ") + argv[0]); int bad = 0; for (const auto& f : files) bad += redetect_one(f, detector, out) != 0; std::printf("%zu session(s), %d failed\n", files.size(), bad); return bad ? 1 : 0;
}

static bool find_profile(const std::string& what, DeviceProfile& p, std::string* err) {
    std::error_code ec; if (fs::exists(what, ec)) return load_profile(what, p, err); const std::string path = (fs::path(default_profile_dir()) / profile_file_name(what)).string(); if (fs::exists(path, ec)) return load_profile(path, p, err);
    for (const auto& b : builtin_profiles()) if (b.name == what) { p = b; return true; } *err = "no profile '" + what + "' (see: eyelive_tool profiles list)"; return false;
}
static int cmd_check(int argc, char** argv) {
    std::string prof = "Simulation (demo)", dir = "."; for (int i = 0; i < argc; ++i) { const std::string a = argv[i]; if (i + 1 >= argc) return usage(); if (a == "--profile") prof = argv[++i]; else if (a == "--session-dir") dir = argv[++i]; else return usage(); }
    DeviceProfile p; std::string err; if (!find_profile(prof, p, &err)) return fail(err); int bad = 0;
    for (const auto& r : run_checks(p, dir)) { std::printf("[%s] %s: %s%s%s\n", r.level == CheckLevel::Ok ? " ok " : (r.level == CheckLevel::Warn ? "warn" : "FAIL"), r.what.c_str(), r.message.c_str(), r.hint.empty() ? "" : "  -> ", r.hint.c_str()); bad += r.level == CheckLevel::Fail; }
    return bad ? 1 : 0;
}
static int cmd_profiles(int argc, char** argv) {
    const std::string sub = argc > 0 ? argv[0] : "list"; std::string err;
    if (sub == "list") { for (const auto& b : builtin_profiles()) std::printf("built-in: %s\n", b.name.c_str()); for (const auto& n : list_profiles(default_profile_dir())) std::printf("saved:    %s\n", n.c_str()); return 0; }
    if (sub == "show" && argc > 1) { DeviceProfile p; if (!find_profile(argv[1], p, &err)) return fail(err); std::fputs(serialize_profile(p).c_str(), stdout); return 0; }
    if (sub == "new" && argc > 3 && std::strcmp(argv[2], "--from") == 0) { DeviceProfile p; if (!find_profile(argv[3], p, &err)) return fail(err); p.name = argv[1]; std::vector<std::string> pr; if (!profile_problems(p, &pr)) return fail(pr[0]); if (!save_profile(default_profile_dir(), p, &err)) return fail(err); std::printf("saved %s/%s\n", default_profile_dir().c_str(), profile_file_name(p.name).c_str()); return 0; }
    return usage();
}

int main(int argc, char** argv) {
    if (argc < 2) return usage(); auto engine = make_engine(); (void)engine; set_models_dir_near(argv[0]);        // (the engine registers the detectors)
    const std::string cmd = argv[1]; if (cmd == "info") return cmd_info(argc - 2, argv + 2); if (cmd == "export") return cmd_export(argc - 2, argv + 2); if (cmd == "redetect") return cmd_redetect(argc - 2, argv + 2);
    if (cmd == "check") return cmd_check(argc - 2, argv + 2); if (cmd == "profiles") return cmd_profiles(argc - 2, argv + 2); return usage();
}

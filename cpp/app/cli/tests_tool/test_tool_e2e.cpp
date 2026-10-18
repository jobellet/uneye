// eyelive_tool end to end: a session recorded by eyelive_cli, then info / export / re-detection / checks / profiles, with their error paths.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#ifndef _WIN32
#include <sys/wait.h>
#else
#define popen _popen
#define pclose _pclose
#endif
namespace fs = std::filesystem;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
static std::string slurp(const std::string& p) { std::ifstream f(p, std::ios::binary); std::stringstream s; s << f.rdbuf(); return s.str(); }
static size_t lines(const std::string& s) { size_t n = 0; for (char c : s) n += c == '\n'; return n; }
static int run(const std::string& cmd, std::string* out = nullptr) {
    std::FILE* p = popen((cmd + " 2>&1").c_str(), "r"); if (!p) return -1; std::string o; char buf[4096]; size_t n; while ((n = std::fread(buf, 1, sizeof buf, p)) > 0) o.append(buf, n); const int rc = pclose(p); if (out) *out = o;
#ifndef _WIN32
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
#else
    return rc;
#endif
}
int main(int argc, char** argv) {
#ifdef _WIN32
    (void)argc; (void)argv; std::printf("tool e2e: skipped on Windows\n"); return 0;
#else
    if (argc < 3) { std::printf("usage: test_tool_e2e <eyelive_cli> <eyelive_tool> [source dir]\n"); return 2; }
    const std::string cli = "\"" + std::string(argv[1]) + "\"", tool = "\"" + std::string(argv[2]) + "\""; const std::string dir = (fs::temp_directory_path() / "eyelive_tool_e2e").string(); fs::remove_all(dir); fs::create_directories(dir + "/sessions"); const std::string home = dir + "/home"; fs::create_directories(home);
    std::string out; const std::string S = dir + "/sessions/a.eyelive";
    CHECK(run(cli + " --sim --duration 6 --session \"" + S + "\"", &out) == 0, "record: %s", out.c_str());
    CHECK(fs::exists(S + ".provenance.json") && fs::exists(S + ".report.html"), "the session did not leave its journal and report next to the file");
    { const std::string pj = slurp(S + ".provenance.json"); CHECK(pj.find("\"samples\": ") != std::string::npos && pj.find("hmm_accel") != std::string::npos && pj.find("\"engine_version\": \"") != std::string::npos, "journal content: %.300s", pj.c_str()); }
    // info
    long long N = 0; CHECK(run(tool + " info \"" + S + "\"", &out) == 0 && out.find("EyeLive session report") != std::string::npos && std::sscanf(out.c_str() + out.find("samples: ") + 9, "%lld", &N) == 1 && N >= 6000, "info: %s", out.c_str());     // (a simulation that runs as fast as possible overshoots the 6 s asked for)
    CHECK(run(tool + " info \"" + dir + "/nothing.eyelive\"", &out) == 1 && out.find("no such file") != std::string::npos, "info on a missing file (%s)", out.c_str());
    { std::ofstream(dir + "/junk.eyelive") << std::string(5000, 'x'); CHECK(run(tool + " info \"" + dir + "/junk.eyelive\"", &out) == 1, "info on a file that is not a session: %s", out.c_str()); }
    // export
    CHECK(run(tool + " export \"" + S + "\" --csv \"" + dir + "/e.csv\" --bids \"" + dir + "/bids\" --mat \"" + dir + "/x.mat\" --report \"" + dir + "/r.html\"", &out) == 0, "export: %s", out.c_str());
    CHECK(lines(slurp(dir + "/e.csv")) > 5 && slurp(dir + "/e.csv").rfind("index,class,", 0) == 0, "events csv"); CHECK((long long)lines(slurp(dir + "/bids_eyetrack.tsv")) == N + 1 && fs::exists(dir + "/bids_events.json") && fs::file_size(dir + "/x.mat") > (unsigned long long)(N * 8 * 6) && slurp(dir + "/r.html").find("<svg") != std::string::npos, "bids / mat / report files");
    CHECK(run(tool + " export \"" + S + "\"", &out) == 1 && out.find("nothing to export") != std::string::npos, "export without a target"); CHECK(run(tool + " export \"" + S + "\" --csv /nonexistent_dir/x.csv", &out) == 1, "export to an unwritable path");
    // re-detection of a folder: the session file is not modified
    const auto before = fs::file_size(S); const std::string h0 = slurp(S).substr(0, 1 << 16);
    CHECK(run(tool + " redetect \"" + dir + "/sessions\" --detector fast_threshold --out \"" + dir + "/redo\"", &out) == 0 && out.find("1 session(s), 0 failed") != std::string::npos, "redetect: %s", out.c_str());
    CHECK(fs::exists(dir + "/redo/a.fast_threshold.events.csv") && lines(slurp(dir + "/redo/a.fast_threshold.events.csv")) > 5 && fs::exists(dir + "/redo/a.fast_threshold.report.html"), "re-detection outputs"); CHECK(fs::file_size(S) == before && slurp(S).substr(0, 1 << 16) == h0, "re-detection modified the session");
    CHECK(run(tool + " redetect \"" + S + "\" --detector no_such_detector --out \"" + dir + "/redo2\"", &out) == 1 && out.find("unknown detector") != std::string::npos, "unknown detector: %s", out.c_str()); CHECK(run(tool + " redetect \"" + S + "\" --out \"" + dir + "/redo2\"", &out) == 1, "missing --detector");
    CHECK(run(tool + " redetect \"" + dir + "/empty_folder\" --detector hmm_accel --out \"" + dir + "/redo3\"", &out) == 1, "no session found");
    // checks and profiles (HOME points to a temporary folder: nothing of the user is touched)
    const std::string H = "HOME=\"" + home + "\" ";
    CHECK(run(H + tool + " check --profile \"Simulation (demo)\" --session-dir \"" + dir + "/chk\"", &out) == 0 && out.find("[ ok ] profile") != std::string::npos && out.find("disk space") != std::string::npos, "check: %s", out.c_str()); CHECK(run(H + tool + " check --profile nonexistent", &out) == 1 && out.find("no profile") != std::string::npos, "unknown profile: %s", out.c_str());
    CHECK(run(H + tool + " profiles list", &out) == 0 && out.find("built-in: Simulation (demo)") != std::string::npos, "profiles list: %s", out.c_str()); CHECK(run(H + tool + " profiles show \"Simulation (demo)\"", &out) == 0 && out.find("source=sim") != std::string::npos, "profiles show");
    CHECK(run(H + tool + " profiles new \"Lab A\" --from \"Tracker over UDP, screen pixels\"", &out) == 0 && fs::exists(home + "/.eyelive/profiles/Lab_A.profile"), "profiles new: %s", out.c_str());
    CHECK(run(H + tool + " profiles list", &out) == 0 && out.find("saved:    Lab_A") != std::string::npos, "saved profile listed"); CHECK(run(H + tool + " check --profile \"Lab A\" --session-dir \"" + dir + "/chk\"", &out) == 0 && out.find("pixels per degree") != std::string::npos, "check of the saved profile: %s", out.c_str());
    CHECK(run(H + tool + " profiles new X --from nonexistent", &out) == 1, "new from an unknown profile"); CHECK(run(tool, &out) == 2 && out.find("usage") != std::string::npos, "usage without arguments"); CHECK(run(tool + " frobnicate", &out) == 2, "unknown command");
    fs::remove_all(dir); if (fails == 0) std::printf("tool e2e: OK\n"); return fails ? 1 : 0;
#endif
}

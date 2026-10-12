// T20: end-to-end. Each golden trace (cpp/app/dsp/tests/golden_hmm_accel.txt, with the Python reference events) is replayed through UDP into eyelive_cli; the CLI's batch events must match the Python reference ('last' rule, within 1 sample),
// the streaming events of the first part must match too, and the documented examples of cli/README.md run.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
namespace fs = std::filesystem;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

struct Golden { std::string name; double fs; std::vector<double> x, y; std::vector<std::pair<int, int>> last; };
static std::vector<Golden> load_golden(const std::string& root) {
    std::vector<Golden> g; std::ifstream f(root + "/app/dsp/tests/golden_hmm_accel.txt"); int n; f >> n;
    auto events = [&]() { int k; f >> k; std::vector<std::pair<int, int>> e(static_cast<size_t>(k), std::pair<int, int>(0, 0)); for (auto& p : e) f >> p.first >> p.second; return e; };
    for (int c = 0; c < n; ++c) { Golden t; int T; f >> t.name >> t.fs >> T; t.x.resize(size_t(T)); t.y.resize(size_t(T)); for (int i = 0; i < T; ++i) { std::string a, b; f >> a >> b; t.x[size_t(i)] = std::strtod(a.c_str(), nullptr); t.y[size_t(i)] = std::strtod(b.c_str(), nullptr); } events(); events(); t.last = events(); g.push_back(std::move(t)); }
    return g;
}
static int run(const std::string& cmd) { return std::system(cmd.c_str()); }
static std::vector<std::pair<int, int>> read_events_csv(const std::string& path) {
    std::vector<std::pair<int, int>> v; std::ifstream f(path); std::string line; std::getline(f, line);
    while (std::getline(f, line)) { std::stringstream ss(line); std::string tok; std::vector<std::string> t; while (std::getline(ss, tok, ',')) t.push_back(tok); if (t.size() >= 7 && std::atoi(t[2].c_str()) == 1) v.push_back({std::atoi(t[5].c_str()), std::atoi(t[6].c_str())}); }      // saccades only: the engine adds class-2 (blink) events from the invalid samples
    return v;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    (void)argc; (void)argv; std::printf("cli end-to-end test: not run on Windows\n"); return 0;
#else
    if (argc < 3) { std::printf("usage: test_cli_e2e <eyelive_cli> <cpp dir>\n"); return 2; }
    const std::string cli = argv[1], root = argv[2]; const fs::path dir = fs::temp_directory_path() / "eyelive_cli_test"; std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir);
    const std::string models = root + "/models"; auto golden = load_golden(root); CHECK(golden.size() == 5, "golden traces: %zu", golden.size());
    // 1. UDP replay of each golden trace into the CLI
    int port = 41000 + (getpid() % 2000);
    for (size_t k = 0; k < golden.size(); ++k) {
        const Golden& g = golden[k]; const std::string session = (dir / ("g" + std::to_string(k) + ".eyelive")).string(), out = (dir / ("g" + std::to_string(k) + ".csv")).string(), log = (dir / ("g" + std::to_string(k) + ".log")).string();
        const int p = port + int(k);
        const std::string cmd = cli + " --udp " + std::to_string(p) + " --samples " + std::to_string(g.x.size()) + " --duration 30 --detector hmm_accel --rate-hz " + std::to_string(int(g.fs)) + " --session " + session + " --prelabel-seconds 0 --events-out " + out + " --models-dir " + models + " > " + log + " 2>&1 &";
        CHECK(run(cmd) == 0, "launch"); std::this_thread::sleep_for(std::chrono::milliseconds(800));       // the CLI binds its socket
        const int s = ::socket(AF_INET, SOCK_DGRAM, 0); sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(uint16_t(p)); inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
        const size_t per = 20; for (size_t i = 0; i < g.x.size(); i += per) { const size_t m = std::min(per, g.x.size() - i); std::vector<unsigned char> d(8 + 20 * m); const uint32_t magic = 0x31455945u, nn = uint32_t(m); std::memcpy(d.data(), &magic, 4); std::memcpy(d.data() + 4, &nn, 4);
            for (size_t j = 0; j < m; ++j) { const double t = double(i + j) / g.fs; const float x = float(g.x[i + j]), y = float(g.y[i + j]); const uint32_t fl = (std::isfinite(x) && std::isfinite(y)) ? 1u : 2u; unsigned char* q = d.data() + 8 + 20 * j; std::memcpy(q, &t, 8); std::memcpy(q + 8, &x, 4); std::memcpy(q + 12, &y, 4); std::memcpy(q + 16, &fl, 4); }
            ::sendto(s, d.data(), d.size(), 0, reinterpret_cast<sockaddr*>(&a), sizeof a); std::this_thread::sleep_for(std::chrono::microseconds(500)); }
        ::close(s);
        for (int w = 0; w < 200 && !fs::exists(out); ++w) std::this_thread::sleep_for(std::chrono::milliseconds(100));        // the CLI writes the batch events when it has finished
        std::this_thread::sleep_for(std::chrono::milliseconds(300)); const auto got = read_events_csv(out); bool ok = got.size() == g.last.size(); int worst = 0; for (size_t i = 0; ok && i < got.size(); ++i) { worst = std::max({worst, std::abs(got[i].first - g.last[i].first), std::abs(got[i].second - g.last[i].second)}); } ok = ok && worst <= 1;
        std::ifstream lf(log); std::stringstream ls; ls << lf.rdbuf(); const std::string text = ls.str(); const size_t pos = text.find("samples "); std::printf("%-22s CLI batch events %zu, Python reference %zu, worst shift %d | %s\n", g.name.c_str(), got.size(), g.last.size(), worst, pos == std::string::npos ? "(no summary)" : text.substr(pos, text.find('\n', pos) - pos).c_str());
        CHECK(ok, "%s: CLI events differ from the Python reference (log: %s)", g.name.c_str(), text.c_str());
        // streaming events printed while recording must be a subset of the batch events (those far from the end of the 1 s trace)
        int stream_ok = 0, stream_n = 0; std::stringstream ts(text); std::string line; while (std::getline(ts, line)) if (line.rfind("event ", 0) == 0) { ++stream_n; double a1, a2; if (std::sscanf(line.c_str(), "event %lf %lf", &a1, &a2) == 2) for (auto& e : g.last) if (std::abs(int(a1 * g.fs) - e.first) <= 15 && std::abs(int(a2 * g.fs) - e.second) <= 30) { ++stream_ok; break; } }
        CHECK(stream_ok == stream_n, "%s: %d of %d streaming events match the reference", g.name.c_str(), stream_ok, stream_n);
    }
    // 2. documented examples (cli/README.md): simulation, replay of a CSV, triggers file, train
    { const std::string o = (dir / "ex1.log").string(); CHECK(run(cli + " --sim --duration 5 --session " + (dir / "ex1.eyelive").string() + " --models-dir " + models + " > " + o + " 2>&1") == 0, "example 1 failed"); std::ifstream f(o); std::stringstream s; s << f.rdbuf(); CHECK(s.str().find("event ") != std::string::npos && s.str().find("samples ") != std::string::npos, "example 1 printed no event: %s", s.str().c_str()); }
    { const std::string csv = (dir / "trace.csv").string(); { std::ofstream f(csv); f << "t,x,y,valid\n"; for (int i = 0; i < 3000; ++i) { const double x = (i > 1500 ? 3.0 : 0.0) + 0.01 * std::sin(i * 0.37), y = 0.01 * std::cos(i * 0.21); f << i / 1000.0 << ',' << x << ',' << y << ",1\n"; } }
      const std::string o = (dir / "ex2.log").string(); CHECK(run(cli + " --replay " + csv + " --speed 0 --detector fast_threshold --session " + (dir / "ex2.eyelive").string() + " --events-out " + (dir / "ex2.csv").string() + " > " + o + " 2>&1") == 0, "example 2 failed"); const auto ev = read_events_csv((dir / "ex2.csv").string()); std::printf("example 2 (replay, fast_threshold): %zu batch events\n", ev.size()); CHECK(ev.size() == 1, "the replayed step must give one event, got %zu", ev.size()); }
    { const std::string rules = (dir / "rules.txt").string(), logf = (dir / "fired.log").string(); { std::ofstream f(rules); f << "# comment\nname=all;any=1;anchor=final;sink=log:" << logf << ";payload=EV {t} {amp}\n"; }
      const std::string o = (dir / "ex3.log").string(); CHECK(run(cli + " --sim --duration 10 --triggers " + rules + " --session " + (dir / "ex3.eyelive").string() + " --models-dir " + models + " > " + o + " 2>&1") == 0, "example 3 failed"); std::ifstream f(logf); std::string line; int n = 0; while (std::getline(f, line)) n += line.rfind("EV ", 0) == 0; std::printf("example 3 (triggers): %d trigger lines written\n", n); CHECK(n > 0, "no trigger fired"); }
    { const std::string o = (dir / "ex4.log").string(); const int rc = run(cli + " --sim --duration 20 --prelabel-seconds 15 --train --session " + (dir / "ex4.eyelive").string() + " --models-dir " + models + " > " + o + " 2>&1"); std::ifstream f(o); std::stringstream s; s << f.rdbuf(); const std::string t = s.str(); const size_t p = t.find("training:"); std::printf("example 4 (train): %s\n", p == std::string::npos ? t.c_str() : t.substr(p, t.find('\n', p) - p).c_str()); CHECK(rc == 0 && t.find("deployed as detector user:cli") != std::string::npos, "example 4 failed: %s", t.c_str()); }
    CHECK(run(cli + " --bogus > /dev/null 2>&1") != 0, "an unknown option must fail"); CHECK(run(cli + " --sim --session /nonexistent_dir_eyelive/x.eyelive --duration 1 > /dev/null 2>&1") != 0, "a session in a missing directory must fail");
    fs::remove_all(dir, ec); std::printf(fails ? "FAILED\n" : "cli end-to-end tests passed\n"); return fails ? 1 : 0;
#endif
}

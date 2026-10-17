// Profiles, geometry, checks and health.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include "eyelive/setup.hpp"
#ifndef _WIN32
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
using namespace eyelive;
namespace fs = std::filesystem;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
static bool has(const std::vector<std::string>& v, const std::string& part) { for (auto& s : v) if (s.find(part) != std::string::npos) return true; return false; }

int main() {
    const std::string dir = (fs::temp_directory_path() / "eyelive_setup_test").string(); fs::remove_all(dir); fs::create_directories(dir);
    // geometry: 53 cm at 60 cm, 1920 px -> atan(26.5 / 60) = 23.83 deg each side
    { ScreenGeometry g; g.width_cm = 53; g.distance_cm = 60; g.res_x = 1920; const double ppd = pixels_per_degree(g); CHECK(std::fabs(ppd - 1920.0 / (2 * 23.8286)) < 0.01, "ppd %.3f", ppd); CHECK(std::fabs(degrees_per_pixel(g) * ppd - 1.0) < 1e-12, "inverse");
      ScreenGeometry bad; CHECK(pixels_per_degree(bad) == 0 && degrees_per_pixel(bad) == 0, "incomplete geometry"); g.distance_cm = 100; CHECK(pixels_per_degree(g) > ppd, "farther screen: more pixels per degree"); }
    // file names
    CHECK(profile_file_name("Lab A / EyeLink 1000") == "Lab_A_EyeLink_1000.profile", "name: %s", profile_file_name("Lab A / EyeLink 1000").c_str());
    { const std::string n = profile_file_name("../../etc/passwd"); CHECK(n.find('/') == std::string::npos && n.find("..") == std::string::npos && n[0] != '.', "traversal: %s", n.c_str()); CHECK(profile_file_name("") == "profile.profile" && profile_file_name("é$%") == "profile.profile", "empty names"); CHECK(profile_file_name(std::string(500, 'a')).size() <= 90, "long name"); }
    // round trip
    DeviceProfile p; p.name = "Lab A"; p.tracker = "EyeLink 1000 Plus"; p.source.kind = "udp"; p.source.port = 5010; p.source.bind_address = "0.0.0.0"; p.rate_hz = 500; p.units = "pixels"; p.screen.width_cm = 53; p.screen.height_cm = 30; p.screen.distance_cm = 60; p.screen.res_x = 1920; p.screen.res_y = 1080;
    p.detector_id = "fast_threshold"; p.landing_model = "landing_mlp_drone.bin"; p.triggers_file = "rules.txt"; p.duration_min = 45.5; p.notes = "line one\nline two"; p.extra["future_key"] = "kept";
    { DeviceProfile q; std::string err; CHECK(parse_profile(serialize_profile(p), q, &err), "parse: %s", err.c_str()); CHECK(q.name == "Lab A" && q.source.port == 5010 && q.source.bind_address == "0.0.0.0" && q.rate_hz == 500 && q.units == "pixels" && q.screen.res_x == 1920 && q.screen.height_cm == 30 && q.detector_id == "fast_threshold" && q.landing_model == "landing_mlp_drone.bin" && q.duration_min == 45.5 && q.extra["future_key"] == "kept", "round trip");
      CHECK(q.notes == "line one line two", "notes are one line: '%s'", q.notes.c_str()); }
    // parse errors carry the line number
    { DeviceProfile q; std::string err; CHECK(!parse_profile("name=x\nport=abc\n", q, &err) && err == "line 2: bad number for port", "err: %s", err.c_str()); CHECK(!parse_profile("name=x\nnonsense\n", q, &err) && err.find("line 2") == 0, "err: %s", err.c_str());
      CHECK(!parse_profile("rate_hz=1e999\n", q, &err) && err.find("rate_hz") != std::string::npos, "overflow accepted"); CHECK(!parse_profile("port=12.5\n", q, &err), "fractional port accepted"); CHECK(!parse_profile("units=meters\n", q, &err) && err.find("units") != std::string::npos, "bad units accepted"); CHECK(parse_profile("# comment\n\n  name = spaced  \n", q, &err) && q.name == "spaced", "whitespace / comments"); }
    // problems
    { std::vector<std::string> pr; CHECK(profile_problems(p, &pr) && pr.empty(), "valid profile: %s", pr.empty() ? "" : pr[0].c_str());
      DeviceProfile b = p; b.source.port = 70000; CHECK(!profile_problems(b, &pr) && has(pr, "port"), "bad port"); b = p; b.screen.width_cm = 0; CHECK(!profile_problems(b, &pr) && has(pr, "pixels"), "pixels without geometry"); b = p; b.units = "feet"; CHECK(!profile_problems(b, &pr) && has(pr, "units"), "bad units"); b = p; b.source.kind = "serial"; CHECK(!profile_problems(b, &pr) && has(pr, "source"), "bad source");
      b = p; b.duration_min = 0; CHECK(!profile_problems(b, &pr), "zero duration"); b = p; b.screen.res_y = 1920; CHECK(profile_problems(b, &pr) && has(pr, "warning: the pixels are not square"), "aspect warning"); b = p; b.screen.distance_cm = 5; CHECK(profile_problems(b, &pr) && has(pr, "unusual"), "distance warning");
      b = p; b.source.kind = "file"; b.source.path = ""; CHECK(!profile_problems(b, &pr) && has(pr, "file"), "replay without file"); b = p; b.rate_hz = -1; CHECK(!profile_problems(b, &pr), "negative rate"); }
    // save / load / list
    { std::string err; CHECK(save_profile(dir + "/profiles", p, &err), "save: %s", err.c_str()); DeviceProfile b = p; b.name = "B lab"; CHECK(save_profile(dir + "/profiles", b, &err), "save b"); CHECK(save_profile(dir + "/profiles", p, &err), "overwrite");
      const auto names = list_profiles(dir + "/profiles"); CHECK(names.size() == 2 && names[0] == "B_lab" && names[1] == "Lab_A", "list: %zu", names.size()); DeviceProfile q; CHECK(load_profile(dir + "/profiles/Lab_A.profile", q, &err) && q.name == "Lab A", "load"); CHECK(!load_profile(dir + "/nothing.profile", q, &err), "missing profile");
      bool tmp_left = false; for (auto& e : fs::directory_iterator(dir + "/profiles")) tmp_left |= e.path().extension() == ".tmp"; CHECK(!tmp_left, "temporary file left behind"); { std::ofstream(dir + "/big.profile") << std::string((1 << 20) + 5, 'x'); CHECK(!load_profile(dir + "/big.profile", q, &err), "oversized profile accepted"); }
      CHECK(list_profiles(dir + "/does_not_exist").empty(), "listing a missing folder"); CHECK(!save_profile("/proc/nowhere/x", p, &err) && !err.empty(), "unwritable folder"); }
    // session config
    { SessionConfig c = session_config(p, "/tmp/x.eyelive"); const double ppd = pixels_per_degree(p.screen); CHECK(c.source.kind == "udp" && c.source.port == 5010 && c.source.pixels_per_degree == 0 && std::fabs(c.deg_per_unit - 1.0 / ppd) < 1e-12 && c.rate_hz == 500 && std::fabs(c.duration_s - 2730.0) < 1e-9 && c.profile_name == "Lab A" && c.landing_model == "landing_mlp_drone.bin" && c.detector_id == "fast_threshold", "session config");
      DeviceProfile d = p; d.units = "degrees"; CHECK(session_config(d, "x").deg_per_unit == 1.0, "degrees stream"); d.units = "auto"; CHECK(session_config(d, "x").deg_per_unit == 0.0, "auto scale"); }
    CHECK(builtin_profiles().size() >= 4, "builtin profiles"); for (const auto& b : builtin_profiles()) { std::vector<std::string> pr; CHECK(profile_problems(b, &pr) || b.source.kind == "file", "builtin '%s' invalid: %s", b.name.c_str(), pr.empty() ? "" : pr[0].c_str()); }
    // trigger rules
    { TriggerRule r; std::string e; CHECK(parse_trigger_rule("name=sacc;cls=1;anchor=provisional;delay_ms=10;refractory_ms=50;amp_min=0.5;sink=udp:127.0.0.1:6000;payload=SACC {t} {amp}", r, &e), "rule: %s", e.c_str());
      CHECK(r.name == "sacc" && r.cls == 1 && r.anchor == TriggerAnchor::OnsetProvisional && r.delay_ms == 10 && r.refractory_ms == 50 && r.amp_min_deg == 0.5f && r.sink == "udp:127.0.0.1:6000" && r.payload == "SACC {t} {amp}" && r.enabled, "rule fields");
      CHECK(!parse_trigger_rule("name=a;sink=stdout;dely_ms=3", r, &e) && e == "unknown key 'dely_ms'", "typo: %s", e.c_str()); CHECK(!parse_trigger_rule("name=a;sink=stdout;delay_ms=-5", r, &e) && e.find("delay_ms") != std::string::npos, "negative delay"); CHECK(!parse_trigger_rule("name=a;sink=udp:host", r, &e), "udp sink without port");
      CHECK(!parse_trigger_rule("name=a;sink=udp:h:99999", r, &e), "port out of range"); CHECK(!parse_trigger_rule("sink=stdout", r, &e) && e.find("name") != std::string::npos, "no name"); CHECK(!parse_trigger_rule("name=a", r, &e) && e.find("sink") != std::string::npos, "no sink");
      CHECK(!parse_trigger_rule("name=a;sink=ftp:x", r, &e), "bad sink"); CHECK(!parse_trigger_rule("name=a;sink=stdout;cls=9", r, &e), "bad class"); CHECK(!parse_trigger_rule("name=a;sink=stdout;amp_min=3;amp_max=1", r, &e), "amplitude range"); CHECK(!parse_trigger_rule("name=a;sink=stdout;anchor=later", r, &e), "bad anchor");
      CHECK(parse_trigger_rule("name=a;sink=log:/tmp/x.log;enabled=0;any=1", r, &e) && !r.enabled && r.any_class, "log sink, disabled");
      { std::ofstream(dir + "/rules.txt") << "# comment\n\nname=a;sink=stdout\nname=b;sink=udp:127.0.0.1:6001;delay_ms=5\n"; std::vector<TriggerRule> v; CHECK(load_trigger_rules(dir + "/rules.txt", v, &e) && v.size() == 2 && v[1].name == "b", "load rules: %s", e.c_str());
        std::ofstream(dir + "/bad.txt") << "name=a;sink=stdout\nname=b;sink=nope\n"; CHECK(!load_trigger_rules(dir + "/bad.txt", v, &e) && e.find("bad.txt:2:") != std::string::npos, "line number in the error: %s", e.c_str()); CHECK(!load_trigger_rules(dir + "/none.txt", v, &e), "missing rules file"); } }
    // ports
#ifndef _WIN32
    { int s = socket(AF_INET, SOCK_DGRAM, 0); sockaddr_in a{}; a.sin_family = AF_INET; inet_pton(AF_INET, "127.0.0.1", &a.sin_addr); a.sin_port = 0; bind(s, reinterpret_cast<sockaddr*>(&a), sizeof a); socklen_t l = sizeof a; getsockname(s, reinterpret_cast<sockaddr*>(&a), &l); const int port = ntohs(a.sin_port); std::string err;
      CHECK(!udp_port_free("127.0.0.1", port, &err) && err.find("already used") != std::string::npos, "busy port reported free (%s)", err.c_str()); DeviceProfile u; u.source.kind = "udp"; u.source.port = port; u.duration_min = 1;
      auto r = run_checks(u, dir + "/s"); bool fail_port = false; for (auto& c : r) fail_port |= c.what == "network port" && c.level == CheckLevel::Fail && !c.hint.empty(); CHECK(fail_port, "run_checks does not flag the busy port"); close(s);
      CHECK(udp_port_free("127.0.0.1", port, &err), "port still busy after closing: %s", err.c_str()); CHECK(!udp_port_free("not-an-address", 5005, &err) && err.find("IPv4") != std::string::npos, "bad address"); }
#endif
    { DeviceProfile f; f.source.kind = "file"; f.source.path = dir + "/missing.csv"; f.duration_min = 1; bool fail = false; for (auto& c : run_checks(f, dir + "/s")) fail |= c.what == "replay file" && c.level == CheckLevel::Fail; CHECK(fail, "missing replay file not flagged");
      DeviceProfile s = builtin_profiles()[0]; bool all_ok = true; for (auto& c : run_checks(s, dir + "/s2")) all_ok &= c.level != CheckLevel::Fail; CHECK(all_ok, "the simulation profile fails a check"); DeviceProfile big = s; big.duration_min = 24 * 60; big.source.rate_hz = 10000; big.rate_hz = 10000; bool disk_fail = false; if (free_disk_bytes(dir) > 0) for (auto& c : run_checks(big, dir + "/s3")) disk_fail |= c.what == "disk space" && c.level == CheckLevel::Fail && free_disk_bytes(dir) < 10000.0 * 86400 * 19 * 1.575; (void)disk_fail;
      CHECK(free_disk_bytes(dir) > 1e6, "free disk space %.0f", free_disk_bytes(dir)); CHECK(free_disk_bytes("/definitely/not/there") < 0, "unknown folder"); }
    // health light
    { EngineStats s; CHECK(assess_health(s, 0).level == HealthLevel::Idle, "idle"); s.running = true; CHECK(assess_health(s, 0.5).level == HealthLevel::Warn, "first sample pending"); CHECK(assess_health(s, 3).level == HealthLevel::Bad, "no sample after 3 s"); s.samples = 5000; s.rate_hz_measured = 1000; CHECK(assess_health(s, 10).level == HealthLevel::Ok, "healthy");
      s.gaps = 2; CHECK(assess_health(s, 10).level == HealthLevel::Warn, "gaps"); s.gaps = 0; s.dropped_packets = 100; CHECK(assess_health(s, 10).level == HealthLevel::Bad, "2 %% of the packets lost"); s.dropped_packets = 1; CHECK(assess_health(s, 10).level == HealthLevel::Ok, "one lost packet"); s.detect_latency_ms_p95 = 40; CHECK(assess_health(s, 10).level == HealthLevel::Warn, "latency");
      s.detect_latency_ms_p95 = 3; s.capacity_s = 100; s.recorded_s = 99; CHECK(assess_health(s, 10).level == HealthLevel::Bad, "file nearly full"); s.recorded_s = 50; s.warning = "scale unsure"; CHECK(assess_health(s, 10).level == HealthLevel::Warn && assess_health(s, 10).message == "scale unsure", "engine warning"); }
    fs::remove_all(dir); if (fails == 0) std::printf("setup: OK\n"); return fails ? 1 : 0;
}
